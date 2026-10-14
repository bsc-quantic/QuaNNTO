"""Verify archived PRR artifacts without retraining or changing results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from quannto.utils.path_utils import repository_root
from quannto.tasks.trainability.trainability_protocol import (
    build_scan_points, unique_architectures, make_fixed_scalar_dataset,
    parameter_types, sample_initial_parameters, parameter_seed,
)

PRODUCTION_RUN = "bp_scalar_identity_energy_controlled_T64_v4_manuscript_c0739531ce"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hashes(root):
    count = 0
    for line in (root / "provenance/artifacts.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        file = root / relative
        require(file.is_file(), f"Missing archived artifact: {relative}")
        require(hashlib.sha256(file.read_bytes()).hexdigest() == digest,
                f"Modified archived artifact: {relative}")
        count += 1
    return count


def verify_trainability(root):
    run = root / "results/prr/trainability/barren_plateaus" / PRODUCTION_RUN
    protocol = json.loads((run / "protocol.json").read_text())
    complete = json.loads((run / "RUN_COMPLETE.json").read_text())
    require(complete["complete"] and complete["num_completed_unique_configurations"] == 20,
            "Incomplete production trainability campaign")
    d = protocol["scan_design"]
    points = unique_architectures(build_scan_points(
        widths=d["widths"], depths=d["depths"], ladder_orders=d["ladder_orders"],
        energy_scales=d["energy_scales"], fixed_modes_for_depth=d["fixed_modes_for_depth"],
        fixed_modes_for_ladder=d["fixed_modes_for_ladder"], fixed_modes_for_energy=d["fixed_modes_for_energy"],
    ))
    x, y = make_fixed_scalar_dataset(batch_size=32, seed=1234, low=-0.75, high=0.75)
    with np.load(run / "fixed_cost_dataset.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["inputs"], x)
        np.testing.assert_array_equal(z["targets"], y)
    with (run / "bp_probe_metrics.csv").open() as handle:
        probes = list(csv.DictReader(handle))
    require(len(points) == 20 and len(probes) == 60, "Unexpected campaign size")
    for point in points:
        file = run / "checkpoints" / f"{point.architecture_key}_T64.npz"
        with np.load(file, allow_pickle=False) as z:
            types = parameter_types(point.n_modes, point.layers)
            require(z["gradients"].shape == (64, len(types)), f"Invalid gradients: {file.name}")
            for name in z.files:
                if z[name].dtype.kind in "fciu":
                    require(np.isfinite(z[name]).all(), f"Non-finite {name}: {file.name}")
            for trial in range(64):
                rng = np.random.default_rng(parameter_seed(
                    base_seed=2026, ensemble="energy_controlled", n_modes=point.n_modes,
                    layers=point.layers, trial=trial,
                ))
                expected = sample_initial_parameters(
                    rng=rng, types=types, layers=point.layers, ensemble="energy_controlled",
                    energy_scale=point.energy_scale, sigma_passive=0.1,
                    sigma_squeezing=0.05, sigma_displacement=0.05,
                )
                np.testing.assert_array_equal(z["parameters"][trial], expected)
            selected = [p for p in probes if p["architecture_key"] == point.architecture_key]
            require(len(selected) == 3, f"Missing probes: {file.name}")
            for row in selected:
                value = np.var(z["gradients"][:, int(row["parameter_index"])], ddof=1)
                np.testing.assert_allclose(value, float(row["gradient_variance"]), rtol=1e-13, atol=1e-15)
    with (run / "bp_gradcheck.csv").open() as handle:
        checks = list(csv.DictReader(handle))
    require(len(checks) == 16 and all(r["passed"] == "True" for r in checks), "Recorded gradchecks failed")
    return dict(configurations=20, initializations=64, derivative_variances=60,
                recorded_gradchecks=16, max_recorded_gradient_error=max(float(r["absolute_error"]) for r in checks))


def verify_quantum_checkpoints(root):
    from quannto.tasks.quantum_tables import audit_model, load_task_data, load_numeric
    manifest = json.loads((root / "quannto/tasks/quantum_tables_manifest.json").read_text())
    specs = [s for s in manifest["models"] if s.get("included_in_paper", True)]
    require(len(specs) == 13, "Expected five cubic and eight cat checkpoints")
    data = {task: load_task_data(task, definition, root / "data/datasets")
            for task, definition in manifest["tasks"].items()}
    for spec in specs:
        row, _ = audit_model(spec, root / "results/models", data[spec["task"]], manifest["tasks"][spec["task"]])
        require(row["status"] == "ready", f"Checkpoint audit failed: {spec['id']}: {row.get('error')}")
        predictions = load_numeric(root / "results/models/testing_results" / (spec["checkpoint_stem"] + ".txt"))
        targets = data[spec["task"]]["targets"]
        require(predictions.shape == targets.shape, f"Invalid saved predictions: {spec['id']}")
        saved_loss = np.mean(np.sum(np.abs(predictions - targets)**2, axis=-1))
        np.testing.assert_allclose(saved_loss, row["recorded_selected_loss"], rtol=1e-10, atol=1e-10)
    return dict(cubic=5, cat=8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repository_root())
    args = parser.parse_args(argv)
    if args.repo is None:
        parser.error("Provide --repo or set QUANNTO_REPO_ROOT")
    root = args.repo.resolve()
    print(json.dumps(dict(artifact_hashes=verify_hashes(root), trainability=verify_trainability(root),
                          quantum_checkpoints=verify_quantum_checkpoints(root)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
