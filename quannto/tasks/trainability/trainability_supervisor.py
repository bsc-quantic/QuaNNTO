"""Failure-tolerant supervisor for the finite-size trainability protocol.

This module intentionally imports neither JAX nor :class:`QNN`.  The parent
process therefore remains small while one fresh subprocess owns each compiled
architecture.  A kernel OOM kill or a Python exception can remove one worker
without discarding later scan points or already completed trials.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from quannto.tasks.trainability.trainability_protocol import (
    EMPIRICALLY_INACTIVE_TOLERANCE,
    PROTOCOL_VERSION,
    SWEEPS,
    ScanPoint,
    build_scan_points,
    canonical_json,
    dataset_sha256,
    make_fixed_scalar_dataset,
    matched_probe_indices,
    parameter_diagnostic_rows,
    parameter_types,
    parse_csv_numbers,
    unique_architectures,
)
from quannto.utils.path_utils import quannto_results_dir


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def point_dict(point: ScanPoint) -> dict[str, Any]:
    return {
        "sweep": point.sweep,
        "family": point.family,
        "size": point.size,
        "n_modes": point.n_modes,
        "layers": point.layers,
        "k_subtractions": point.k_subtractions,
        "energy_scale": point.energy_scale,
    }


def classify_worker_outcome(
    returncode: int, *, timed_out: bool = False,
) -> dict[str, Any]:
    """Classify a worker exit without pretending every SIGKILL is an OOM."""

    if timed_out:
        return {
            "worker_status": "failed",
            "failure_kind": "timeout",
            "returncode": int(returncode),
            "signal": None,
            "probable_oom": False,
        }
    signal_number = -returncode if returncode < 0 else None
    signal_name = None
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIGNAL_{signal_number}"
    probable_oom = returncode in {-signal.SIGKILL, 137}
    return {
        "worker_status": "completed" if returncode == 0 else "failed",
        "failure_kind": None if returncode == 0 else (
            "probable_oom_or_external_sigkill" if probable_oom
            else "worker_error"
        ),
        "returncode": int(returncode),
        "signal": signal_name,
        "probable_oom": bool(probable_oom),
    }


def run_worker_subprocess(
    request_path: Path,
    *,
    log_path: Path,
    timeout_minutes: float,
) -> dict[str, Any]:
    """Run one architecture in its own process group and return its outcome."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "quannto.tasks.trainability.trainability",
        "--worker-request",
        str(request_path.resolve()),
    ]
    started = time.monotonic()
    timed_out = False
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== worker start {utc_now()} =====\n")
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                    "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
                ),
            },
        )
        try:
            returncode = process.wait(
                timeout=None if timeout_minutes <= 0 else 60.0 * timeout_minutes
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        log.write(
            f"\n===== worker end {utc_now()} returncode={returncode} =====\n"
        )
    outcome = classify_worker_outcome(returncode, timed_out=timed_out)
    outcome.update({
        "duration_seconds": float(time.monotonic() - started),
        "log_path": str(log_path.resolve()),
    })
    return outcome


def run_worker_request(runtime: Any, request_path: Path) -> None:
    """Execute one serialized configuration inside the worker process."""

    with request_path.open(encoding="utf-8") as stream:
        request = json.load(stream)
    if int(request["protocol_version"]) != PROTOCOL_VERSION:
        raise ValueError("worker request uses a different protocol version")
    point = ScanPoint(**request["point"])
    output_directory = Path(request["output_directory"])
    status_path = (
        output_directory / "worker_status" / f"{point.architecture_key}.json"
    )
    started = time.monotonic()
    runtime._write_json(status_path, {
        "architecture_key": point.architecture_key,
        "status": "running",
        "started_utc": utc_now(),
        "request_path": str(request_path.resolve()),
    })
    try:
        with np.load(request["dataset_path"], allow_pickle=False) as archive:
            inputs = np.asarray(archive["inputs"], dtype=np.float64)
            targets = np.asarray(archive["targets"], dtype=np.float64)
        found_hash = dataset_sha256(inputs, targets)
        if found_hash != request["dataset_hash"]:
            raise ValueError("worker dataset hash does not match its request")
        runtime._run_configuration(
            point,
            ensemble=request["ensemble"],
            trials=int(request["trials"]),
            inputs=inputs,
            targets=targets,
            dataset_hash=request["dataset_hash"],
            parameter_seed_value=int(request["parameter_seed"]),
            expectation_backend=request["expectation_backend"],
            output_directory=output_directory,
            resume=bool(request["resume"]),
            overwrite=bool(request["overwrite"]),
        )
    except Exception as error:
        runtime._write_json(status_path, {
            "architecture_key": point.architecture_key,
            "status": "failed",
            "finished_utc": utc_now(),
            "duration_seconds": float(time.monotonic() - started),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise
    runtime._write_json(status_path, {
        "architecture_key": point.architecture_key,
        "status": "completed",
        "finished_utc": utc_now(),
        "duration_seconds": float(time.monotonic() - started),
    })


def _optional_csv(runtime: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if rows:
        runtime._write_csv(path, rows)
    else:
        path.unlink(missing_ok=True)


def _completed_partial_trials(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=False) as archive:
            return int(archive["completed_trials"].item())
    except Exception:
        return 0


def aggregate_results(
    runtime: Any,
    *,
    all_points: Sequence[ScanPoint],
    requested_keys: set[str],
    worker_outcomes: Mapping[str, Mapping[str, Any]],
    ensemble: str,
    trials: int,
    dataset_hash_value: str,
    parameter_seed_value: int,
    expectation_backend: str,
    bootstrap_repetitions: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Validate every checkpoint and rebuild all tables/figures from successes."""

    representatives = {
        point.architecture_key: point for point in unique_architectures(all_points)
    }
    raw_by_key: dict[str, dict[str, np.ndarray]] = {}
    status_rows: list[dict[str, Any]] = []
    for key, point in representatives.items():
        signature = runtime._configuration_signature(
            point,
            ensemble=ensemble,
            trials=trials,
            dataset_hash=dataset_hash_value,
            parameter_seed_value=parameter_seed_value,
            expectation_backend=expectation_backend,
        )
        checkpoint = runtime._checkpoint_path(
            output_directory, point, trials=trials,
        )
        partial = runtime._partial_checkpoint_path(
            output_directory, point, trials=trials,
        )
        validation_error = None
        if checkpoint.exists():
            try:
                raw_by_key[key] = runtime._load_checkpoint(
                    checkpoint,
                    expected_signature=signature,
                    expected_parameters=len(
                        parameter_types(point.n_modes, point.layers)
                    ),
                )
            except Exception as error:
                validation_error = f"{type(error).__name__}: {error}"
        worker = dict(worker_outcomes.get(key, {}))
        previous_supervisor_status = (
            output_directory / "supervisor_status" / f"{key}.json"
        )
        if not worker and previous_supervisor_status.exists():
            try:
                with previous_supervisor_status.open(encoding="utf-8") as stream:
                    worker = json.load(stream)
            except (OSError, ValueError, TypeError):
                worker = {}
        if key in raw_by_key:
            status = "completed"
        elif validation_error is not None:
            status = "invalid_checkpoint"
        elif worker.get("worker_status") == "failed":
            status = "failed"
        elif partial.exists():
            status = "partial"
        else:
            status = "missing" if key in requested_keys else "not_requested"
        status_rows.append({
            "architecture_key": key,
            "N": point.n_modes,
            "L": point.layers,
            "K": point.k_subtractions,
            "energy_scale": point.energy_scale,
            "requested_this_invocation": key in requested_keys,
            "status": status,
            "completed_trials": (
                trials if status == "completed"
                else _completed_partial_trials(partial)
            ),
            "checkpoint_path": str(checkpoint.resolve()),
            "partial_checkpoint_path": str(partial.resolve()),
            "checkpoint_validation_error": validation_error,
            **worker,
        })

    summaries: dict[str, dict[str, Any]] = {}
    for index, (key, raw) in enumerate(raw_by_key.items()):
        summaries[key] = runtime._configuration_summary(
            representatives[key],
            raw,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=9347 + index,
        )

    summary_rows: list[dict[str, Any]] = []
    for point in all_points:
        if point.architecture_key not in summaries:
            continue
        row: dict[str, Any] = {
            "sweep": point.sweep,
            "family": point.family,
            "size": point.size,
            "architecture_key": point.architecture_key,
            "ensemble": ensemble,
            "N": point.n_modes,
            "L": point.layers,
            "K": point.k_subtractions,
            "energy_scale": point.energy_scale,
            "ladder_modes_zero_indexed": str(point.ladder_modes),
        }
        row.update(summaries[point.architecture_key])
        summary_rows.append(row)

    trial_rows: list[dict[str, Any]] = []
    gradcheck_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for key, raw in raw_by_key.items():
        point = representatives[key]
        trial_rows.extend(runtime._trial_rows(point, raw, ensemble=ensemble))
        gradcheck_rows.extend(
            runtime._gradcheck_rows(point, raw, ensemble=ensemble)
        )
        probes = matched_probe_indices(point.n_modes)
        inverse_probes = {index: name for name, index in probes.items()}
        for diagnostic in parameter_diagnostic_rows(
            raw["gradients"],
            types=parameter_types(point.n_modes, point.layers),
        ):
            diagnostic.update({
                "architecture_key": key,
                "ensemble": ensemble,
                "N": point.n_modes,
                "L": point.layers,
                "K": point.k_subtractions,
                "energy_scale": point.energy_scale,
                "matched_probe": inverse_probes.get(
                    diagnostic["parameter_index"], ""
                ),
            })
            parameter_rows.append(diagnostic)
        summary = summaries[key]
        for kind in ("passive", "squeezing", "displacement"):
            metric = f"probe_{kind}__gradient_variance"
            probe_rows.append({
                "architecture_key": key,
                "ensemble": ensemble,
                "N": point.n_modes,
                "L": point.layers,
                "K": point.k_subtractions,
                "energy_scale": point.energy_scale,
                "probe_family": kind,
                "parameter_index": summary[
                    f"probe_{kind}__parameter_index"
                ],
                "gradient_variance": summary[metric],
                "gradient_variance_ci95_low": summary[f"{metric}_ci95_low"],
                "gradient_variance_ci95_high": summary[f"{metric}_ci95_high"],
                "mean_gradient": summary[f"probe_{kind}__mean_gradient"],
                "gradient_second_moment": summary[
                    f"probe_{kind}__gradient_second_moment"
                ],
            })

    runtime._write_csv(output_directory / "configuration_status.csv", status_rows)
    _optional_csv(runtime, output_directory / "bp_metrics.csv", summary_rows)
    _optional_csv(
        runtime, output_directory / "bp_trial_metrics.csv", trial_rows,
    )
    _optional_csv(
        runtime, output_directory / "bp_parameter_metrics.csv", parameter_rows,
    )
    _optional_csv(
        runtime, output_directory / "bp_probe_metrics.csv", probe_rows,
    )
    _optional_csv(runtime, output_directory / "bp_gradcheck.csv", gradcheck_rows)
    runtime._write_json(
        output_directory / "bp_fit_summary.json",
        runtime._fit_sweeps(summary_rows),
    )
    if summary_rows:
        runtime._plot_gradient_variance(summary_rows, output_directory)
        runtime._plot_nongaussian_and_energy(summary_rows, output_directory)
        runtime._plot_parameter_families(summary_rows, output_directory)
        runtime._plot_aggregate_diagnostics(summary_rows, output_directory)
        runtime._plot_physical_controls(summary_rows, output_directory)

    completed_keys = set(raw_by_key)
    designed_keys = set(representatives)
    requested_complete = requested_keys <= completed_keys
    design_complete = designed_keys <= completed_keys
    requested_failures = [
        row for row in status_rows
        if row["requested_this_invocation"]
        and row["status"] in {"failed", "invalid_checkpoint"}
    ]
    run_status = {
        "protocol_version": PROTOCOL_VERSION,
        "updated_utc": utc_now(),
        "ensemble": ensemble,
        "trials": trials,
        "requested_complete": requested_complete,
        "design_complete": design_complete,
        "num_designed_unique_configurations": len(designed_keys),
        "num_completed_unique_configurations": len(completed_keys),
        "num_requested_unique_configurations": len(requested_keys),
        "num_requested_failures": len(requested_failures),
        "num_probable_oom_or_sigkill": sum(
            bool(row.get("probable_oom")) for row in status_rows
        ),
        "num_timeouts": sum(
            row.get("failure_kind") == "timeout" for row in status_rows
        ),
        "num_partial_configurations": sum(
            row["status"] == "partial" for row in status_rows
        ),
        "num_plotted_scan_points_available": len(summary_rows),
        "all_recorded_gradchecks_passed": (
            None if not gradcheck_rows
            else all(row["passed"] for row in gradcheck_rows)
        ),
        "failed_requested_architectures": [
            row["architecture_key"] for row in requested_failures
        ],
        "incomplete_designed_architectures": sorted(
            designed_keys - completed_keys
        ),
        "status_table": "configuration_status.csv",
    }
    runtime._write_json(output_directory / "RUN_STATUS.json", run_status)
    complete_path = output_directory / "RUN_COMPLETE.json"
    if design_complete:
        runtime._write_json(complete_path, {
            **run_status,
            "complete": True,
            "outputs": [
                "protocol.json",
                "fixed_cost_dataset.npz",
                "bp_metrics.csv",
                "bp_trial_metrics.csv",
                "bp_parameter_metrics.csv",
                "bp_probe_metrics.csv",
                "bp_gradcheck.csv",
                "bp_fit_summary.json",
                "configuration_status.csv",
                "paper_style_gradient_variance.pdf",
                "non_gaussian_energy_gradient_variance.pdf",
                "aggregate_gradient_diagnostics.pdf",
                "parameter_family_gradient_variance.pdf",
                "physical_control_diagnostics.pdf",
            ],
        })
    else:
        complete_path.unlink(missing_ok=True)
    return run_status


def resolved_design(args: Any, scan_profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    profile = scan_profiles[args.profile]
    design = {
        "profile": args.profile,
        "widths": tuple(
            profile["widths"] if args.widths is None
            else parse_csv_numbers(args.widths, cast=int)
        ),
        "depths": tuple(
            profile["depths"] if args.depths is None
            else parse_csv_numbers(args.depths, cast=int)
        ),
        "ladder_orders": tuple(
            profile["ladder_orders"] if args.ladder_orders is None
            else parse_csv_numbers(args.ladder_orders, cast=int)
        ),
        "energy_scales": tuple(
            profile["energy_scales"] if args.energy_scales is None
            else parse_csv_numbers(args.energy_scales, cast=float)
        ),
        "fixed_modes_for_depth": int(
            profile["fixed_modes_for_depth"]
            if args.fixed_modes_for_depth is None else args.fixed_modes_for_depth
        ),
        "fixed_modes_for_ladder": int(
            profile["fixed_modes_for_ladder"]
            if args.fixed_modes_for_ladder is None else args.fixed_modes_for_ladder
        ),
        "fixed_modes_for_energy": int(
            profile["fixed_modes_for_energy"]
            if args.fixed_modes_for_energy is None else args.fixed_modes_for_energy
        ),
    }
    if args.quick:
        design["widths"] = tuple(n for n in design["widths"] if n <= 3) or (2,)
        design["depths"] = tuple(l for l in design["depths"] if l <= 2) or (1,)
        design["ladder_orders"] = tuple(
            k for k in design["ladder_orders"] if k <= 2
        )
        design["energy_scales"] = tuple(
            e for e in design["energy_scales"] if e <= 1.0
        )
    return design


def run_supervised(runtime: Any, args: Any) -> Path:
    sweeps = tuple(
        SWEEPS if args.sweeps == "all" else args.sweeps.split(",")
    )
    if args.ensemble == "legacy_uniform" and "energy" in sweeps:
        sweeps = tuple(sweep for sweep in sweeps if sweep != "energy")
        print(
            "[protocol] Skipping the energy sweep for legacy_uniform; "
            "rescaling it would define a different ensemble."
        )
    design = resolved_design(args, runtime.SCAN_PROFILES)
    trials = min(int(args.trials), 3) if args.quick else int(args.trials)
    bootstrap_repetitions = (
        min(int(args.bootstrap_repetitions), 100)
        if args.quick else int(args.bootstrap_repetitions)
    )
    batch_size = (
        min(int(args.batch_size), 8) if args.quick else int(args.batch_size)
    )
    if trials < 2:
        raise ValueError("at least two initialization trials are required")

    point_kwargs = {
        key: design[key] for key in (
            "widths",
            "depths",
            "ladder_orders",
            "energy_scales",
            "fixed_modes_for_depth",
            "fixed_modes_for_ladder",
            "fixed_modes_for_energy",
        )
    }
    all_points = build_scan_points(
        sweeps=SWEEPS, ensemble=args.ensemble, **point_kwargs,
    )
    requested_points = build_scan_points(
        sweeps=sweeps, ensemble=args.ensemble, **point_kwargs,
    )
    if not requested_points:
        raise ValueError("the selected protocol contains no scan points")

    inputs, targets = make_fixed_scalar_dataset(
        batch_size=batch_size,
        low=runtime.INPUT_LOW,
        high=runtime.INPUT_HIGH,
        seed=args.dataset_seed,
    )
    data_hash = dataset_sha256(inputs, targets)
    design_identity = {
        **design,
        "batch_size": batch_size,
        "dataset_seed": int(args.dataset_seed),
        "parameter_seed": int(args.parameter_seed),
        "ensemble": args.ensemble,
        "expectation_backend": args.expectation_backend,
        "bootstrap_repetitions": bootstrap_repetitions,
    }
    design_digest = hashlib.sha256(
        canonical_json(design_identity).encode("utf-8")
    ).hexdigest()[:10]
    backend_suffix = (
        "" if args.expectation_backend == "loop_matchings" else "_subset_dp"
    )
    smoke_suffix = "_smoke" if args.quick else ""
    run_name = (
        f"bp_scalar_identity_{args.ensemble}_T{trials}_v{PROTOCOL_VERSION}_"
        f"{args.profile}_{design_digest}{backend_suffix}{smoke_suffix}"
    )
    output_directory = quannto_results_dir() / "barren_plateaus" / run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    supervisor_lock = (output_directory / ".supervisor.lock").open("a+")
    try:
        import fcntl
        fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        pass
    except BlockingIOError as error:
        supervisor_lock.close()
        raise RuntimeError(
            f"another trainability supervisor is using {output_directory}"
        ) from error

    max_modes = max(point.n_modes for point in all_points)
    max_layers = max(point.layers for point in all_points)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "run_name": run_name,
        "design_hash": design_digest,
        "scan_design": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in design.items()
        },
        "finite_size_interpretation": (
            f"This diagnostic probes only the accessible N<={max_modes} and "
            f"L<={max_layers} regime. It neither establishes nor excludes an "
            "asymptotic barren plateau."
        ),
        "cost_specificity": (
            "Every conclusion is specific to the fixed scalar homodyne MSE "
            "cost and the recorded initialization ensemble."
        ),
        "width_resource_caveat": (
            "The scalar input carries fixed total displacement energy, not "
            "fixed energy per mode, while initialized trainable operations can "
            "add size-dependent energy. Measured total and per-mode core "
            "energies are therefore reported, and the width scan is not a "
            "direct test of a fixed-per-mode-energy theorem."
        ),
        "literature_context": {
            "reference": (
                "B. Zhang and Q. Zhuang, Energy-dependent barren plateau in "
                "bosonic variational quantum circuits, arXiv:2305.01799; "
                "Quantum Sci. Technol. 10, 015009 (2025)."
            ),
            "scope_caveat": (
                "That result treats different bosonic variational ansatze and "
                "costs. It is context, not a theorem asserted for this "
                "conditionally normalized Gaussian-plus-ladder QONN."
            ),
        },
        "dataset": {
            "task": "scalar identity regression x -> x",
            "batch_size": batch_size,
            "input_interval": [runtime.INPUT_LOW, runtime.INPUT_HIGH],
            "endpoints_included": True,
            "dataset_seed": args.dataset_seed,
            "sha256": data_hash,
            "fixed_across_all_parameter_trials_and_scan_points": True,
        },
        "circuit": {
            "observable": runtime.OBSERVABLE,
            "is_addition": runtime.IS_ADDITION,
            "active_gaussian_layers": True,
            "initial_squeezing": runtime.INCLUDE_INITIAL_SQUEEZING,
            "initial_mixing": runtime.INCLUDE_INITIAL_MIXING,
            "expectation_backend": args.expectation_backend,
        },
        "ensemble": args.ensemble,
        "trials": trials,
        "parameter_seed": args.parameter_seed,
        "paired_design": (
            "K and energy scale are excluded from the parameter seed, so "
            "fixed-N,L comparisons reuse the underlying random draws."
        ),
        "initialization": {
            "energy_controlled": {
                "passive": (
                    f"Normal(0,{runtime.SIGMA_PASSIVE}^2), independent of L"
                ),
                "squeezing": (
                    f"Normal(0,({runtime.SIGMA_SQUEEZING}*"
                    "energy_scale/sqrt(L))^2)"
                ),
                "displacement": (
                    f"Normal(0,({runtime.SIGMA_DISPLACEMENT}*"
                    "energy_scale/sqrt(L))^2)"
                ),
            },
            "legacy_uniform": (
                "Independent U[0,1) parameters, matching the historical "
                "initialization distribution, not a selected old vector."
            ),
        },
        "co_primary_metrics": [
            "Var_theta[dC/dtheta_i] for the matched passive coordinate",
            "Var_theta[dC/dtheta_i] for the matched squeezing coordinate",
            "Var_theta[dC/dtheta_i] for the matched displacement coordinate",
        ],
        "co_primary_metric_rationale": (
            "Each derivative coordinate is fixed before sampling and occupies "
            "the same relative first-layer Bloch-Messiah block across N and L."
        ),
        "secondary_metrics": [
            "median and mean parameterwise variance",
            "empirically inactive-coordinate count and fraction",
            "mean squared gradient per parameter and total gradient norm",
            "complete one-row-per-parameter diagnostics",
            "Gaussian-core photons per mode and in total",
            "heralding norm and tiny/nonpositive-norm fractions",
        ],
        "inactive_coordinate_definition": {
            "criterion": "max_trial_abs_gradient <= tolerance",
            "tolerance": EMPIRICALLY_INACTIVE_TOLERANCE,
            "interpretation": (
                "Numerical diagnostic only; it is reported because pooling a "
                "size-dependent number of inactive coordinates can bias global "
                "median summaries."
            ),
        },
        "bootstrap": {
            "confidence": 0.95,
            "repetitions": bootstrap_repetitions,
            "resampling_unit": "initialization trial",
        },
        "execution": {
            "default_isolation": "one subprocess per unique architecture",
            "partial_checkpoint_frequency": "after every completed trial",
            "failure_policy": (
                "A failed, timed-out, or OOM-killed architecture is recorded "
                "and does not prevent later architectures from running."
            ),
        },
    }
    protocol_path = output_directory / "protocol.json"
    if protocol_path.exists():
        with protocol_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if canonical_json(existing) != canonical_json(protocol):
            raise ValueError(
                f"existing protocol differs in {protocol_path}; use a different "
                "QUANNTO_RESULTS_DIR or an explicit new scan design"
            )
    else:
        runtime._write_json(protocol_path, protocol)

    dataset_path = output_directory / "fixed_cost_dataset.npz"
    runtime._write_npz_atomic(
        dataset_path, {"inputs": inputs, "targets": targets},
    )
    requested_representatives = {
        point.architecture_key: point
        for point in unique_architectures(requested_points)
    }
    runtime._write_json(output_directory / "last_requested_run.json", {
        "requested_utc": utc_now(),
        "sweeps": list(sweeps),
        "requested_architecture_keys": list(requested_representatives),
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
        "aggregate_only": bool(args.aggregate_only),
        "isolation": args.isolation,
        "quick": bool(args.quick),
    })

    worker_outcomes: dict[str, dict[str, Any]] = {}
    if not args.aggregate_only:
        total = len(requested_representatives)
        for index, (key, point) in enumerate(
            requested_representatives.items(), 1,
        ):
            print(f"\n[configuration {index}/{total}] {key}")
            request = {
                "protocol_version": PROTOCOL_VERSION,
                "point": point_dict(point),
                "ensemble": args.ensemble,
                "trials": trials,
                "dataset_path": str(dataset_path.resolve()),
                "dataset_hash": data_hash,
                "parameter_seed": int(args.parameter_seed),
                "expectation_backend": args.expectation_backend,
                "output_directory": str(output_directory.resolve()),
                "resume": bool(args.resume),
                "overwrite": bool(args.overwrite),
            }
            request_path = (
                output_directory / "worker_requests" / f"{key}.json"
            )
            runtime._write_json(request_path, request)
            if args.isolation == "process":
                outcome = run_worker_subprocess(
                    request_path,
                    log_path=(
                        output_directory / "configuration_logs" / f"{key}.log"
                    ),
                    timeout_minutes=float(args.configuration_timeout_minutes),
                )
            else:
                started = time.monotonic()
                try:
                    run_worker_request(runtime, request_path)
                    outcome = classify_worker_outcome(0)
                except Exception as error:
                    traceback.print_exc()
                    outcome = classify_worker_outcome(1)
                    outcome["error"] = f"{type(error).__name__}: {error}"
                outcome["duration_seconds"] = float(
                    time.monotonic() - started
                )
                outcome["log_path"] = "stdout/stderr (isolation=none)"
            worker_outcomes[key] = outcome
            runtime._write_json(
                output_directory / "supervisor_status" / f"{key}.json",
                {"architecture_key": key, "updated_utc": utc_now(), **outcome},
            )
            if outcome["worker_status"] == "failed":
                print(
                    f"[failed but continuing] {key}: "
                    f"{outcome.get('failure_kind')} "
                    f"(see {outcome.get('log_path')})"
                )

    run_status = aggregate_results(
        runtime,
        all_points=all_points,
        requested_keys=set(requested_representatives),
        worker_outcomes=worker_outcomes,
        ensemble=args.ensemble,
        trials=trials,
        dataset_hash_value=data_hash,
        parameter_seed_value=args.parameter_seed,
        expectation_backend=args.expectation_backend,
        bootstrap_repetitions=bootstrap_repetitions,
        output_directory=output_directory,
    )
    state = "complete" if run_status["design_complete"] else "partial"
    print(f"\nSaved {state} trainability results to {output_directory.resolve()}")
    print(
        f"Completed {run_status['num_completed_unique_configurations']}/"
        f"{run_status['num_designed_unique_configurations']} designed "
        f"configurations; requested failures: "
        f"{run_status['num_requested_failures']}."
    )
    if run_status["num_requested_failures"] and not args.allow_partial:
        raise RuntimeError(
            "all requested configurations were attempted, but at least one "
            "failed; inspect RUN_STATUS.json/configuration_status.csv and "
            "rerun with --resume"
        )
    return output_directory
