"""Audit/evaluate the saved models used in the manuscript quantum-application tables; never train.

python -m quannto.tasks.quantum_tables --audit --output quantum_evaluation
python -m quannto.tasks.quantum_tables --task cat --output quantum_evaluation
"""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

from quannto.utils.path_utils import quannto_results_dir, datasets_dir
from quannto.utils.strawberryfields.quantum_state_evaluation import (
    validate_layers, operators_from_parameters, projected_qonn_ket,
    reduced_output, output_moments, reference_ket, output_fidelity,
)


def load_numeric(path):
    """Read NPY by magic bytes (including NPY files named .txt), or plain text."""
    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(6)
    array = np.load(path, allow_pickle=False) if magic == b"\x93NUMPY" else np.loadtxt(path)
    array = np.asarray(array)
    if array.dtype.kind not in "fciu" or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite numeric array: {path}")
    return array


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _real(array, label):
    if np.max(np.abs(array.imag), initial=0) > 1e-12:
        raise ValueError(f"{label} must be real")
    return array.real


def _file_info(path):
    return {"path": str(Path(path).resolve()), "sha256": _sha(path)}


def _stem(value):
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Checkpoint names must be basenames, not paths")
    return value


def load_checkpoint(spec, models_root):
    """Use only explicitly named operators/parameters, checking both if available."""
    stem = _stem(spec["checkpoint_stem"])
    S_path = models_root / "trained_operators" / (stem + "_S_l.npy")
    d_path = models_root / "trained_operators" / (stem + "_D_l.npy")
    parameter_paths = [models_root / "params" / (stem + ext) for ext in (".npy", ".txt")]
    parameter_paths = [p for p in parameter_paths if p.exists()]
    requested_source = spec.get("checkpoint_source", "auto")
    if requested_source not in ("auto", "operators", "parameters"):
        raise ValueError("checkpoint_source must be auto, operators or parameters")
    # Explicit source choices allow evaluation of legacy operator exports even
    # when companion parameters used a different historical parameterization.
    if requested_source == "operators":
        parameter_paths = []
    N, L = spec["N"], len(spec["subtractions"])
    files = []
    decoded = None
    if parameter_paths:
        parameters = load_numeric(parameter_paths[0]).reshape(-1)
        for other in parameter_paths[1:]:
            second = load_numeric(other).reshape(-1)
            if parameters.shape != second.shape or not np.allclose(parameters, second, atol=1e-10, rtol=1e-10):
                raise ValueError("Conflicting .npy/.txt parameter copies; choose the intended checkpoint explicitly")
        decoded = operators_from_parameters(parameters, N, L)
        files.extend(_file_info(p) for p in parameter_paths)
    if requested_source != "parameters" and S_path.exists() != d_path.exists():
        raise ValueError(f"Incomplete operator pair: both {S_path.name} and {d_path.name} are needed")
    if requested_source != "parameters" and S_path.exists():
        S, d = load_numeric(S_path), load_numeric(d_path)
        S, d = validate_layers(S, d, spec["subtractions"])
        if S.shape != (L, 2*N, 2*N):
            raise ValueError("Saved matrices disagree with the figure architecture")
        if decoded is not None and (not np.allclose(S, decoded[0], rtol=1e-8, atol=1e-9)
                                    or not np.allclose(d, decoded[1], rtol=1e-8, atol=1e-9)):
            raise ValueError("Saved parameters and operators describe different checkpoints")
        files.extend([_file_info(S_path), _file_info(d_path)])
        source = "operators_and_parameters" if decoded is not None else "operators"
    elif decoded is not None:
        S, d = decoded
        source = "parameters_decoded_without_training"
    else:
        raise FileNotFoundError(
            f"Missing checkpoint: supply trained_operators/{stem}_S_l.npy and {stem}_D_l.npy, "
            f"or params/{stem}.npy (or .txt). Losses and predicted moments cannot reconstruct a state.")
    return S, d, files, source



def load_initial_operators(spec, models_root):
    """Reconstruct the exact random initialization used by the saved run.

    The baseline is paired to the trained checkpoint: parameters are read from
    the run's immutable training record rather than regenerated from a seed.
    This avoids dependence on RNG implementation details and guarantees that
    ``F_init`` refers to the state from which that optimization actually began.
    """
    stem = _stem(spec["checkpoint_stem"])
    record_path = models_root / "training_records" / (stem + ".json")
    if not record_path.exists():
        return None, None
    record = json.loads(record_path.read_text())
    try:
        initialization = record["initialization"]
        parameters = np.asarray(initialization["parameters"], dtype=float)
        N, L = spec["N"], len(spec["subtractions"])
        expected = L * (2*N*N + 3*N)
        if parameters.shape != (expected,) or not np.all(np.isfinite(parameters)):
            raise ValueError("Training record contains an invalid initialization vector")
        S, d = operators_from_parameters(parameters, N, L)
        S, d = validate_layers(S, d, spec["subtractions"])
        metadata = {
            "source": initialization.get("source"),
            "seed": initialization.get("seed"),
            "bit_generator": initialization.get("bit_generator"),
            "parameters_sha256": hashlib.sha256(np.ascontiguousarray(parameters).tobytes()).hexdigest(),
            "file": _file_info(record_path),
        }
        return (S, d), metadata
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid initialization in training record: {exc}") from exc

def load_task_data(task, definition, data_root):
    stem = _stem(definition["dataset_stem"])
    ip, op = (data_root / (stem + suffix) for suffix in ("_inputs.npy", "_outputs.npy"))
    x, y = load_numeric(ip), load_numeric(op)
    if x.ndim != 2 or x.shape[1] != 1 or y.ndim != 2 or y.shape[0] != len(x):
        raise ValueError("Expected inputs (M,1) and target moments (M,O)")
    x = _real(x, "Training amplitudes")
    if task == "cat":
        if x.shape != (1, 1) or y.shape != (1, 15):
            raise ValueError("Fig. 4 requires one input and all 15 target moments")
        evaluation = x[:, 0].copy()
    elif task == "cubic":
        if y.shape != (50, 6) or not np.allclose(x[:, 0], np.linspace(-2, 2, 50), atol=1e-12, rtol=0):
            raise ValueError("Fig. 3 requires its 50 ordered real inputs in [-2,2] and six moments")
        evaluation = (x[:-1, 0] + x[1:, 0]) / 2
        if np.any(np.isclose(evaluation[:, None], x[:, 0][None, :], atol=1e-12, rtol=0)):
            raise ValueError("Cubic evaluation inputs overlap the training grid")
    else:
        raise ValueError("Unknown task")
    return {"inputs": x, "targets": y, "evaluation": evaluation,
            "files": [_file_info(ip), _file_info(op)]}


def audit_model(spec, models_root, data, definition):
    row = {"id": spec["id"], "figure": spec["figure"], "task": spec["task"],
           "N": spec["N"], "L": len(spec["subtractions"]), "subtractions": spec["subtractions"],
           "checkpoint_stem": spec["checkpoint_stem"], "notes": spec.get("notes", ""),
           "status": "ready", "files": [], "loss_history_min": None,
           "checkpoint_loss": None, "fidelity_mean": None, "fidelity_min": None,
           "fidelity_train_mean": None, "fidelity_test_mean": None,
           "initial_fidelity_mean": None, "initial_fidelity_min": None,
           "initial_fidelity_train_mean": None, "initial_fidelity_test_mean": None,
           "initial_checkpoint_loss": None, "initial_loss_matches_record": None,
           "basinhopping_iterations": None}
    try:
        record_path = models_root / "training_records" / (_stem(spec["checkpoint_stem"]) + ".json")
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        loss = None
        if spec.get("loss_history"):
            lp = models_root / "train_losses" / _stem(spec["loss_history"])
            if lp.exists():
                loss = _real(load_numeric(lp), "Loss history").reshape(-1)
                row["files"].append(_file_info(lp))
        if record is not None:
            row["files"].append(_file_info(record_path))
            if record.get("schema_version") != 1:
                raise ValueError("Unsupported training_record schema")
            arch = record["architecture"]
            expected = {"N": spec["N"], "layers": len(spec["subtractions"]),
                        "ladder_modes": spec["subtractions"], "is_addition": False,
                        "include_initial_squeezing": False, "include_initial_mixing": False,
                        "is_passive_gaussian": False, "n_inputs": 1, "n_outputs": 1,
                        "observable": "catstates" if spec["task"] == "cat" else "cubicphase"}
            if any(arch.get(k) != v for k, v in expected.items()):
                raise ValueError("Training record does not match the requested architecture")
            if record["selection"]["metric"] != "train_loss":
                raise ValueError("Quantum table expects selection by training loss")
            history = np.asarray([h["train_loss"] for h in record["history"]], dtype=float)
            if not np.all(np.isfinite(history)):
                raise ValueError("Training record has non-finite losses")
            if loss is not None and (loss.shape != history.shape or not np.allclose(loss, history, atol=1e-10, rtol=1e-10)):
                raise ValueError("Loss file and training_record belong to different trajectories")
            loss = history
            for key, value in (("train_inputs", data["inputs"]), ("train_targets", data["targets"])):
                fingerprint = record["data"][key]
                if fingerprint["shape"] != list(value.shape) or fingerprint["sha256"] != hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest():
                    raise ValueError("Dataset differs from the saved training_record")
            row["recorded_selected_loss"] = float(record["selection"]["train_loss"])
            row["basinhopping_iterations"] = record["basinhopping"]["niter"]
            row["selection_history_index"] = record["selection"]["history_index"]
            selected_index = row["selection_history_index"]
            if not isinstance(selected_index, int) or not 0 <= selected_index < len(history):
                raise ValueError("Invalid selected history index")
            selected_loss = row["recorded_selected_loss"]
            if not np.isclose(history[selected_index], selected_loss, atol=1e-12, rtol=1e-12):
                raise ValueError("Selected loss and selected history index disagree")
            if not np.isclose(history.min(), selected_loss, atol=1e-12, rtol=1e-12):
                raise ValueError("Recorded selection is not the minimum training loss")
            initialization = record.get("initialization", {})
            row["training_seed"] = initialization.get("seed")
            row["initialization_source"] = initialization.get("source")
            initial_parameters = np.asarray(initialization.get("parameters", []), dtype=float)
            expected_count = len(spec["subtractions"]) * (2*spec["N"]*spec["N"] + 3*spec["N"])
            if initial_parameters.shape != (expected_count,) or not np.all(np.isfinite(initial_parameters)):
                raise ValueError("Training record contains an invalid initialization vector")
            first = record["history"][0]
            if first.get("phase") != "lbfgsb" or first.get("event") != "initial" or first.get("basin") != 0:
                raise ValueError("Training record does not begin at the saved random initialization")
            row["recorded_initial_loss"] = float(first["train_loss"])
            row["initial_parameters_sha256"] = hashlib.sha256(np.ascontiguousarray(initial_parameters).tobytes()).hexdigest()
            row["training_software_versions"] = record.get("software_versions", {})
            row["training_elapsed_seconds"] = record.get("elapsed_seconds")
            row["local_minimization_failures"] = record.get("scipy_result", {}).get("minimization_failures")
        if loss is not None:
            if not loss.size or np.any(loss < 0):
                raise ValueError("Loss history must be nonempty and nonnegative")
            row["loss_history_min"] = float(loss.min())
            row["loss_history_min_index"] = int(loss.argmin())
            row["loss_history_count"] = int(loss.size)
        S, d, files, source = load_checkpoint(spec, models_root)
        row["files"].extend(files)
        row["checkpoint_source"] = source
        row["dataset_files"] = data["files"]
        return row, (S, d)
    except FileNotFoundError as exc:
        row.update(status="missing_checkpoint", error=str(exc))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        row.update(status="invalid_checkpoint", error=str(exc))
    return row, None


def prepare_references(task, definition, data, cutoffs, target_tolerance):
    """Validate the target convention against cached training moments and cutoffs."""
    train = data["inputs"][:, 0]
    amplitudes = train if task == "cat" else np.r_[train, data["evaluation"]]
    kwargs = {"gamma": definition.get("gamma_sf", 0.2), "hbar": definition.get("hbar_sf", 2.0),
              "cat_alpha": definition.get("target_alpha", 1.0), "phi": definition.get("phi", 0.0)}
    refs, diagnostics = [], []
    for i, alpha in enumerate(amplitudes):
        if i < len(train):
            old = reference_ket(task, alpha, definition["target_generation_cutoff"], **kwargs)
            old_moments = output_moments(np.outer(old, old.conj()), task)
            if not np.allclose(old_moments, data["targets"][i], atol=1e-7, rtol=1e-8):
                raise ValueError(f"Target mismatch at alpha={alpha:g}: cached moments do not match the declared target convention")
        low = reference_ket(task, alpha, cutoffs[0], **kwargs)
        high = reference_ket(task, alpha, cutoffs[1], **kwargs)
        nl, nh = float(np.vdot(low, low).real), float(np.vdot(high, high).real)
        if min(nl, nh) <= 0:
            raise ValueError("Zero target norm")
        agreement = abs(np.vdot(low, high[:len(low)]))**2 / (nl*nh)
        error = max(0.0, float(1-agreement))
        if error > target_tolerance or abs(1-nl) > target_tolerance or abs(1-nh) > target_tolerance:
            raise ValueError(f"Reference cutoff not converged at alpha={alpha:g}; increase --reference-cutoffs")
        refs.append(high)
        diagnostics.append({"alpha": float(alpha), "lower_norm": nl, "upper_norm": nh,
                            "reference_infidelity": error})
    return amplitudes, refs, diagnostics



def _evaluate_operator_family(operators, row, data, prepared, args, label, cutoffs=None, full_convergence=True):
    """Evaluate one fixed parameter point with cutoff convergence checks.

    Trained checkpoints use the strict moment/norm/fidelity criterion required
    to validate the reported moment loss.  The random baseline is used only for
    fidelity, so its convergence criterion intentionally excludes high-order
    moments, which can converge much more slowly than the normalized-state
    fidelity for energetic random initializations.
    """
    S, d = operators
    amplitudes, refs, target_diagnostics = prepared
    n_train = len(data["inputs"])
    selected = slice(0, n_train) if row["task"] == "cat" else slice(n_train, None)
    cutoff_history = []
    previous, stable_count = None, 0
    converged = False
    state_diagnostics = None
    moments = fidelities = weights = None
    D = None

    cutoffs = args.cutoffs if cutoffs is None else cutoffs
    for D in cutoffs:
        moments_D, fidelities_D, weights_D, edges = [], [], [], []
        diagnostics_D = []
        for i, alpha in enumerate(amplitudes):
            ket, weight = projected_qonn_ket(
                S, d, row["subtractions"], alpha, D,
                max_tensor_bytes=int(args.max_memory_mib*1024**2),
            )
            rho = reduced_output(ket)
            populations = np.diag(rho).real
            diagnostics_D.append({
                "purity": float(np.trace(rho @ rho).real),
                "parity": float(np.dot((-1.) ** np.arange(D), populations)),
                "even_photon_probability": float(populations[::2].sum()),
                "mean_photon_number": float(np.dot(np.arange(D), populations)),
            })
            moments_D.append(output_moments(rho, row["task"]))
            fidelities_D.append(output_fidelity(rho, refs[i]))
            weights_D.append(weight)
            inside = ket[(slice(0, D-2),) * row["N"]]
            edges.append(max(0.0, float(1-np.vdot(inside, inside).real / weight)))

        moments = np.asarray(moments_D)
        fidelities = np.asarray(fidelities_D)
        weights = np.asarray(weights_D)
        state_diagnostics = diagnostics_D
        loss = float(np.mean(np.sum(abs(moments[:n_train] - data["targets"])**2, axis=1)))
        diagnostic = {
            "cutoff": D,
            "checkpoint_loss": loss,
            "fidelity_mean": float(fidelities[selected].mean()),
            "fidelity_min": float(fidelities[selected].min()),
            "max_edge_population": max(edges),
        }
        if previous is not None:
            # Exact nested projections: increasing D cannot reduce the raw norm.
            if np.any(weights < previous["weights"] * (1-1e-9)):
                raise ValueError(f"{label}: projected norm decreased with cutoff; numerical instability")
            norm_delta = float(np.max(abs(weights-previous["weights"]) / weights))
            moment_delta = float(np.max(abs(moments-previous["moments"]) / (1+abs(moments))))
            fidelity_delta = float(np.max(abs(fidelities-previous["fidelities"])))
            if full_convergence:
                stable = (norm_delta <= args.norm_tol and moment_delta <= args.moment_tol
                          and fidelity_delta <= args.fidelity_tol)
            else:
                stable = (norm_delta <= args.initial_norm_tol
                          and fidelity_delta <= args.initial_fidelity_tol)
            stable_count = stable_count + 1 if stable else 0
            diagnostic.update(relative_norm_change=norm_delta,
                              scaled_moment_change=moment_delta,
                              max_fidelity_change=fidelity_delta,
                              convergence_criterion="full" if full_convergence else "fidelity_and_norm",
                              stable=stable)
        cutoff_history.append(diagnostic)
        print(f"{row['id']} [{label}]: D={D}, loss={loss:.10g}, "
              f"F={diagnostic['fidelity_mean']:.10g}", flush=True)
        previous = {"weights": weights, "moments": moments, "fidelities": fidelities}
        if stable_count >= 2:
            converged = True
            break

    result = {
        "converged": converged,
        "cutoff": D,
        "cutoff_history": cutoff_history,
        "reference_diagnostics": target_diagnostics,
    }
    if not converged:
        return result

    train_fidelities = fidelities[:n_train]
    result.update({
        "loss": loss,
        "fidelity_mean": float(fidelities[selected].mean()),
        "fidelity_min": float(fidelities[selected].min()),
        "fidelity_sample_count": int(fidelities[selected].size),
        "fidelity_min_alpha": float(amplitudes[selected][fidelities[selected].argmin()]),
        "fidelity_train_mean": float(train_fidelities.mean()),
        "fidelity_train_min": float(train_fidelities.min()),
        "points": [{
            "alpha": float(alpha),
            "role": "train" if i < n_train else "evaluation",
            "fidelity": float(fidelities[i]),
            "ideal_operator_weight": float(weights[i]),
            **state_diagnostics[i],
            "moments_real": moments[i].real.tolist(),
            "moments_imag": moments[i].imag.tolist(),
        } for i, alpha in enumerate(amplitudes)],
    })
    if row["task"] == "cubic":
        test_fidelities = fidelities[n_train:]
        result.update({
            "fidelity_test_mean": float(test_fidelities.mean()),
            "fidelity_test_min": float(test_fidelities.min()),
        })
    else:
        result.update({
            "fidelity_test_mean": None,
            "fidelity_test_min": None,
        })
    return result


def _initial_cutoffs(N, args):
    """Memory-aware default cutoff ladders for the untrained baseline."""
    if args.initial_cutoffs is not None:
        return args.initial_cutoffs
    defaults = {
        1: [20, 40, 60, 80, 100, 120, 140, 160],
        2: [20, 30, 40, 50, 60, 80],
        3: [16, 20, 24, 30, 36, 42, 50],
        4: [12, 16, 20, 24, 30, 36, 40],
    }
    if N in defaults:
        return defaults[N]
    # For larger mode counts the tensor workspace grows rapidly; require an
    # explicit choice rather than silently requesting an impractical cutoff.
    raise ValueError("For N>4, provide --initial-cutoffs explicitly")


def evaluate_model(row, operators, initial_operators, data, prepared, args):
    """Evaluate the trained checkpoint and its paired random initialization."""
    trained = _evaluate_operator_family(operators, row, data, prepared, args, "trained")
    row["cutoff_history"] = trained["cutoff_history"]
    row["reference_diagnostics"] = trained["reference_diagnostics"]
    row["converged"] = trained["converged"]
    row["cutoff"] = trained["cutoff"]
    if not trained["converged"]:
        row.update(status="not_converged",
                   error="Trained checkpoint did not pass two successive cutoff comparisons; increase --cutoffs")
        return

    row.update(
        checkpoint_loss=trained["loss"],
        fidelity_mean=trained["fidelity_mean"],
        fidelity_min=trained["fidelity_min"],
        fidelity_train_mean=trained["fidelity_train_mean"],
        fidelity_test_mean=trained["fidelity_test_mean"],
        fidelity_sample_count=trained["fidelity_sample_count"],
        fidelity_min_alpha=trained["fidelity_min_alpha"],
        points=trained["points"],
        status="ok",
    )
    if row["task"] == "cubic":
        row["fidelity_test_min"] = trained["fidelity_test_min"]

    row["loss_matches_history_min"] = (
        None if row["loss_history_min"] is None else
        bool(np.isclose(trained["loss"], row["loss_history_min"],
                        atol=args.loss_atol, rtol=args.loss_rtol))
    )
    row["loss_selection_evidence"] = "numeric_agreement_only; no historical parameter trace"
    if "recorded_selected_loss" in row:
        row["loss_selection_evidence"] = "training_record_and_recomputed_checkpoint_loss"
        if not np.isclose(trained["loss"], row["recorded_selected_loss"],
                          atol=args.loss_atol, rtol=args.loss_rtol):
            row.update(status="loss_mismatch",
                       error="Recomputed physical loss differs from the recorded selected loss")
            return
    if row["loss_matches_history_min"] is False:
        row.update(status="loss_mismatch",
                   error="Saved checkpoint does not reproduce the historical minimum; do not pair that minimum with this fidelity")
        return

    if initial_operators is None:
        row["initial_baseline_status"] = "unavailable"
        return

    initial = _evaluate_operator_family(initial_operators, row, data, prepared, args, "initial",
                                        cutoffs=_initial_cutoffs(row["N"], args), full_convergence=False)
    row["initial_cutoff_history"] = initial["cutoff_history"]
    row["initial_converged"] = initial["converged"]
    row["initial_cutoff"] = initial["cutoff"]
    if not initial["converged"]:
        row.update(status="initial_not_converged",
                   initial_baseline_status="not_converged",
                   error="Random initialization did not pass two successive cutoff comparisons; extend --initial-cutoffs")
        return

    row.update(
        initial_baseline_status="ok",
        initial_checkpoint_loss=initial["loss"],
        initial_fidelity_mean=initial["fidelity_mean"],
        initial_fidelity_min=initial["fidelity_min"],
        initial_fidelity_train_mean=initial["fidelity_train_mean"],
        initial_fidelity_test_mean=initial["fidelity_test_mean"],
        initial_fidelity_sample_count=initial["fidelity_sample_count"],
        initial_fidelity_min_alpha=initial["fidelity_min_alpha"],
    )
    if row["task"] == "cubic":
        row["initial_fidelity_test_min"] = initial["fidelity_test_min"]

    if "recorded_initial_loss" in row:
        # Diagnostic only: initial convergence is certified for normalized-state
        # fidelity and norm, not for the sixth-order moment loss.
        row["initial_loss_matches_record"] = bool(np.isclose(
            initial["loss"], row["recorded_initial_loss"],
            atol=args.loss_atol, rtol=args.loss_rtol,
        ))

def _atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = handle.name
    os.replace(temporary, path)



def write_outputs(report, output, audit=False):
    prefix = "audit" if audit else "quantum_tables"
    _atomic_text(output / (prefix + ".json"), json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = [
        "id", "figure", "N", "L", "subtractions", "checkpoint_stem", "status",
        "basinhopping_iterations", "loss_history_min", "checkpoint_loss",
        "loss_matches_history_min", "recorded_initial_loss", "initial_checkpoint_loss",
        "initial_loss_matches_record", "initial_fidelity_mean", "initial_fidelity_train_mean",
        "initial_fidelity_test_mean", "fidelity_mean", "fidelity_train_mean", "fidelity_test_mean",
        "fidelity_min", "fidelity_test_min", "fidelity_min_alpha", "fidelity_sample_count",
        "cutoff", "initial_cutoff", "error",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in report["models"]:
        writer.writerow({c: row.get(c) for c in columns})
    _atomic_text(output / (prefix + ".csv"), stream.getvalue())
    if audit:
        return

    for task in ("cat", "cubic"):
        rows = [r for r in report["models"] if r["task"] == task]
        if not rows:
            continue
        if task == "cat":
            lines = [
                r"\begin{tabular}{lrrr}", r"\hline",
                r"QONN & Checkpoint loss & $F_{\mathrm{init}}$ & $F$ \\", r"\hline",
            ]
        else:
            lines = [
                r"\begin{tabular}{lrrrrr}", r"\hline",
                r"QONN & Checkpoint loss & $\overline{F}_{\mathrm{test}}^{\mathrm{init}}$ & "
                r"$\overline{F}_{\mathrm{train}}$ & $\overline{F}_{\mathrm{test}}$ & "
                r"$F_{\mathrm{test,min}}$ \\", r"\hline",
            ]
        for row in rows:
            k = ";".join("[" + ",".join(str(m+1) for m in layer) + "]" for layer in row["subtractions"])
            label = f"$N={row['N']},L={row['L']},K=({k})$"
            if row["status"] == "ok":
                if task == "cat":
                    values = [
                        f"{row['checkpoint_loss']:.6g}",
                        f"{row['initial_fidelity_mean']:.6f}" if row.get("initial_fidelity_mean") is not None else "--",
                        f"{row['fidelity_mean']:.6f}",
                    ]
                else:
                    values = [
                        f"{row['checkpoint_loss']:.6g}",
                        f"{row['initial_fidelity_test_mean']:.6f}" if row.get("initial_fidelity_test_mean") is not None else "--",
                        f"{row['fidelity_train_mean']:.6f}",
                        f"{row['fidelity_test_mean']:.6f}",
                        f"{row['fidelity_test_min']:.6f}",
                    ]
            else:
                values = ["--"] * (3 if task == "cat" else 5)
            lines += ["% " + row["id"] + ": " + row["status"], label + " & " + " & ".join(values) + r" \\"]
        lines += [
            r"\hline", r"\end{tabular}",
            "% K lists photon subtractions in each layer, with 1-based mode indices.",
            "% F_init uses the exact random parameter vector saved at the start of the corresponding training run.",
            "% Trained loss and fidelity refer to the selected saved checkpoint; -- means missing or unvalidated.",
            "% See quantum_tables.json for cutoff tests, provenance and full-precision values.",
        ]
        if task == "cubic":
            lines.append("% F_train uses the 50 training amplitudes; F_test and F_init use the 49 unseen midpoint amplitudes.")
            lines.append("% These are conditional output-state fidelities, not process/gate fidelities.")
        _atomic_text(output / (task + "_table.tex"), "\n".join(lines) + "\n")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("quantum_tables_manifest.json"))
    parser.add_argument("--models-root", type=Path, default=quannto_results_dir() / "models")
    parser.add_argument("--datasets-root", type=Path, default=datasets_dir())
    parser.add_argument("--output", type=Path, default=Path("results/local/quantum_evaluation"))
    parser.add_argument("--task", choices=("all", "cat", "cubic"), default="all")
    parser.add_argument("--model", action="append", help="Evaluate only this manifest id (repeatable)")
    parser.add_argument("--audit", action="store_true", help="Check files and provenance without Fock evaluation")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[20, 30, 40, 60])
    parser.add_argument("--initial-cutoffs", type=int, nargs="+", default=None,
                        help="Optional common cutoff ladder for random-initialization fidelities; defaults are memory-aware by N")
    parser.add_argument("--reference-cutoffs", type=int, nargs=2, default=[200, 300])
    parser.add_argument("--fidelity-tol", type=float, default=1e-7)
    parser.add_argument("--norm-tol", type=float, default=1e-8)
    parser.add_argument("--initial-fidelity-tol", type=float, default=1e-7)
    parser.add_argument("--initial-norm-tol", type=float, default=1e-6)
    parser.add_argument("--moment-tol", type=float, default=1e-7)
    parser.add_argument("--target-tol", type=float, default=1e-8)
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--max-memory-mib", type=float, default=512)
    args = parser.parse_args(argv)
    if len(args.cutoffs) < 3 or min(args.cutoffs) < 2 or args.cutoffs != sorted(set(args.cutoffs)):
        parser.error("--cutoffs requires at least three strictly increasing integers >=2")
    if args.initial_cutoffs is not None:
        if (len(args.initial_cutoffs) < 3 or min(args.initial_cutoffs) < 2
                or args.initial_cutoffs != sorted(set(args.initial_cutoffs))):
            parser.error("--initial-cutoffs requires at least three strictly increasing integers >=2")
    requested_max = max(args.cutoffs + ([] if args.initial_cutoffs is None else args.initial_cutoffs))
    if args.reference_cutoffs[0] < requested_max or args.reference_cutoffs[1] <= args.reference_cutoffs[0]:
        parser.error("Reference cutoffs must increase and cover every explicitly requested QONN cutoff")
    for key in ("fidelity_tol", "norm_tol", "initial_fidelity_tol", "initial_norm_tol",
                "moment_tol", "target_tol", "loss_atol", "loss_rtol", "max_memory_mib"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be positive and finite")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        parser.error("Unsupported manifest schema")
    ids = [s["id"] for s in manifest["models"]]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_]+", x) for x in ids):
        parser.error("Manifest ids must be unique lowercase alphanumeric names")
    if args.model and any(x not in ids for x in args.model):
        parser.error("Unknown model id")
    specs = [s for s in manifest["models"] if (s.get("included_in_paper", True) or args.model) and (args.task == "all" or s["task"] == args.task)
             and (not args.model or s["id"] in args.model)]
    if not specs:
        parser.error("No models match the requested task/id")
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "strawberryfields", "thewalrus", "numba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    report = {"schema_version": 1, "mode": "audit" if args.audit else "evaluation", "complete": False,
              "manifest": _file_info(args.manifest), "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "software_versions": versions, "target_definitions": manifest["tasks"],
              "fidelity_convention": "squared fidelity of normalized conditional mode-0 output versus a pure target",
              "initial_baseline_definition": "paired fidelity at the exact random parameter vector saved before optimization in each training_record",
              "cubic_scope": "output-state fidelities on real coherent inputs; not process, Choi or Haar-average gate fidelity",
              "models": []}
    prepared_tasks = {}
    for spec in specs:
        definition = manifest["tasks"][spec["task"]]
        try:
            data = load_task_data(spec["task"], definition, args.datasets_root)
        except (OSError, ValueError) as exc:
            row = dict(spec, L=len(spec["subtractions"]), status="invalid_dataset", error=str(exc))
            report["models"].append(row)
            write_outputs(report, args.output, args.audit)
            continue
        row, operators = audit_model(spec, args.models_root, data, definition)
        report["models"].append(row)
        if not args.audit and operators is not None:
            try:
                initial_operators, initial_metadata = load_initial_operators(spec, args.models_root)
                if initial_metadata is not None:
                    row["initial_baseline"] = initial_metadata
                    if row.get("initial_parameters_sha256") != initial_metadata["parameters_sha256"]:
                        raise ValueError("Initialization provenance changed between audit and evaluation")
                if spec["task"] not in prepared_tasks:
                    prepared_tasks[spec["task"]] = prepare_references(spec["task"], definition, data, args.reference_cutoffs, args.target_tol)
                evaluate_model(row, operators, initial_operators, data, prepared_tasks[spec["task"]], args)
            except (ValueError, ImportError, RuntimeError, MemoryError) as exc:
                row.update(status="evaluation_error", error=str(exc))
        print(f"{row['id']}: {row['status']}" + (f" — {row['error']}" if row.get("error") else ""), flush=True)
        write_outputs(report, args.output, args.audit)
    expected_status = "ready" if args.audit else "ok"
    report["complete"] = all(row["status"] == expected_status for row in report["models"])
    write_outputs(report, args.output, args.audit)
    print(f"Reports: {args.output.resolve()}", flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
