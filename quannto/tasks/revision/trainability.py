"""Finite-size QONN trainability diagnostic for the manuscript revision.

This command does *not* train a model. It evaluates a fixed scalar homodyne
MSE cost and its automatic derivative over pre-registered random-initialization
ensembles. Width, depth, ladder order, and circuit energy are varied in
separate scans so a small gradient cannot be attributed to several resources
changing at once.

The experiment is deliberately described as a finite-size diagnostic. It can
reveal gradient suppression in the explicitly recorded accessible regime, but
it cannot establish or exclude an asymptotic barren plateau.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from quannto.tasks.revision.trainability_protocol import (
    ENSEMBLES, PROTOCOL_VERSION, SWEEPS, ScanPoint, canonical_json,
    loglog_fit, matched_probe_indices, parameter_seed,
    parameter_types, sample_initial_parameters, semilog_fit,
    summarize_gradients,
)
from quannto.utils.results_utils import colors, linestyles


# Fixed scientific protocol.
DEFAULT_TRIALS = 64
DEFAULT_BOOTSTRAP_REPETITIONS = 1000
DEFAULT_BATCH_SIZE = 32
DEFAULT_DATASET_SEED = 1234
DEFAULT_PARAMETER_SEED = 2026
SCAN_PROFILES = {
    "manuscript": {
        "widths": (2, 3, 4, 5),
        "depths": (1, 2, 3),
        "ladder_orders": (0, 1, 2, 3, 4),
        "energy_scales": (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        "fixed_modes_for_depth": 2,
        "fixed_modes_for_ladder": 4,
        "fixed_modes_for_energy": 3,
    },
    "extended_mn5": {
        "widths": (2, 3, 4, 5, 6),
        "depths": (1, 2, 3, 4),
        "ladder_orders": (0, 1, 2, 3, 4, 5),
        "energy_scales": (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        "fixed_modes_for_depth": 2,
        "fixed_modes_for_ladder": 5,
        "fixed_modes_for_energy": 3,
    },
}
INPUT_LOW = -0.75
INPUT_HIGH = 0.75

SIGMA_PASSIVE = 0.10
SIGMA_SQUEEZING = 0.05
SIGMA_DISPLACEMENT = 0.05
TINY_NORM_THRESHOLD = 1e-10
SMALL_GRADIENT_THRESHOLDS = (1e-8, 1e-6, 1e-4)

GRADCHECK_CONFIGS = {(2, 1, 1, 1.0), (2, 2, 1, 1.0)}
GRADCHECK_NUM_PARAMETERS = 8
GRADCHECK_EPS = 1e-5
GRADCHECK_ABS_TOL = 1e-6
GRADCHECK_REL_TOL = 1e-4

OBSERVABLE = "position"
IS_ADDITION = False
INCLUDE_INITIAL_SQUEEZING = False
INCLUDE_INITIAL_MIXING = False
IS_PASSIVE_GAUSSIAN = False


def _load_jax_runtime():
    """Import JAX only inside an isolated configuration worker."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


# Small I/O helpers.
def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _real_checked(value: Any, *, name: str, tolerance: float = 1e-10) -> np.ndarray:
    array = np.asarray(value)
    maximum_imaginary = (
        float(np.max(np.abs(np.imag(array)))) if np.iscomplexobj(array) else 0.0
    )
    if maximum_imaginary > tolerance:
        raise ValueError(
            f"{name} has a material imaginary component: "
            f"max |Im|={maximum_imaginary:.3e}"
        )
    real = np.asarray(np.real(array), dtype=np.float64)
    if not np.all(np.isfinite(real)):
        raise ValueError(f"{name} contains non-finite values")
    return real


def _stderr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _release_runtime_memory() -> None:
    gc.collect()
    try:
        import jax
        jax.clear_caches()
    except (AttributeError, ImportError):
        pass
    gc.collect()


# QONN construction and deterministic cost.
def _build_qnn(point: ScanPoint, *, expectation_backend: str):
    from quannto.core.qnn import QNN

    return QNN(
        model_name=f"trainability_{point.architecture_key}",
        task_name="finite_size_trainability",
        N=point.n_modes,
        layers=point.layers,
        n_in=1,
        n_out=1,
        ladder_modes=point.ladder_modes,
        is_addition=IS_ADDITION,
        observable=OBSERVABLE,
        include_initial_squeezing=INCLUDE_INITIAL_SQUEEZING,
        include_initial_mixing=INCLUDE_INITIAL_MIXING,
        is_passive_gaussian=IS_PASSIVE_GAUSSIAN,
        parameters=None,
        in_preprocessors=[],
        out_preprocessors=[],
        postprocessors=[],
        expectation_backend=expectation_backend,
        expectation_batch_size=1,
    )


def _preprocess_inputs(qnn: Any, inputs: np.ndarray):
    _, jnp = _load_jax_runtime()
    processed: Any = inputs
    for function in qnn.in_preprocessors:
        processed = function(processed)
    processed = jnp.asarray(processed, dtype=jnp.float64)
    expected_shape = (len(inputs), 2 * qnn.N)
    if processed.shape != expected_shape:
        raise AssertionError(
            f"preprocessed inputs have shape {processed.shape}, expected {expected_shape}"
        )
    return processed


def _gaussian_core_photons_per_mode(
    mean_vector: Any, covariance: Any, n_modes: int,
):
    """Photon numbers of the Gaussian core used by the ladder expansion."""

    _, jnp = _load_jax_runtime()
    mean_x = jnp.real(mean_vector[:n_modes])
    mean_p = jnp.real(mean_vector[n_modes:2 * n_modes])
    diagonal = jnp.real(jnp.diag(covariance))
    var_x = diagonal[:n_modes]
    var_p = diagonal[n_modes:2 * n_modes]
    photons = 0.5 * (var_x + var_p + mean_x**2 + mean_p**2 - 1.0)
    return jnp.maximum(photons, 0.0)


def _compiled_functions(qnn: Any, inputs: np.ndarray, targets: np.ndarray):
    jax, jnp = _load_jax_runtime()
    inputs_jax = _preprocess_inputs(qnn, inputs)
    targets_jax = jnp.asarray(targets[:, 0], dtype=jnp.float64)

    def batch_forward(parameters: jnp.ndarray):
        qnn.build_QNN(parameters)

        def single_forward(input_displacement: jnp.ndarray):
            mean_vector, covariance = qnn.build_gaussian_core_xxpp(
                input_displacement
            )
            photons = _gaussian_core_photons_per_mode(
                mean_vector, covariance, qnn.N,
            )
            output, norm = qnn.eval_QNN(input_displacement)
            return output[0], norm, jnp.mean(photons), jnp.sum(photons)

        return jax.vmap(single_forward, in_axes=(0,))(inputs_jax)

    def scalar_homodyne_mse_with_aux(parameters: jnp.ndarray):
        predictions, norms, energy_per_mode, total_energy = batch_forward(parameters)
        residual = jnp.real(predictions) - targets_jax
        loss = jnp.mean(residual**2)
        return loss, (predictions, norms, energy_per_mode, total_energy)

    def scalar_homodyne_mse(parameters: jnp.ndarray) -> jnp.ndarray:
        loss, _ = scalar_homodyne_mse_with_aux(parameters)
        return loss

    return (
        jax.jit(jax.value_and_grad(
            scalar_homodyne_mse_with_aux, has_aux=True,
        )),
        jax.jit(scalar_homodyne_mse),
    )


# Checkpoint signatures and validation.
def _configuration_signature(
    point: ScanPoint,
    *,
    ensemble: str,
    trials: int,
    dataset_hash: str,
    parameter_seed_value: int,
    expectation_backend: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "ensemble": ensemble,
        "trials": int(trials),
        "N": point.n_modes,
        "L": point.layers,
        "K": point.k_subtractions,
        "energy_scale": point.energy_scale,
        "ladder_modes_zero_indexed": point.ladder_modes,
        "dataset_sha256": dataset_hash,
        "parameter_seed": int(parameter_seed_value),
        "parameter_seed_excludes_K_and_energy_scale_for_paired_scans": True,
        "sigma_passive": SIGMA_PASSIVE,
        "sigma_squeezing": SIGMA_SQUEEZING,
        "sigma_displacement": SIGMA_DISPLACEMENT,
        "energy_parameters_scaled_by_inverse_sqrt_depth": (
            ensemble == "energy_controlled"
        ),
        "passive_parameters_scaled_with_depth": False,
        "expectation_backend": expectation_backend,
        "observable": OBSERVABLE,
        "cost": "mean((Re(<x_1>) - x)^2) on one fixed scalar dataset",
    }


def _checkpoint_path(
    output_directory: Path, point: ScanPoint, *, trials: int,
) -> Path:
    return output_directory / "checkpoints" / (
        f"{point.architecture_key}_T{trials}.npz"
    )


def _partial_checkpoint_path(
    output_directory: Path, point: ScanPoint, *, trials: int,
) -> Path:
    return output_directory / "checkpoints" / (
        f"{point.architecture_key}_T{trials}.partial.npz"
    )


def _load_checkpoint(
    path: Path,
    *,
    expected_signature: Mapping[str, Any],
    expected_parameters: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        found_signature = str(archive["signature_json"].item())
        expected_json = canonical_json(expected_signature)
        if found_signature != expected_json:
            raise ValueError(
                f"checkpoint protocol mismatch in {path}; use a new protocol "
                "version/output directory or rerun with --overwrite"
            )
        result = {key: np.asarray(archive[key]) for key in archive.files}

    trials = int(expected_signature["trials"])
    if result["gradients"].shape != (trials, expected_parameters):
        raise ValueError(
            f"checkpoint {path} has gradient shape {result['gradients'].shape}, "
            f"expected {(trials, expected_parameters)}"
        )
    if result["parameters"].shape != (trials, expected_parameters):
        raise ValueError(f"checkpoint {path} has an invalid parameter array")
    for name, array in result.items():
        if name == "signature_json" or array.dtype.kind in {"U", "S", "O"}:
            continue
        if not np.all(np.isfinite(array)):
            raise ValueError(f"checkpoint {path} contains non-finite {name}")
    return result


def _load_partial_checkpoint(
    path: Path,
    *,
    expected_signature: Mapping[str, Any],
    expected_parameters: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        found_signature = str(archive["signature_json"].item())
        if found_signature != canonical_json(expected_signature):
            raise ValueError(
                f"partial checkpoint protocol mismatch in {path}; use "
                "--overwrite or a different results directory"
            )
        result = {key: np.asarray(archive[key]) for key in archive.files}
    completed = int(result["completed_trials"].item())
    total = int(expected_signature["trials"])
    if not 0 <= completed <= total:
        raise ValueError(f"invalid completed_trials={completed} in {path}")
    if result["gradients"].shape != (completed, expected_parameters):
        raise ValueError(f"partial checkpoint {path} has invalid gradients")
    if result["parameters"].shape != (completed, expected_parameters):
        raise ValueError(f"partial checkpoint {path} has invalid parameters")
    for name, array in result.items():
        if name in {"signature_json", "completed_trials"}:
            continue
        if array.dtype.kind not in {"U", "S", "O"} and not np.all(np.isfinite(array)):
            raise ValueError(f"partial checkpoint {path} contains non-finite {name}")
    return result


def _trial_payload(
    *,
    signature: Mapping[str, Any],
    completed_trials: int,
    parameters: np.ndarray,
    gradients: np.ndarray,
    losses: np.ndarray,
    gradient_norms: np.ndarray,
    mean_norms: np.ndarray,
    median_norms: np.ndarray,
    tiny_norm_fractions: np.ndarray,
    nonpositive_norm_fractions: np.ndarray,
    mean_core_energy_per_mode: np.ndarray,
    mean_core_total_energy: np.ndarray,
    mean_abs_outputs: np.ndarray,
    trial_small_gradient_fractions: np.ndarray,
    gradcheck: Mapping[str, np.ndarray],
    include_completed_counter: bool,
) -> dict[str, np.ndarray]:
    count = int(completed_trials)
    payload: dict[str, np.ndarray] = {
        "signature_json": np.asarray(canonical_json(signature)),
        "parameters": parameters[:count],
        "gradients": gradients[:count],
        "losses": losses[:count],
        "gradient_norms": gradient_norms[:count],
        "mean_norms": mean_norms[:count],
        "median_norms": median_norms[:count],
        "tiny_norm_fractions": tiny_norm_fractions[:count],
        "nonpositive_norm_fractions": nonpositive_norm_fractions[:count],
        "mean_core_energy_per_mode": mean_core_energy_per_mode[:count],
        "mean_core_total_energy": mean_core_total_energy[:count],
        "mean_abs_outputs": mean_abs_outputs[:count],
        "trial_small_gradient_fractions_1em6": (
            trial_small_gradient_fractions[:count]
        ),
        **gradcheck,
    }
    if include_completed_counter:
        payload["completed_trials"] = np.asarray(count, dtype=np.int64)
    return payload


# Gradient validation and one-configuration execution.
def _finite_difference_gradcheck(
    *,
    parameters: np.ndarray,
    autodiff_gradient: np.ndarray,
    scalar_cost,
    bounds: Sequence[tuple[float | None, float | None]],
    seed: int,
) -> dict[str, np.ndarray]:
    _, jnp = _load_jax_runtime()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(
        len(parameters),
        size=min(GRADCHECK_NUM_PARAMETERS, len(parameters)),
        replace=False,
    ))
    finite_difference = np.empty(len(indices), dtype=np.float64)
    autodiff = autodiff_gradient[indices].astype(np.float64)

    for position, index in enumerate(indices):
        lower, upper = bounds[int(index)]
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += GRADCHECK_EPS
        minus[index] -= GRADCHECK_EPS
        if lower is not None:
            plus[index] = max(plus[index], lower)
            minus[index] = max(minus[index], lower)
        if upper is not None:
            plus[index] = min(plus[index], upper)
            minus[index] = min(minus[index], upper)
        denominator = plus[index] - minus[index]
        if denominator == 0:
            raise ValueError("finite-difference step collapsed at a parameter bound")
        cost_plus = float(scalar_cost(jnp.asarray(plus, dtype=jnp.float64)))
        cost_minus = float(scalar_cost(jnp.asarray(minus, dtype=jnp.float64)))
        finite_difference[position] = (cost_plus - cost_minus) / denominator

    absolute_error = np.abs(autodiff - finite_difference)
    relative_error = absolute_error / np.maximum(
        1e-12, np.maximum(np.abs(autodiff), np.abs(finite_difference))
    )
    passed = (absolute_error <= GRADCHECK_ABS_TOL) | (
        relative_error <= GRADCHECK_REL_TOL
    )
    return {
        "gradcheck_indices": indices.astype(np.int64),
        "gradcheck_autodiff": autodiff,
        "gradcheck_finite_difference": finite_difference,
        "gradcheck_absolute_error": absolute_error,
        "gradcheck_relative_error": relative_error,
        "gradcheck_passed": passed.astype(np.int8),
    }


def _run_configuration(
    point: ScanPoint,
    *,
    ensemble: str,
    trials: int,
    inputs: np.ndarray,
    targets: np.ndarray,
    dataset_hash: str,
    parameter_seed_value: int,
    expectation_backend: str,
    output_directory: Path,
    resume: bool,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    types = parameter_types(point.n_modes, point.layers)
    n_parameters = len(types)
    signature = _configuration_signature(
        point, ensemble=ensemble, trials=trials, dataset_hash=dataset_hash,
        parameter_seed_value=parameter_seed_value,
        expectation_backend=expectation_backend,
    )
    checkpoint = _checkpoint_path(output_directory, point, trials=trials)
    partial_checkpoint = _partial_checkpoint_path(
        output_directory, point, trials=trials,
    )
    if checkpoint.exists() and resume and not overwrite:
        print(f"[resume] {point.architecture_key}")
        return _load_checkpoint(
            checkpoint, expected_signature=signature,
            expected_parameters=n_parameters,
        )
    if checkpoint.exists() and not overwrite:
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint}; pass --resume to reuse it "
            "or --overwrite to recompute it"
        )
    if overwrite:
        checkpoint.unlink(missing_ok=True)
        partial_checkpoint.unlink(missing_ok=True)
    elif partial_checkpoint.exists() and not resume:
        raise FileExistsError(
            f"partial checkpoint already exists: {partial_checkpoint}; pass "
            "--resume to continue it or --overwrite to restart it"
        )

    parameters = np.empty((trials, n_parameters), dtype=np.float64)
    gradients = np.empty_like(parameters)
    losses = np.empty(trials, dtype=np.float64)
    gradient_norms = np.empty(trials, dtype=np.float64)
    mean_norms = np.empty(trials, dtype=np.float64)
    median_norms = np.empty(trials, dtype=np.float64)
    tiny_norm_fractions = np.empty(trials, dtype=np.float64)
    nonpositive_norm_fractions = np.empty(trials, dtype=np.float64)
    mean_core_energy_per_mode = np.empty(trials, dtype=np.float64)
    mean_core_total_energy = np.empty(trials, dtype=np.float64)
    mean_abs_outputs = np.empty(trials, dtype=np.float64)
    trial_small_gradient_fractions = np.empty(trials, dtype=np.float64)
    gradcheck: dict[str, np.ndarray] = {
        "gradcheck_indices": np.empty(0, dtype=np.int64),
        "gradcheck_autodiff": np.empty(0, dtype=np.float64),
        "gradcheck_finite_difference": np.empty(0, dtype=np.float64),
        "gradcheck_absolute_error": np.empty(0, dtype=np.float64),
        "gradcheck_relative_error": np.empty(0, dtype=np.float64),
        "gradcheck_passed": np.empty(0, dtype=np.int8),
    }
    first_trial = 0
    if partial_checkpoint.exists() and resume:
        partial = _load_partial_checkpoint(
            partial_checkpoint,
            expected_signature=signature,
            expected_parameters=n_parameters,
        )
        first_trial = int(partial["completed_trials"].item())
        for destination, name in (
            (parameters, "parameters"),
            (gradients, "gradients"),
            (losses, "losses"),
            (gradient_norms, "gradient_norms"),
            (mean_norms, "mean_norms"),
            (median_norms, "median_norms"),
            (tiny_norm_fractions, "tiny_norm_fractions"),
            (nonpositive_norm_fractions, "nonpositive_norm_fractions"),
            (mean_core_energy_per_mode, "mean_core_energy_per_mode"),
            (mean_core_total_energy, "mean_core_total_energy"),
            (mean_abs_outputs, "mean_abs_outputs"),
            (trial_small_gradient_fractions, "trial_small_gradient_fractions_1em6"),
        ):
            destination[:first_trial] = partial[name]
        for name in gradcheck:
            gradcheck[name] = partial[name]
        print(
            f"[resume] {point.architecture_key}: continuing at trial "
            f"{min(first_trial + 1, trials)}/{trials}"
        )

    if first_trial == trials:
        payload = _trial_payload(
            signature=signature, completed_trials=trials,
            parameters=parameters, gradients=gradients, losses=losses,
            gradient_norms=gradient_norms, mean_norms=mean_norms,
            median_norms=median_norms,
            tiny_norm_fractions=tiny_norm_fractions,
            nonpositive_norm_fractions=nonpositive_norm_fractions,
            mean_core_energy_per_mode=mean_core_energy_per_mode,
            mean_core_total_energy=mean_core_total_energy,
            mean_abs_outputs=mean_abs_outputs,
            trial_small_gradient_fractions=trial_small_gradient_fractions,
            gradcheck=gradcheck, include_completed_counter=False,
        )
        _write_npz_atomic(checkpoint, payload)
        partial_checkpoint.unlink(missing_ok=True)
        print(f"[resume] finalized {point.architecture_key} without recompilation")
        return payload

    jax, jnp = _load_jax_runtime()
    from quannto.core.qnn_trainers import _build_bounds, _compute_num_parameters
    implemented_parameters = _compute_num_parameters(
        point.n_modes, point.layers, INCLUDE_INITIAL_SQUEEZING,
        INCLUDE_INITIAL_MIXING, IS_PASSIVE_GAUSSIAN,
    )
    if implemented_parameters != n_parameters:
        raise AssertionError("parameter protocol does not match QONN parameter count")
    bounds = _build_bounds(
        point.n_modes, point.layers, INCLUDE_INITIAL_SQUEEZING,
        INCLUDE_INITIAL_MIXING, IS_PASSIVE_GAUSSIAN,
    )
    qnn = _build_qnn(point, expectation_backend=expectation_backend)
    value_and_grad, scalar_cost = _compiled_functions(qnn, inputs, targets)

    print("\n" + "=" * 92)
    print(
        f"Trainability | ensemble={ensemble} | N={point.n_modes} | "
        f"L={point.layers} | K={point.k_subtractions} | "
        f"energy_scale={point.energy_scale:g}"
    )
    print("=" * 92)

    for trial in range(first_trial, trials):
        rng = np.random.default_rng(parameter_seed(
            base_seed=parameter_seed_value, ensemble=ensemble,
            n_modes=point.n_modes, layers=point.layers, trial=trial,
        ))
        parameter_vector = sample_initial_parameters(
            rng=rng, types=types, layers=point.layers, ensemble=ensemble,
            energy_scale=point.energy_scale, sigma_passive=SIGMA_PASSIVE,
            sigma_squeezing=SIGMA_SQUEEZING,
            sigma_displacement=SIGMA_DISPLACEMENT,
        )
        (loss, auxiliary), gradient = value_and_grad(
            jnp.asarray(parameter_vector, dtype=jnp.float64)
        )
        gradient_real = _real_checked(
            gradient, name=f"gradient[{point.architecture_key}, trial={trial}]"
        )
        prediction, norms, energy_per_mode, total_energy = auxiliary
        prediction_real = _real_checked(prediction, name="homodyne output")
        norms_real = _real_checked(norms, name="heralding norm")
        energy_per_mode_real = _real_checked(
            energy_per_mode, name="Gaussian-core energy per mode"
        )
        total_energy_real = _real_checked(
            total_energy, name="Gaussian-core total energy"
        )

        parameters[trial] = parameter_vector
        gradients[trial] = gradient_real
        losses[trial] = float(loss)
        gradient_norms[trial] = float(np.linalg.norm(gradient_real))
        mean_norms[trial] = float(np.mean(norms_real))
        median_norms[trial] = float(np.median(norms_real))
        tiny_norm_fractions[trial] = float(
            np.mean(norms_real < TINY_NORM_THRESHOLD)
        )
        nonpositive_norm_fractions[trial] = float(np.mean(norms_real <= 0.0))
        mean_core_energy_per_mode[trial] = float(np.mean(energy_per_mode_real))
        mean_core_total_energy[trial] = float(np.mean(total_energy_real))
        mean_abs_outputs[trial] = float(np.mean(np.abs(prediction_real)))
        trial_small_gradient_fractions[trial] = float(
            np.mean(np.abs(gradient_real) < 1e-6)
        )

        if (
            trial == 0
            and (
                point.n_modes, point.layers, point.k_subtractions,
                point.energy_scale,
            ) in GRADCHECK_CONFIGS
        ):
            gradcheck = _finite_difference_gradcheck(
                parameters=parameter_vector, autodiff_gradient=gradient_real,
                scalar_cost=scalar_cost, bounds=bounds,
                seed=777 + point.n_modes * 100 + point.layers * 10,
            )
            if not np.all(gradcheck["gradcheck_passed"]):
                raise AssertionError(
                    f"finite-difference gradcheck failed for {point.architecture_key}"
                )

        print(
            f"trial {trial + 1:03d}/{trials} | C={losses[trial]:.3e} | "
            f"||g||={gradient_norms[trial]:.3e} | "
            f"E_core/mode={mean_core_energy_per_mode[trial]:.3e} | "
            f"mean(norm)={mean_norms[trial]:.3e}",
            end="\r", flush=True,
        )
        partial_payload = _trial_payload(
            signature=signature,
            completed_trials=trial + 1,
            parameters=parameters,
            gradients=gradients,
            losses=losses,
            gradient_norms=gradient_norms,
            mean_norms=mean_norms,
            median_norms=median_norms,
            tiny_norm_fractions=tiny_norm_fractions,
            nonpositive_norm_fractions=nonpositive_norm_fractions,
            mean_core_energy_per_mode=mean_core_energy_per_mode,
            mean_core_total_energy=mean_core_total_energy,
            mean_abs_outputs=mean_abs_outputs,
            trial_small_gradient_fractions=trial_small_gradient_fractions,
            gradcheck=gradcheck,
            include_completed_counter=True,
        )
        _write_npz_atomic(partial_checkpoint, partial_payload)
    print()

    payload = _trial_payload(
        signature=signature,
        completed_trials=trials,
        parameters=parameters,
        gradients=gradients,
        losses=losses,
        gradient_norms=gradient_norms,
        mean_norms=mean_norms,
        median_norms=median_norms,
        tiny_norm_fractions=tiny_norm_fractions,
        nonpositive_norm_fractions=nonpositive_norm_fractions,
        mean_core_energy_per_mode=mean_core_energy_per_mode,
        mean_core_total_energy=mean_core_total_energy,
        mean_abs_outputs=mean_abs_outputs,
        trial_small_gradient_fractions=trial_small_gradient_fractions,
        gradcheck=gradcheck,
        include_completed_counter=False,
    )
    _write_npz_atomic(checkpoint, payload)
    partial_checkpoint.unlink(missing_ok=True)
    del qnn, value_and_grad, scalar_cost
    del jax, jnp
    _release_runtime_memory()
    return payload


# Aggregation, fits, and figures.
def _configuration_summary(
    point: ScanPoint,
    raw: Mapping[str, np.ndarray],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    types = parameter_types(point.n_modes, point.layers)
    summary = summarize_gradients(
        raw["gradients"], types=types,
        probe_indices=matched_probe_indices(point.n_modes),
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        small_gradient_thresholds=SMALL_GRADIENT_THRESHOLDS,
    )
    losses = np.asarray(raw["losses"], dtype=np.float64)
    energies = np.asarray(raw["mean_core_energy_per_mode"], dtype=np.float64)
    total_energies = np.asarray(raw["mean_core_total_energy"], dtype=np.float64)
    mean_norms = np.asarray(raw["mean_norms"], dtype=np.float64)
    median_norms = np.asarray(raw["median_norms"], dtype=np.float64)
    tiny = np.asarray(raw["tiny_norm_fractions"], dtype=np.float64)
    nonpositive = np.asarray(raw["nonpositive_norm_fractions"], dtype=np.float64)
    summary.update({
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses, ddof=1)),
        "mean_gaussian_core_photons_per_mode": float(np.mean(energies)),
        "stderr_gaussian_core_photons_per_mode": _stderr(energies),
        "mean_gaussian_core_total_photons": float(np.mean(total_energies)),
        "stderr_gaussian_core_total_photons": _stderr(total_energies),
        "mean_heralding_norm": float(np.mean(mean_norms)),
        "stderr_mean_heralding_norm": _stderr(mean_norms),
        "median_heralding_norm": float(np.median(median_norms)),
        "mean_tiny_norm_fraction": float(np.mean(tiny)),
        "mean_nonpositive_norm_fraction": float(np.mean(nonpositive)),
        "mean_abs_output": float(np.mean(raw["mean_abs_outputs"])),
    })
    return summary


def _trial_rows(
    point: ScanPoint, raw: Mapping[str, np.ndarray], *, ensemble: str,
) -> list[dict[str, Any]]:
    rows = []
    for trial in range(len(raw["losses"])):
        rows.append({
            "architecture_key": point.architecture_key,
            "ensemble": ensemble,
            "N": point.n_modes,
            "L": point.layers,
            "K": point.k_subtractions,
            "energy_scale": point.energy_scale,
            "trial": trial,
            "loss": float(raw["losses"][trial]),
            "gradient_norm": float(raw["gradient_norms"][trial]),
            "mean_grad_sq_per_param": float(np.mean(raw["gradients"][trial] ** 2)),
            "fraction_abs_grad_lt_1em6": float(
                raw["trial_small_gradient_fractions_1em6"][trial]
            ),
            "mean_gaussian_core_photons_per_mode": float(
                raw["mean_core_energy_per_mode"][trial]
            ),
            "mean_gaussian_core_total_photons": float(
                raw["mean_core_total_energy"][trial]
            ),
            "mean_heralding_norm": float(raw["mean_norms"][trial]),
            "median_heralding_norm": float(raw["median_norms"][trial]),
            "tiny_norm_fraction": float(raw["tiny_norm_fractions"][trial]),
            "nonpositive_norm_fraction": float(
                raw["nonpositive_norm_fractions"][trial]
            ),
            "mean_abs_output": float(raw["mean_abs_outputs"][trial]),
        })
    return rows


def _gradcheck_rows(
    point: ScanPoint, raw: Mapping[str, np.ndarray], *, ensemble: str,
) -> list[dict[str, Any]]:
    rows = []
    for position, index in enumerate(raw["gradcheck_indices"]):
        rows.append({
            "architecture_key": point.architecture_key,
            "ensemble": ensemble,
            "N": point.n_modes,
            "L": point.layers,
            "K": point.k_subtractions,
            "parameter_index": int(index),
            "autodiff_gradient": float(raw["gradcheck_autodiff"][position]),
            "finite_difference_gradient": float(
                raw["gradcheck_finite_difference"][position]
            ),
            "absolute_error": float(raw["gradcheck_absolute_error"][position]),
            "relative_error": float(raw["gradcheck_relative_error"][position]),
            "passed": bool(raw["gradcheck_passed"][position]),
        })
    return rows


def _fit_sweeps(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "interpretation": (
            "Exploratory finite-size fits of the three pre-registered matched "
            "derivatives only. Width/depth semilog slopes and energy log-log "
            "slopes do not establish or exclude an asymptotic barren plateau."
        ),
        "ladder_fit_omitted": (
            "No fit is assigned to ladder order K: conditional normalization "
            "and heralding probability change with K, so K is not a circuit-"
            "size scaling variable for a barren-plateau claim."
        ),
        "fits": {},
    }
    metrics = tuple(
        f"probe_{kind}__gradient_variance"
        for kind in ("passive", "squeezing", "displacement")
    )
    groups = sorted({(str(row["sweep"]), str(row["family"])) for row in rows})
    for sweep, family in groups:
        if sweep == "ladder":
            continue
        selected = sorted(
            (row for row in rows if row["sweep"] == sweep and row["family"] == family),
            key=lambda row: float(row["size"]),
        )
        if sweep == "energy":
            x_values = [row["mean_gaussian_core_photons_per_mode"] for row in selected]
            fit_function = loglog_fit
            x_description = "measured mean Gaussian-core photons per mode"
        elif sweep in {"width", "depth"}:
            x_values = [row["size"] for row in selected]
            fit_function = semilog_fit
            x_description = {"width": "N", "depth": "L"}[sweep]
        else:
            continue
        key = f"{sweep}:{family}"
        metric_fits = {
            metric: fit_function(x_values, [row[metric] for row in selected])
            for metric in metrics
        }
        result["fits"][key] = {"x": x_description, **metric_fits}
    return result


def _plot_gradient_variance(
    rows: Sequence[Mapping[str, Any]], output_directory: Path, *, show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[warning] matplotlib unavailable; skipping plots: {error}")
        return

    available = [
        sweep for sweep in ("width", "depth")
        if any(r["sweep"] == sweep for r in rows)
    ]
    if not available:
        return
    figure, axes = plt.subplots(
        len(available), 3, figsize=(14.1, 4.0 * len(available)), squeeze=False,
    )
    family_style = {
        "gaussian": (colors[0], linestyles[0], "o", "Gaussian control ($K=0$)"),
        "subtracted": (colors[1], linestyles[1], "s", "QONN ($K=1$ per layer)"),
    }
    probe_labels = {
        "passive": "Passive probe",
        "squeezing": "Squeezing probe",
        "displacement": "Displacement probe",
    }
    panel_index = 0
    for row_index, sweep in enumerate(available):
        for column, kind in enumerate(("passive", "squeezing", "displacement")):
            axis = axes[row_index, column]
            for family in ("gaussian", "subtracted"):
                selected = sorted(
                    (r for r in rows if r["sweep"] == sweep and r["family"] == family),
                    key=lambda r: float(r["size"]),
                )
                if not selected:
                    continue
                color, linestyle, marker, label = family_style[family]
                x = np.asarray([r["size"] for r in selected], dtype=float)
                metric = f"probe_{kind}__gradient_variance"
                y = np.asarray([r[metric] for r in selected], dtype=float)
                lower = np.asarray([
                    r[f"{metric}_ci95_low"] for r in selected
                ], dtype=float)
                upper = np.asarray([
                    r[f"{metric}_ci95_high"] for r in selected
                ], dtype=float)
                positive = y > 0
                if not np.any(positive):
                    continue
                yerr = np.vstack((
                    np.maximum(y[positive] - lower[positive], 0.0),
                    np.maximum(upper[positive] - y[positive], 0.0),
                ))
                axis.errorbar(
                    x[positive], y[positive], yerr=yerr,
                    color=color, linestyle=linestyle, linewidth=1.8,
                    marker=marker, markersize=6, capsize=3, label=label,
                )
            axis.set_yscale("log")
            axis.set_xlabel({"width": "$N$", "depth": "$L$"}[sweep])
            axis.set_xticks(sorted({
                float(r["size"]) for r in rows if r["sweep"] == sweep
            }))
            axis.set_ylabel(r"$\mathrm{Var}_{\theta}(\partial_{\theta_i} C)$")
            relevant_rows = [r for r in rows if r["sweep"] == sweep]
            fixed_layers = sorted({int(r["L"]) for r in relevant_rows})
            fixed_modes = sorted({int(r["N"]) for r in relevant_rows})
            title = {
                "width": (
                    f"Width ($L={fixed_layers[0]}$)"
                    if len(fixed_layers) == 1 else "Width"
                ),
                "depth": (
                    f"Depth ($N={fixed_modes[0]}$)"
                    if len(fixed_modes) == 1 else "Depth"
                ),
            }[sweep]
            axis.set_title(
                f"({'abcdef'[panel_index]}) {title}: {probe_labels[kind]}",
                fontsize=12,
            )
            panel_index += 1
            axis.tick_params(axis="both", labelsize=10)
            axis.grid(linestyle="--", linewidth=0.4, which="both")
            axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(
        output_directory / "paper_style_gradient_variance.pdf", bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "paper_style_gradient_variance.png",
        dpi=220, bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(figure)


def _plot_nongaussian_and_energy(
    rows: Sequence[Mapping[str, Any]], output_directory: Path, *, show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    available = [
        sweep for sweep in ("ladder", "energy")
        if any(r["sweep"] == sweep for r in rows)
    ]
    if not available:
        return
    figure, axes = plt.subplots(
        1, len(available), figsize=(5.3 * len(available), 4.2), squeeze=False,
    )
    for panel, sweep in enumerate(available):
        axis = axes.flat[panel]
        selected = sorted(
            (r for r in rows if r["sweep"] == sweep),
            key=lambda r: float(r["size"]),
        )
        if sweep == "energy":
            x = np.asarray([
                r["mean_gaussian_core_photons_per_mode"] for r in selected
            ], dtype=float)
            axis.set_xscale("log")
            axis.set_xlabel("Mean Gaussian-core photons per mode")
            title = "Energy dependence"
        else:
            x = np.asarray([r["size"] for r in selected], dtype=float)
            axis.set_xlabel("$K$")
            axis.set_xticks(x)
            title = "Ladder-order diagnostic"
        for index, kind in enumerate(("passive", "squeezing", "displacement")):
            metric = f"probe_{kind}__gradient_variance"
            y = np.asarray([r[metric] for r in selected], dtype=float)
            low = np.asarray([r[f"{metric}_ci95_low"] for r in selected])
            high = np.asarray([r[f"{metric}_ci95_high"] for r in selected])
            positive = y > 0
            if not np.any(positive):
                continue
            axis.errorbar(
                x[positive], y[positive],
                yerr=np.vstack((
                    np.maximum(y[positive] - low[positive], 0.0),
                    np.maximum(high[positive] - y[positive], 0.0),
                )),
                color=colors[index], linestyle=linestyles[index],
                marker=("o", "s", "D")[index], linewidth=1.8,
                markersize=6, capsize=3, label=kind.capitalize(),
            )
        axis.set_yscale("log")
        axis.set_ylabel(r"$\mathrm{Var}_{\theta}(\partial_{\theta_i} C)$")
        axis.set_title(f"({'ab'[panel]}) {title}")
        axis.grid(linestyle="--", linewidth=0.4, which="both")
        axis.legend(fontsize=9)
    figure.tight_layout()
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        figure.savefig(
            output_directory / f"non_gaussian_energy_gradient_variance.{suffix}",
            bbox_inches="tight", **kwargs,
        )
    if show:
        plt.show()
    plt.close(figure)


def _plot_aggregate_diagnostics(
    rows: Sequence[Mapping[str, Any]], output_directory: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    sweeps = [s for s in SWEEPS if any(r["sweep"] == s for r in rows)]
    if not sweeps:
        return
    figure, axes = plt.subplots(
        2, len(sweeps), figsize=(4.2 * len(sweeps), 7.0), squeeze=False,
    )
    styles = {
        "gaussian": (colors[0], linestyles[0], "o", "Gaussian"),
        "subtracted": (colors[1], linestyles[1], "s", "QONN"),
        "ladder": (colors[2], linestyles[2], "D", "Ladder"),
    }
    for column, sweep in enumerate(sweeps):
        for family in sorted({r["family"] for r in rows if r["sweep"] == sweep}):
            selected = sorted(
                (r for r in rows if r["sweep"] == sweep and r["family"] == family),
                key=lambda r: float(r["size"]),
            )
            color, linestyle, marker, label = styles[str(family)]
            x = [r["size"] for r in selected]
            axes[0, column].plot(
                x, [r["mean_total_grad_norm"] for r in selected],
                color=color, linestyle=linestyle, marker=marker, label=label,
            )
            axes[1, column].plot(
                x, [r["empirically_inactive_parameter_fraction"] for r in selected],
                color=color, linestyle=linestyle, marker=marker, label=label,
            )
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel(r"Mean $\|\nabla C\|_2$")
        axes[1, column].set_ylabel("Inactive-coordinate fraction")
        axes[1, column].set_ylim(-0.03, 1.03)
        axes[0, column].set_title(sweep.capitalize())
        for row_index in (0, 1):
            axes[row_index, column].set_xlabel({
                "width": "$N$", "depth": "$L$", "ladder": "$K$",
                "energy": "Energy scale",
            }[sweep])
            axes[row_index, column].grid(linestyle="--", linewidth=0.4, which="both")
            axes[row_index, column].legend(fontsize=8)
    figure.tight_layout()
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        figure.savefig(
            output_directory / f"aggregate_gradient_diagnostics.{suffix}",
            bbox_inches="tight", **kwargs,
        )
    plt.close(figure)


def _plot_parameter_families(
    rows: Sequence[Mapping[str, Any]], output_directory: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    selected_sweeps = [
        sweep for sweep in ("width", "depth", "ladder")
        if any(r["sweep"] == sweep for r in rows)
    ]
    if not selected_sweeps:
        return
    figure, axes = plt.subplots(
        1, len(selected_sweeps), figsize=(5.0 * len(selected_sweeps), 4.2),
        squeeze=False,
    )
    family_labels = {
        "passive": "Passive optics",
        "squeezing": "Squeezing",
        "displacement": "Displacement",
    }
    for panel, sweep in enumerate(selected_sweeps):
        axis = axes.flat[panel]
        relevant = [r for r in rows if r["sweep"] == sweep]
        if sweep in {"width", "depth"}:
            relevant = [r for r in relevant if r["family"] == "subtracted"]
        relevant = sorted(relevant, key=lambda r: float(r["size"]))
        for index, kind in enumerate(("passive", "squeezing", "displacement")):
            axis.plot(
                [r["size"] for r in relevant],
                [r[f"{kind}__median_per_param_var"] for r in relevant],
                color=colors[index], linestyle=linestyles[index], linewidth=1.8,
                marker=("o", "s", "D")[index], markersize=6,
                label=family_labels[kind],
            )
        axis.set_yscale("log")
        axis.set_xlabel({"width": "$N$", "depth": "$L$", "ladder": "$K$"}[sweep])
        axis.set_ylabel(r"Median $\mathrm{Var}(\partial_\theta C)$")
        axis.set_title({
            "width": "QONN width", "depth": "QONN depth",
            "ladder": "Ladder order",
        }[sweep])
        axis.grid(linestyle="--", linewidth=0.4, which="both")
        axis.legend(fontsize=10)
    figure.tight_layout()
    figure.savefig(
        output_directory / "parameter_family_gradient_variance.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "parameter_family_gradient_variance.png",
        dpi=220, bbox_inches="tight",
    )
    plt.close(figure)


def _plot_physical_controls(
    rows: Sequence[Mapping[str, Any]], output_directory: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    sweeps = [sweep for sweep in SWEEPS if any(r["sweep"] == sweep for r in rows)]
    figure, axes = plt.subplots(
        2, len(sweeps), figsize=(4.3 * len(sweeps), 7.2), squeeze=False,
    )
    for column, sweep in enumerate(sweeps):
        families = sorted({str(r["family"]) for r in rows if r["sweep"] == sweep})
        for family in families:
            selected = sorted(
                (r for r in rows if r["sweep"] == sweep and r["family"] == family),
                key=lambda r: float(r["size"]),
            )
            color, linestyle, marker, label = {
                "gaussian": (colors[0], linestyles[0], "o", "Gaussian"),
                "subtracted": (colors[1], linestyles[1], "s", "QONN"),
                "ladder": (colors[2], linestyles[2], "D", "Ladder sweep"),
            }[family]
            x = [r["size"] for r in selected]
            axes[0, column].plot(
                x, [r["mean_gaussian_core_photons_per_mode"] for r in selected],
                color=color, linestyle=linestyle, marker=marker, label=label,
            )
            axes[1, column].plot(
                x, [r["mean_heralding_norm"] for r in selected],
                color=color, linestyle=linestyle, marker=marker, label=label,
            )
        axes[0, column].set_title(sweep.capitalize())
        axes[0, column].set_ylabel("Core photons per mode")
        axes[1, column].set_ylabel("Mean heralding norm")
        axes[1, column].set_yscale("log")
        for row_index in (0, 1):
            axes[row_index, column].set_xlabel({
                "width": "$N$", "depth": "$L$", "ladder": "$K$",
                "energy": "Energy scale",
            }[sweep])
            axes[row_index, column].grid(
                linestyle="--", linewidth=0.4, which="both",
            )
            axes[row_index, column].legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(
        output_directory / "physical_control_diagnostics.pdf", bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "physical_control_diagnostics.png",
        dpi=220, bbox_inches="tight",
    )
    plt.close(figure)


# Failure-tolerant protocol entry point.
def run(args: argparse.Namespace) -> Path:
    from quannto.tasks.revision.trainability_supervisor import run_supervised

    return run_supervised(sys.modules[__name__], args)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-registered finite-size QONN gradient diagnostic. "
            "This evaluates random initializations and does not train models."
        )
    )
    parser.add_argument(
        "--ensemble", choices=ENSEMBLES, default="energy_controlled",
        help="Initialization ensemble; run both ensembles in separate processes.",
    )
    parser.add_argument(
        "--profile", choices=tuple(SCAN_PROFILES), default="manuscript",
        help="Recorded finite-size grid; extended_mn5 adds costly N/L/K points.",
    )
    parser.add_argument(
        "--sweeps", default="all",
        help="'all' or comma-separated subset of width,depth,ladder,energy",
    )
    parser.add_argument(
        "--widths", default=None,
        help="Optional comma-separated override of the profile widths.",
    )
    parser.add_argument(
        "--depths", default=None,
        help="Optional comma-separated override of the profile depths.",
    )
    parser.add_argument(
        "--ladder-orders", default=None,
        help="Optional comma-separated override of the profile ladder orders.",
    )
    parser.add_argument(
        "--energy-scales", default=None,
        help="Optional comma-separated override of the profile energy scales.",
    )
    parser.add_argument("--fixed-modes-for-depth", type=int, default=None)
    parser.add_argument("--fixed-modes-for-ladder", type=int, default=None)
    parser.add_argument("--fixed-modes-for-energy", type=int, default=None)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--bootstrap-repetitions", type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dataset-seed", type=int, default=DEFAULT_DATASET_SEED)
    parser.add_argument("--parameter-seed", type=int, default=DEFAULT_PARAMETER_SEED)
    parser.add_argument(
        "--expectation-backend",
        choices=("loop_matchings", "subset_dp"),
        default="loop_matchings",
        help="Use loop_matchings for the manuscript protocol.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse final checkpoints and continue per-trial partial checkpoints.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly recompute and replace matching configuration checkpoints.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Three-trial workflow smoke test; its values are not paper results.",
    )
    parser.add_argument(
        "--isolation", choices=("process", "none"), default="process",
        help="Use one process per architecture (recommended and OOM-tolerant).",
    )
    parser.add_argument(
        "--configuration-timeout-minutes", type=float, default=0.0,
        help="Optional per-architecture timeout; zero disables the timeout.",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Revalidate existing checkpoints and rebuild tables/figures only.",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help=(
            "Return exit status zero after all attempts even if some workers "
            "failed; RUN_STATUS.json remains authoritative."
        ),
    )
    parser.add_argument("--worker-request", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_request is not None:
        from quannto.tasks.revision.trainability_supervisor import (
            run_worker_request,
        )
        run_worker_request(sys.modules[__name__], Path(args.worker_request))
        return
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.configuration_timeout_minutes < 0:
        parser.error("--configuration-timeout-minutes must be non-negative")
    requested_sweeps = SWEEPS if args.sweeps == "all" else tuple(
        item.strip() for item in args.sweeps.split(",") if item.strip()
    )
    unknown = sorted(set(requested_sweeps) - set(SWEEPS))
    if unknown:
        parser.error(f"unknown sweeps: {', '.join(unknown)}")
    args.sweeps = "all" if args.sweeps == "all" else ",".join(requested_sweeps)
    run(args)


if __name__ == "__main__":
    cli()
