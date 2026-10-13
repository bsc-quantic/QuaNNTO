"""Dependency-light protocol helpers for the finite-size trainability scan.

The JAX/QONN runner lives in :mod:`quannto.tasks.revision.trainability`.  This
module deliberately depends only on NumPy so the experimental design,
initialization ensembles, and statistical summaries can be unit-tested without
compiling a photonic circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = 4
ENSEMBLES = ("energy_controlled", "legacy_uniform")
SWEEPS = ("width", "depth", "ladder", "energy")
EMPIRICALLY_INACTIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ScanPoint:
    """One plotted point in a trainability sweep.

    ``family`` distinguishes the Gaussian control from the QONN with one
    subtraction per layer.  The ladder sweep uses ``family='ladder'`` and the
    energy sweep uses ``family='subtracted'``.
    """

    sweep: str
    family: str
    size: float
    n_modes: int
    layers: int
    k_subtractions: int
    energy_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.sweep not in SWEEPS:
            raise ValueError(f"unknown sweep {self.sweep!r}")
        if self.n_modes < 1 or self.layers < 1:
            raise ValueError("n_modes and layers must be positive")
        if not 0 <= self.k_subtractions <= self.n_modes:
            raise ValueError("k_subtractions must satisfy 0 <= K <= N")
        if self.energy_scale <= 0:
            raise ValueError("energy_scale must be positive")

    @property
    def ladder_modes(self) -> list[list[int]]:
        subset = list(range(self.k_subtractions))
        return [subset.copy() for _ in range(self.layers)]

    @property
    def architecture_key(self) -> str:
        scale = format(self.energy_scale, ".8g").replace(".", "p")
        return (
            f"N{self.n_modes}_L{self.layers}_K{self.k_subtractions}"
            f"_es{scale}"
        )


def parse_csv_numbers(value: str, *, cast=float) -> tuple:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one comma-separated value is required")
    return values


def build_scan_points(
    *,
    sweeps: Sequence[str] = SWEEPS,
    widths: Sequence[int] = (2, 3, 4, 5),
    depths: Sequence[int] = (1, 2, 3),
    ladder_orders: Sequence[int] = (0, 1, 2, 3, 4),
    energy_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    fixed_modes_for_depth: int = 2,
    fixed_modes_for_ladder: int = 4,
    fixed_modes_for_energy: int = 3,
    ensemble: str = "energy_controlled",
) -> tuple[ScanPoint, ...]:
    """Construct the pre-registered, mutually interpretable scan points.

    Width and depth contain paired Gaussian and single-subtraction families.
    The ladder-order sweep changes only K at fixed N and L.  The energy sweep
    is defined only for the energy-controlled ensemble because rescaling a
    historical uniform initialization would no longer represent that ensemble.
    """

    if ensemble not in ENSEMBLES:
        raise ValueError(f"ensemble must be one of {ENSEMBLES}")
    requested = tuple(sweeps)
    unknown = sorted(set(requested) - set(SWEEPS))
    if unknown:
        raise ValueError(f"unknown sweeps: {', '.join(unknown)}")
    if any(n < 1 for n in widths):
        raise ValueError("all widths must be positive")
    if any(l < 1 for l in depths):
        raise ValueError("all depths must be positive")
    if any(k < 0 or k > fixed_modes_for_ladder for k in ladder_orders):
        raise ValueError("ladder orders must satisfy 0 <= K <= fixed_modes_for_ladder")
    if any(scale <= 0 for scale in energy_scales):
        raise ValueError("energy scales must be positive")

    points: list[ScanPoint] = []
    if "width" in requested:
        for family, k in (("gaussian", 0), ("subtracted", 1)):
            for n_modes in widths:
                points.append(ScanPoint(
                    "width", family, float(n_modes), int(n_modes), 1, k,
                ))

    if "depth" in requested:
        for family, k in (("gaussian", 0), ("subtracted", 1)):
            for layers in depths:
                points.append(ScanPoint(
                    "depth", family, float(layers), fixed_modes_for_depth,
                    int(layers), k,
                ))

    if "ladder" in requested:
        for k in ladder_orders:
            points.append(ScanPoint(
                "ladder", "ladder", float(k), fixed_modes_for_ladder, 1,
                int(k),
            ))

    if "energy" in requested and ensemble == "energy_controlled":
        for scale in energy_scales:
            points.append(ScanPoint(
                "energy", "subtracted", float(scale),
                fixed_modes_for_energy, 1, 1, float(scale),
            ))

    return tuple(points)


def unique_architectures(points: Iterable[ScanPoint]) -> tuple[ScanPoint, ...]:
    """Deduplicate scan points that share the exact same raw experiment."""

    unique: dict[str, ScanPoint] = {}
    for point in points:
        unique.setdefault(point.architecture_key, point)
    return tuple(unique.values())


def make_fixed_scalar_dataset(
    *,
    batch_size: int = 32,
    low: float = -0.75,
    high: float = 0.75,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one fixed scalar identity-regression dataset for every scan.

    The endpoints are included explicitly and the interior points are drawn
    once from a recorded generator.  Varying a circuit parameter seed never
    changes the cost function being differentiated.
    """

    if batch_size < 3:
        raise ValueError("batch_size must be at least three")
    if not low < high:
        raise ValueError("low must be smaller than high")
    rng = np.random.default_rng(seed)
    interior = rng.uniform(low, high, size=batch_size - 2)
    x = np.sort(np.concatenate(([low, high], interior))).astype(np.float64)
    x = x[:, None]
    return x, x.copy()


def dataset_sha256(inputs: np.ndarray, targets: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (inputs, targets):
        arr = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def parameter_types(n_modes: int, layers: int) -> tuple[str, ...]:
    """Parameter families in the order consumed by the active QONN layer."""

    types: list[str] = []
    for _ in range(layers):
        types.extend(["passive"] * (2 * n_modes**2))
        types.extend(["squeezing"] * n_modes)
        types.extend(["displacement"] * (2 * n_modes))
    return tuple(types)


def matched_probe_indices(n_modes: int) -> dict[str, int]:
    """One architecture-matched first-block probe per parameter family.

    Barren-plateau definitions concern the variance of a specified derivative.
    These indices complement aggregate/median metrics with the same relative
    Bloch--Messiah parameter locations for every N and L.
    """

    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    return {
        "passive": 0,
        "squeezing": 2 * n_modes**2,
        "displacement": 2 * n_modes**2 + n_modes,
    }


def parameter_seed(
    *, base_seed: int, ensemble: str, n_modes: int, layers: int, trial: int,
) -> np.random.SeedSequence:
    """Seed parameters independently of K and energy scale for paired scans."""

    if ensemble not in ENSEMBLES:
        raise ValueError(f"ensemble must be one of {ENSEMBLES}")
    ensemble_code = ENSEMBLES.index(ensemble) + 1
    return np.random.SeedSequence(
        [int(base_seed), ensemble_code, int(n_modes), int(layers), int(trial)]
    )


def sample_initial_parameters(
    *,
    rng: np.random.Generator,
    types: Sequence[str],
    layers: int,
    ensemble: str,
    energy_scale: float = 1.0,
    sigma_passive: float = 0.10,
    sigma_squeezing: float = 0.05,
    sigma_displacement: float = 0.05,
    squeezing_bound: float = float(np.log(5.5)),
) -> np.ndarray:
    """Sample one of the two explicitly recorded initialization ensembles.

    ``legacy_uniform`` reproduces the historical *distribution* ``U[0,1)``.
    ``energy_controlled`` keeps passive angles independent of depth and scales
    only energy-injecting squeezing/displacement parameters by ``1/sqrt(L)``.
    The energy sweep multiplies those two families by ``energy_scale``.
    """

    if ensemble not in ENSEMBLES:
        raise ValueError(f"ensemble must be one of {ENSEMBLES}")
    if layers < 1 or energy_scale <= 0:
        raise ValueError("layers and energy_scale must be positive")

    if ensemble == "legacy_uniform":
        if not np.isclose(energy_scale, 1.0):
            raise ValueError("legacy_uniform does not define an energy-scale sweep")
        return rng.random(len(types), dtype=np.float64)

    energy_depth_scale = energy_scale / np.sqrt(float(layers))
    values = np.empty(len(types), dtype=np.float64)
    for index, kind in enumerate(types):
        if kind == "passive":
            sigma = sigma_passive
        elif kind == "squeezing":
            sigma = sigma_squeezing * energy_depth_scale
        elif kind == "displacement":
            sigma = sigma_displacement * energy_depth_scale
        else:
            raise ValueError(f"unknown parameter type {kind!r}")
        values[index] = rng.normal(0.0, sigma)

    squeezing = np.asarray(types) == "squeezing"
    values[squeezing] = np.clip(
        values[squeezing], -squeezing_bound, squeezing_bound,
    )
    return values


def bootstrap_gradient_variance_ci(
    gradients: np.ndarray,
    *,
    statistic: str = "median",
    confidence: float = 0.95,
    repetitions: int = 1000,
    seed: int = 9347,
) -> tuple[float, float]:
    """Bootstrap a confidence interval over initialization trials."""

    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("gradients must have shape (trials, parameters), trials >= 2")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    if repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap repetitions or confidence")

    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    n_trials = gradients.shape[0]
    reducer = np.mean if statistic == "mean" else np.median
    for repetition in range(repetitions):
        indices = rng.integers(0, n_trials, size=n_trials)
        per_parameter = np.var(gradients[indices], axis=0, ddof=1)
        estimates[repetition] = reducer(per_parameter)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, (alpha, 1.0 - alpha))
    return float(lower), float(upper)


def bootstrap_scalar_variance_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    repetitions: int = 1000,
    seed: int = 9347,
) -> tuple[float, float]:
    """Bootstrap the sample variance of one pre-registered derivative.

    Barren-plateau statements concern ``Var_theta[dC/dtheta_i]`` for a
    specified coordinate.  This helper therefore resamples initialization
    trials only; it never pools different parameter coordinates.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be one-dimensional with at least two trials")
    if repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap repetitions or confidence")
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite values cannot be bootstrapped")

    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = rng.integers(0, values.size, size=values.size)
        estimates[repetition] = np.var(values[indices], ddof=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, (alpha, 1.0 - alpha))
    return float(lower), float(upper)


def summarize_gradients(
    gradients: np.ndarray,
    *,
    types: Sequence[str],
    probe_indices: Mapping[str, int] | None = None,
    bootstrap_repetitions: int = 1000,
    bootstrap_seed: int = 9347,
    small_gradient_thresholds: Sequence[float] = (1e-8, 1e-6, 1e-4),
    inactive_tolerance: float = EMPIRICALLY_INACTIVE_TOLERANCE,
) -> dict[str, float | int]:
    """Return probe-primary and aggregate trainability diagnostics.

    The matched probe variances are the co-primary quantities.  Aggregates
    across parameters are retained as secondary diagnostics because a local
    scalar cost can contain exactly or numerically inactive coordinates whose
    fraction changes with architecture size.
    """

    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("gradients must have shape (trials, parameters), trials >= 2")
    if gradients.shape[1] != len(types):
        raise ValueError("types must contain one entry per parameter")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("non-finite gradients cannot be summarized")
    if inactive_tolerance < 0:
        raise ValueError("inactive_tolerance must be non-negative")

    per_parameter_mean = np.mean(gradients, axis=0)
    per_parameter_var = np.var(gradients, axis=0, ddof=1)
    per_parameter_var_pop = np.var(gradients, axis=0, ddof=0)
    per_parameter_second_moment = np.mean(gradients**2, axis=0)
    moment_error = np.max(np.abs(
        per_parameter_second_moment
        - (per_parameter_var_pop + per_parameter_mean**2)
    ))
    per_trial_mean_sq = np.mean(gradients**2, axis=1)
    per_trial_norm = np.linalg.norm(gradients, axis=1)
    empirically_inactive = (
        np.max(np.abs(gradients), axis=0) <= inactive_tolerance
    )

    median_ci = bootstrap_gradient_variance_ci(
        gradients, statistic="median", repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    mean_ci = bootstrap_gradient_variance_ci(
        gradients, statistic="mean", repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 1,
    )

    summary: dict[str, float | int] = {
        "num_trials": int(gradients.shape[0]),
        "num_parameters": int(gradients.shape[1]),
        "mean_grad_sq_per_param": float(np.mean(per_trial_mean_sq)),
        "median_trial_grad_sq_per_param": float(np.median(per_trial_mean_sq)),
        "mean_total_grad_norm": float(np.mean(per_trial_norm)),
        "median_total_grad_norm": float(np.median(per_trial_norm)),
        "mean_abs_grad": float(np.mean(np.abs(gradients))),
        "median_abs_grad": float(np.median(np.abs(gradients))),
        "mean_per_param_var": float(np.mean(per_parameter_var)),
        "median_per_param_var": float(np.median(per_parameter_var)),
        "q25_per_param_var": float(np.quantile(per_parameter_var, 0.25)),
        "q75_per_param_var": float(np.quantile(per_parameter_var, 0.75)),
        "mean_per_param_var_ci95_low": mean_ci[0],
        "mean_per_param_var_ci95_high": mean_ci[1],
        "median_per_param_var_ci95_low": median_ci[0],
        "median_per_param_var_ci95_high": median_ci[1],
        "mean_abs_per_param_mean_grad": float(np.mean(np.abs(per_parameter_mean))),
        "max_abs_per_param_mean_grad": float(np.max(np.abs(per_parameter_mean))),
        "moment_identity_max_abs_error": float(moment_error),
        "empirically_inactive_tolerance": float(inactive_tolerance),
        "empirically_inactive_parameter_count": int(
            np.count_nonzero(empirically_inactive)
        ),
        "empirically_inactive_parameter_fraction": float(
            np.mean(empirically_inactive)
        ),
    }

    for threshold in small_gradient_thresholds:
        label = format(float(threshold), ".0e").replace("-", "m")
        summary[f"fraction_abs_grad_lt_{label}"] = float(
            np.mean(np.abs(gradients) < threshold)
        )

    type_array = np.asarray(types)
    for kind in ("passive", "squeezing", "displacement"):
        mask = type_array == kind
        if not np.any(mask):
            continue
        block = gradients[:, mask]
        block_var = np.var(block, axis=0, ddof=1)
        summary[f"{kind}__num_parameters"] = int(np.count_nonzero(mask))
        summary[f"{kind}__mean_grad_sq_per_param"] = float(np.mean(block**2))
        summary[f"{kind}__mean_abs_grad"] = float(np.mean(np.abs(block)))
        summary[f"{kind}__mean_per_param_var"] = float(np.mean(block_var))
        summary[f"{kind}__median_per_param_var"] = float(np.median(block_var))
        family_inactive = empirically_inactive[mask]
        summary[f"{kind}__empirically_inactive_count"] = int(
            np.count_nonzero(family_inactive)
        )
        summary[f"{kind}__empirically_inactive_fraction"] = float(
            np.mean(family_inactive)
        )

    if probe_indices:
        for name, index in probe_indices.items():
            if not 0 <= int(index) < gradients.shape[1]:
                raise ValueError(f"probe index {index} for {name!r} is out of range")
            probe = gradients[:, int(index)]
            summary[f"probe_{name}__parameter_index"] = int(index)
            summary[f"probe_{name}__mean_gradient"] = float(np.mean(probe))
            summary[f"probe_{name}__gradient_variance"] = float(
                np.var(probe, ddof=1)
            )
            summary[f"probe_{name}__gradient_second_moment"] = float(
                np.mean(probe**2)
            )
            probe_ci = bootstrap_scalar_variance_ci(
                probe,
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed + 101 + int(index),
            )
            summary[f"probe_{name}__gradient_variance_ci95_low"] = probe_ci[0]
            summary[f"probe_{name}__gradient_variance_ci95_high"] = probe_ci[1]
            summary[f"probe_{name}__mean_abs_gradient"] = float(
                np.mean(np.abs(probe))
            )

    return summary


def parameter_diagnostic_rows(
    gradients: np.ndarray,
    *,
    types: Sequence[str],
    inactive_tolerance: float = EMPIRICALLY_INACTIVE_TOLERANCE,
) -> list[dict[str, float | int | str | bool]]:
    """Return one auditable row per parameter coordinate."""

    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("gradients must have shape (trials, parameters)")
    if gradients.shape[1] != len(types):
        raise ValueError("types must contain one entry per parameter")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("non-finite gradients cannot be summarized")

    rows: list[dict[str, float | int | str | bool]] = []
    for index, kind in enumerate(types):
        values = gradients[:, index]
        maximum = float(np.max(np.abs(values)))
        rows.append({
            "parameter_index": int(index),
            "parameter_family": str(kind),
            "mean_gradient": float(np.mean(values)),
            "mean_abs_gradient": float(np.mean(np.abs(values))),
            "gradient_variance": float(np.var(values, ddof=1)),
            "gradient_second_moment": float(np.mean(values**2)),
            "max_abs_gradient": maximum,
            "empirically_inactive": bool(maximum <= inactive_tolerance),
            "inactive_tolerance": float(inactive_tolerance),
        })
    return rows


def semilog_fit(
    x_values: Sequence[float], y_values: Sequence[float],
) -> dict[str, float | None] | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if np.count_nonzero(mask) < 3:
        return None
    x, log_y = x[mask], np.log(y[mask])
    slope, intercept = np.polyfit(x, log_y, deg=1)
    prediction = slope * x + intercept
    residual = float(np.sum((log_y - prediction) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    return {
        "slope": float(slope),
        "decay_rate": float(-slope),
        "intercept": float(intercept),
        "r2_log_fit": None if total == 0 else 1.0 - residual / total,
    }


def loglog_fit(
    x_values: Sequence[float], y_values: Sequence[float],
) -> dict[str, float | None] | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.count_nonzero(mask) < 3:
        return None
    log_x, log_y = np.log(x[mask]), np.log(y[mask])
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    prediction = slope * log_x + intercept
    residual = float(np.sum((log_y - prediction) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    return {
        "power_exponent": float(slope),
        "intercept": float(intercept),
        "r2_loglog_fit": None if total == 0 else 1.0 - residual / total,
    }


def canonical_json(value: Mapping | Sequence | str | int | float | bool | None) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
