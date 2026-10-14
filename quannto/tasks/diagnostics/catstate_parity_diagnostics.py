#!/usr/bin/env python3
"""Diagnostics for the even-cat state-engineering checkpoints.

Run from the QuaNNTO repository root:
    python -m quannto.tasks.diagnostics.catstate_parity_diagnostics

The script reconstructs the same reduced output states used by the independent
fidelity/Wigner evaluator and checks whether the low-moment-loss solutions are
being attracted to an opposite-parity (odd-cat-like) region.

Outputs:
  * one-line diagnostics for all Table-III architectures
  * pairwise Uhlmann fidelities among the three low-F+ solutions
  * even-vs-odd ideal-cat moment comparison for the exact 15-observable loss
  * per-observable moment-loss breakdown for the low-F+ solutions
  * best real odd-cat fit and best squeezed-single-photon fit
  * CSV and JSON summaries in ./catstate_parity_diagnostics/
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

from quannto.utils.path_utils import models_operators_path
from quannto.utils.strawberryfields.quantum_state_evaluation import (
    output_fidelity,
    projected_qonn_ket,
    reduced_output,
    reference_ket,
)


# -----------------------------------------------------------------------------
# Conventions used by the state-engineering benchmark
# -----------------------------------------------------------------------------
INPUT_ALPHA = -1.0 + 0.0j
CAT_ALPHA = 1.0
PHI_EVEN = 0.0
PHI_ODD = np.pi
TRAINING_CAT_CUTOFF = 20
TRAINED_STATS = 15
INPUT_RANGE = (-1, 1)
OUTPUT_CUTOFF = 60
REFERENCE_CUTOFF = 120
OUTPUT_DIR = Path("results/local/catstate_parity_diagnostics")


@dataclass(frozen=True)
class Config:
    label: str
    N: int
    L: int
    subtractions: Tuple[Tuple[int, ...], ...]  # zero-based mode indices
    reported_loss: float
    reported_F_plus: float
    mode: int = 0


# Table III. One-based K in the paper has been converted to zero-based indices.
CONFIGS: Tuple[Config, ...] = (
    Config("N1_L1_K(1,1)", 1, 1, ((0, 0),), 8.774, 0.982),
    Config("N2_L1_K(2,2)", 2, 1, ((1, 1),), 0.0622, 0.851),
    Config("N2_L1_K(1,2)", 2, 1, ((0, 1),), 2.329, 0.0),
    Config("N3_L1_K(2,3)", 3, 1, ((1, 2),), 0.0613, 0.829),
    Config("N2_L2_K(2);(2)", 2, 2, ((1,), (1,)), 0.152, 0.547),
    Config("N1_L1_K(1)", 1, 1, ((0,),), 2.329, 0.0),
    Config("N1_L1_K(1,1,1)", 1, 1, ((0, 0, 0),), 103.730, 0.839),
    Config("N1_L2_K(1);(1)", 1, 2, ((0,), (0,)), 2.270, 3.22e-4),
)

# The three solutions whose target overlap is negligible in Table III.
SUSPECT_LABELS = (
    "N2_L1_K(1,2)",
    "N1_L1_K(1)",
    "N1_L2_K(1);(1)",
)


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------
def checkpoint_stem(cfg: Config) -> str:
    subs = [list(layer) for layer in cfg.subtractions]
    return (
        f"catstate_phi{PHI_EVEN}_trainsize1_stats{TRAINED_STATS}_cut{TRAINING_CAT_CUTOFF}"
        f"_rng{float(INPUT_RANGE[0])}to{float(INPUT_RANGE[0])}"
        f"_N{cfg.N}_L{cfg.L}_sub{subs}_inNone"
    )


def load_checkpoint(cfg: Config):
    stem = checkpoint_stem(cfg)
    arrays = []
    for name, shape in (
        ("S_l", (cfg.L, 2 * cfg.N, 2 * cfg.N)),
        ("D_l", (cfg.L, 2 * cfg.N)),
    ):
        path = models_operators_path(stem, name, "npy")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing checkpoint array for {cfg.label}: {path}\n"
                f"Expected stem: {stem}"
            )
        arr = np.real_if_close(np.load(path, allow_pickle=False))
        if arr.shape != shape:
            raise ValueError(f"{path}: expected shape {shape}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{path}: contains non-finite values")
        if np.iscomplexobj(arr):
            raise ValueError(f"{path}: expected real saved symplectic/displacement arrays")
        arrays.append(arr)
    return stem, arrays[0], arrays[1]


# -----------------------------------------------------------------------------
# Basic Fock-space utilities
# -----------------------------------------------------------------------------
def annihilation(cutoff: int) -> np.ndarray:
    a = np.zeros((cutoff, cutoff), dtype=np.complex128)
    n = np.arange(1, cutoff, dtype=float)
    a[np.arange(cutoff - 1), np.arange(1, cutoff)] = np.sqrt(n)
    return a


def coherent_ket(alpha: complex, cutoff: int) -> np.ndarray:
    # Stable recurrence avoids factorial overflow.
    out = np.empty(cutoff, dtype=np.complex128)
    out[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, cutoff):
        out[n] = out[n - 1] * alpha / np.sqrt(n)
    norm = np.linalg.norm(out)
    return out / norm


def cat_ket(alpha: complex, parity: str, cutoff: int) -> np.ndarray:
    plus = coherent_ket(alpha, cutoff)
    minus = coherent_ket(-alpha, cutoff)
    if parity == "even":
        ket = plus + minus
    elif parity == "odd":
        ket = plus - minus
    else:
        raise ValueError("parity must be 'even' or 'odd'")
    norm = np.linalg.norm(ket)
    if norm == 0:
        raise ValueError("Cat-state norm vanished")
    return ket / norm


def pure_fidelity(rho: np.ndarray, ket: np.ndarray) -> float:
    d = rho.shape[0]
    v = np.asarray(ket[:d], dtype=np.complex128)
    nv = np.linalg.norm(v)
    if nv == 0:
        return 0.0
    v = v / nv
    return float(np.clip(np.real(np.vdot(v, rho @ v)), 0.0, 1.0))


def psd_sqrt(rho: np.ndarray) -> np.ndarray:
    h = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(h)
    vals = np.clip(vals.real, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    d = min(rho.shape[0], sigma.shape[0])
    r = rho[:d, :d]
    s = sigma[:d, :d]
    sr = psd_sqrt(r)
    middle = sr @ s @ sr
    middle = 0.5 * (middle + middle.conj().T)
    vals = np.linalg.eigvalsh(middle)
    trsqrt = np.sum(np.sqrt(np.clip(vals.real, 0.0, None)))
    return float(np.clip(trsqrt ** 2, 0.0, 1.0))


def parity_stats(rho: np.ndarray) -> Tuple[float, float, float]:
    diag = np.real(np.diag(rho))
    signs = (-1.0) ** np.arange(len(diag))
    parity = float(np.dot(signs, diag))
    p_even = float(diag[::2].sum())
    p_odd = float(diag[1::2].sum())
    return parity, p_even, p_odd


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


# -----------------------------------------------------------------------------
# Exact 15-observable state-engineering loss from Appendix D
# -----------------------------------------------------------------------------
def cat_observables(cutoff: int) -> List[Tuple[str, np.ndarray]]:
    a = annihilation(cutoff)
    ad = a.conj().T
    N = ad @ a
    a2 = a @ a
    a3 = a2 @ a
    a4 = a3 @ a
    a5 = a4 @ a
    a6 = a5 @ a
    ad2 = ad @ ad
    return [
        ("a", a),
        ("a^2", a2),
        ("N", N),
        ("a^3", a3),
        ("a^2 a†", a2 @ ad),
        ("N^2", N @ N),
        ("a^4", a4),
        ("a^3 a†", a3 @ ad),
        ("a^5", a5),
        ("a^4 a†", a4 @ ad),
        ("a^3 (a†)^2", a3 @ ad2),
        ("N^3", N @ N @ N),
        ("a^6", a6),
        ("a^5 a†", a5 @ ad),
        ("a^4 (a†)^2", a4 @ ad2),
    ]


def moments_from_rho(rho: np.ndarray, observables) -> Dict[str, complex]:
    return {name: np.trace(rho @ op) for name, op in observables}


def moments_from_ket(ket: np.ndarray, observables) -> Dict[str, complex]:
    return {name: np.vdot(ket, op @ ket) for name, op in observables}


def moment_loss_and_breakdown(
    moments: Dict[str, complex], target: Dict[str, complex]
) -> Tuple[float, List[dict]]:
    rows = []
    total = 0.0
    for name in target:
        delta = target[name] - moments[name]
        contribution = float(abs(delta) ** 2)
        total += contribution
        rows.append(
            {
                "observable": name,
                "target_re": float(np.real(target[name])),
                "target_im": float(np.imag(target[name])),
                "state_re": float(np.real(moments[name])),
                "state_im": float(np.imag(moments[name])),
                "delta_abs": float(abs(delta)),
                "loss_contribution": contribution,
            }
        )
    return total, rows


# -----------------------------------------------------------------------------
# Fits to physically meaningful odd-parity families
# -----------------------------------------------------------------------------
def best_real_cat_fit(rho: np.ndarray, parity: str, bounds=(0.05, 2.5)):
    d = rho.shape[0]

    def objective(alpha):
        ket = cat_ket(float(alpha), parity, d)
        return -pure_fidelity(rho, ket)

    res = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-10})
    return float(res.x), float(-res.fun)


def squeezed_single_photon_ket(r: float, cutoff: int) -> np.ndarray:
    # Standard real single-mode squeeze operator:
    # S(r)=exp[(r/2)(a^2-a†^2)].  Optimizing over signed r makes the convention
    # robust to the sign convention used elsewhere in the manuscript.
    a = annihilation(cutoff)
    ad = a.conj().T
    generator = 0.5 * float(r) * (a @ a - ad @ ad)
    S = expm(generator)
    one = np.zeros(cutoff, dtype=np.complex128)
    one[1] = 1.0
    ket = S @ one
    return ket / np.linalg.norm(ket)


def best_squeezed_one_fit(rho: np.ndarray, bounds=(-1.8, 1.8)):
    d = rho.shape[0]

    def objective(r):
        return -pure_fidelity(rho, squeezed_single_photon_ket(float(r), d))

    res = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-9})
    return float(res.x), float(-res.fun)


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def reconstruct(cfg: Config):
    stem, S_l, D_l = load_checkpoint(cfg)
    subs = [list(layer) for layer in cfg.subtractions]
    ket, weight = projected_qonn_ket(S_l, D_l, subs, INPUT_ALPHA, OUTPUT_CUTOFF)
    if not np.isfinite(ket).all() or np.linalg.norm(ket) == 0:
        raise ValueError(f"{cfg.label}: invalid projected QONN ket")
    rho = reduced_output(ket, mode=cfg.mode)
    rho = np.asarray(rho, dtype=np.complex128)
    tr = np.trace(rho)
    rho = rho / tr
    return stem, rho, float(np.real(weight)) if np.ndim(weight) == 0 else weight


def fmt(x: float) -> str:
    if abs(x) < 1e-4 or abs(x) >= 1e4:
        return f"{x:.3e}"
    return f"{x:.6f}"


def main(argv=None):
    import argparse
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    OUTPUT_DIR = args.output

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 110)
    print("CAT-STATE PARITY / OPPOSITE-SECTOR DIAGNOSTICS")
    print("=" * 110)
    print(f"Input alpha = {INPUT_ALPHA.real:g}; target |C_+({CAT_ALPHA:g})>; output cutoff = {OUTPUT_CUTOFF}")
    print()

    # Build target states at the same cutoff for moment calculations.
    even = cat_ket(CAT_ALPHA, "even", OUTPUT_CUTOFF)
    odd = cat_ket(CAT_ALPHA, "odd", OUTPUT_CUTOFF)
    rho_even = np.outer(even, even.conj())
    rho_odd = np.outer(odd, odd.conj())
    obs = cat_observables(OUTPUT_CUTOFF)
    m_even = moments_from_ket(even, obs)
    m_odd = moments_from_ket(odd, obs)
    ideal_odd_loss, ideal_odd_breakdown = moment_loss_and_breakdown(m_odd, m_even)
    identical = [r["observable"] for r in ideal_odd_breakdown if r["loss_contribution"] < 1e-12]

    print("Ideal even-vs-odd cat check for the exact 15-observable objective:")
    print(f"  <C+|C->^2               = {pure_fidelity(rho_even, odd):.3e}")
    print(f"  moment loss L(C-, C+)   = {ideal_odd_loss:.12g}")
    print(f"  identical observables   = {len(identical)}/15: {', '.join(identical)}")
    print()

    states: Dict[str, np.ndarray] = {}
    summaries: List[dict] = []
    breakdowns: Dict[str, List[dict]] = {}

    header = (
        f"{'configuration':<22} {'Lrep':>8} {'Lrec':>10} {'F+':>10} {'F-':>10} "
        f"{'<Pi>':>10} {'Podd':>10} {'purity':>10} {'best odd F':>11} {'alpha*':>8} {'best S|1> F':>12} {'r*':>8}"
    )
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        stem, rho, weight = reconstruct(cfg)
        states[cfg.label] = rho

        Fplus = pure_fidelity(rho, even)
        Fminus = pure_fidelity(rho, odd)
        par, pe, po = parity_stats(rho)
        pur = purity(rho)
        moments = moments_from_rho(rho, obs)
        Lrec, br = moment_loss_and_breakdown(moments, m_even)
        breakdowns[cfg.label] = br

        # These fits are especially meaningful for low-F+ states, but computing
        # them for all rows makes the comparison self-contained.
        alpha_odd, F_odd_best = best_real_cat_fit(rho, "odd")
        r_one, F_one_best = best_squeezed_one_fit(rho)

        row = {
            "label": cfg.label,
            "stem": stem,
            "N": cfg.N,
            "L": cfg.L,
            "subtractions_zero_based": [list(x) for x in cfg.subtractions],
            "reported_moment_loss": cfg.reported_loss,
            "recomputed_moment_loss": Lrec,
            "reported_F_plus": cfg.reported_F_plus,
            "F_plus": Fplus,
            "F_minus_alpha1": Fminus,
            "parity": par,
            "P_even": pe,
            "P_odd": po,
            "purity": pur,
            "best_real_odd_cat_alpha": alpha_odd,
            "best_real_odd_cat_fidelity": F_odd_best,
            "best_squeezed_one_r": r_one,
            "best_squeezed_one_fidelity": F_one_best,
        }
        summaries.append(row)

        print(
            f"{cfg.label:<22} {cfg.reported_loss:8.4f} {Lrec:10.6f} {Fplus:10.6g} {Fminus:10.6g} "
            f"{par:10.6f} {po:10.6f} {pur:10.6f} {F_odd_best:11.6f} {alpha_odd:8.4f} "
            f"{F_one_best:12.6f} {r_one:8.4f}"
        )

    print("\n" + "=" * 110)
    print("PAIRWISE UHLMANN FIDELITIES AMONG THE THREE LOW-F+ SOLUTIONS")
    print("=" * 110)
    pairwise = []
    for i, a in enumerate(SUSPECT_LABELS):
        for b in SUSPECT_LABELS[i + 1 :]:
            f = uhlmann_fidelity(states[a], states[b])
            pairwise.append({"state_a": a, "state_b": b, "uhlmann_fidelity": f})
            print(f"  F({a}, {b}) = {f:.12f}")

    print("\n" + "=" * 110)
    print("MOMENT-LOSS BREAKDOWN FOR LOW-F+ SOLUTIONS")
    print("=" * 110)
    for label in SUSPECT_LABELS:
        print(f"\n{label}")
        rows = sorted(breakdowns[label], key=lambda r: r["loss_contribution"], reverse=True)
        for r in rows:
            print(
                f"  {r['observable']:<14} contribution={r['loss_contribution']:.9g}  "
                f"|delta|={r['delta_abs']:.6g}  "
                f"target={r['target_re']:+.6g}{r['target_im']:+.6g}j  "
                f"state={r['state_re']:+.6g}{r['state_im']:+.6g}j"
            )

    # Save machine-readable output.
    json_payload = {
        "conventions": {
            "input_alpha": [INPUT_ALPHA.real, INPUT_ALPHA.imag],
            "cat_alpha": CAT_ALPHA,
            "output_cutoff": OUTPUT_CUTOFF,
            "reference_cutoff": REFERENCE_CUTOFF,
        },
        "ideal_even_vs_odd": {
            "overlap_squared": pure_fidelity(rho_even, odd),
            "moment_loss": ideal_odd_loss,
            "identical_observables": identical,
            "breakdown": ideal_odd_breakdown,
        },
        "models": summaries,
        "pairwise_low_fidelity_states": pairwise,
        "moment_breakdowns": {k: breakdowns[k] for k in SUSPECT_LABELS},
    }
    (OUTPUT_DIR / "diagnostics.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    csv_fields = [
        "label", "N", "L", "reported_moment_loss", "recomputed_moment_loss",
        "reported_F_plus", "F_plus", "F_minus_alpha1", "parity", "P_even", "P_odd",
        "purity", "best_real_odd_cat_alpha", "best_real_odd_cat_fidelity",
        "best_squeezed_one_r", "best_squeezed_one_fidelity",
    ]
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    for label in SUSPECT_LABELS:
        safe = label.replace("(", "").replace(")", "").replace(";", "_").replace(",", "_")
        with (OUTPUT_DIR / f"moment_breakdown_{safe}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(breakdowns[label][0].keys()))
            writer.writeheader()
            writer.writerows(breakdowns[label])

    print("\n" + "=" * 110)
    print("INTERPRETATION FLAGS")
    print("=" * 110)
    by_label = {x["label"]: x for x in summaries}
    for label in SUSPECT_LABELS:
        x = by_label[label]
        if x["P_odd"] > 0.99 and x["best_real_odd_cat_fidelity"] > 0.9:
            verdict = "STRONG odd-cat-like opposite-parity basin"
        elif x["P_odd"] > 0.99:
            verdict = "STRONG opposite-parity basin, but not necessarily an odd cat"
        elif x["P_odd"] > 0.9:
            verdict = "predominantly odd-parity basin"
        else:
            verdict = "NOT an odd-parity explanation; inspect the broader low-loss manifold"
        print(f"  {label:<22}: {verdict}")

    print(f"\nSaved: {OUTPUT_DIR / 'summary.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'diagnostics.json'}")
    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
