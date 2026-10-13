#!/usr/bin/env python3
"""Moment-hierarchy geometry of the even-cat state-engineering benchmark.

Purpose
-------
The paper trains an even-cat target |C_+(alpha=1)> with an unweighted squared
error over a finite set of ladder-operator moments.  The degree-6 benchmark has
an empirically discovered low-loss basin in the opposite parity sector:
S(r)|1>, r ~ -0.2508, which is nearly an odd cat.

This script asks what happens when the *same moment-basis construction* is
extended to higher total degree d = 2,4,6,8,10,12.

It performs four complementary tests for every d:
  1. Ideal even-vs-odd cat distinguishability in the finite moment space.
  2. Best odd-cat fit C_-(alpha) to the even-cat moment target.
  3. Best squeezed-single-photon branch S(r)|1>.
  4. A two-parameter, physically motivated slice of the N=1,L=1,K=(1)
     QONN state manifold,

         |psi(beta,r)> propto a D(beta) S(r)|0>,

     with real beta and r.  This slice contains the observed odd basin at
     beta=0.  We optimize it from multiple deterministic starts and track the
     continuation of the degree-6 odd branch.  A numerical Hessian is used to
     determine whether the basin remains a local minimum or turns into a
     saddle when higher moments are added.

The moment basis exactly reproduces the 15-operator basis used in the paper at
d=6 (up to ordering): for each total degree n,

  a^n, a^(n-1)a^dag, ..., a^(n-k)(a^dag)^k,

for k < n/2, plus N^(n/2) for even n.  Therefore the d=6 squeezed-one minimum
must reproduce the reported moment loss ~2.329902.

Run
---
From anywhere with NumPy/SciPy/Matplotlib installed:

    python catstate_moment_hierarchy_sweep.py

Outputs are written to results/local/catstate_moment_hierarchy_sweep/ and
include CSV, JSON, PNG/PDF plots, and a text summary. Use --output to select
a different directory.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

# Matplotlib is used only after all numerical work is complete.
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
CAT_ALPHA = 1.0
DEGREES = (2, 4, 6, 8, 10, 12)
CUTOFF = 100
R_BOUND = 1.2
BETA_BOUND = 1.5
OUTPUT_DIR = Path("results/local/catstate_moment_hierarchy_sweep")

# Confirmed d=6 basin from the previous checkpoint diagnostic.
CONFIRMED_R_D6 = -0.2508
CONFIRMED_LOSS_D6 = 2.329902

# Deterministic multistarts in (beta, r).  The first is the measured basin.
STARTS = (
    (0.0, CONFIRMED_R_D6),
    (0.0, -0.10),
    (0.0, -0.50),
    (0.0, 0.25),
    (0.15, -0.25),
    (-0.15, -0.25),
    (0.40, -0.25),
    (-0.40, -0.25),
    (0.70, -0.40),
    (-0.70, -0.40),
    (0.50, 0.30),
    (-0.50, 0.30),
)


# -----------------------------------------------------------------------------
# Fock-space primitives
# -----------------------------------------------------------------------------
def annihilation(cutoff: int) -> np.ndarray:
    a = np.zeros((cutoff, cutoff), dtype=np.complex128)
    n = np.arange(1, cutoff, dtype=float)
    a[np.arange(cutoff - 1), np.arange(1, cutoff)] = np.sqrt(n)
    return a


A = annihilation(CUTOFF)
AD = A.conj().T
N_OP = AD @ A
I_OP = np.eye(CUTOFF, dtype=np.complex128)
# Real displacement generator: D(beta)=exp[beta(a^dag-a)].
DISP_GENERATOR = csc_matrix(AD - A)


def coherent_ket(alpha: complex, cutoff: int = CUTOFF) -> np.ndarray:
    out = np.empty(cutoff, dtype=np.complex128)
    out[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, cutoff):
        out[n] = out[n - 1] * alpha / np.sqrt(n)
    return out / np.linalg.norm(out)


def cat_ket(alpha: float, parity: str, cutoff: int = CUTOFF) -> np.ndarray:
    p = coherent_ket(float(alpha), cutoff)
    m = coherent_ket(-float(alpha), cutoff)
    if parity == "even":
        v = p + m
    elif parity == "odd":
        v = p - m
    else:
        raise ValueError("parity must be 'even' or 'odd'")
    return v / np.linalg.norm(v)


def squeezed_vacuum_ket(r: float, cutoff: int = CUTOFF) -> np.ndarray:
    """S(r)|0> for S(r)=exp[(r/2)(a^2-a^dag^2)], via stable recurrence."""
    out = np.zeros(cutoff, dtype=np.complex128)
    t = np.tanh(float(r))
    out[0] = 1.0 / np.sqrt(np.cosh(float(r)))
    # coefficient of |2(n+1)> from |2n>
    for n in range((cutoff - 2) // 2 + 1):
        i = 2 * n
        j = i + 2
        if j >= cutoff:
            break
        out[j] = out[i] * (-t) * np.sqrt((2*n + 2) * (2*n + 1)) / (2 * (n + 1))
    return out / np.linalg.norm(out)


def squeezed_one_ket(r: float, cutoff: int = CUTOFF) -> np.ndarray:
    """S(r)|1> for the same convention, via stable recurrence."""
    out = np.zeros(cutoff, dtype=np.complex128)
    t = np.tanh(float(r))
    out[1] = 1.0 / (np.cosh(float(r)) ** 1.5)
    # coefficient of |2(n+1)+1> from |2n+1>
    for n in range((cutoff - 3) // 2 + 1):
        i = 2 * n + 1
        j = i + 2
        if j >= cutoff:
            break
        out[j] = out[i] * (-t) * np.sqrt((2*n + 3) * (2*n + 2)) / (2 * (n + 1))
    return out / np.linalg.norm(out)


def displace_real(ket: np.ndarray, beta: float) -> np.ndarray:
    if abs(beta) < 1e-15:
        return ket.copy()
    out = expm_multiply(float(beta) * DISP_GENERATOR, ket)
    return np.asarray(out, dtype=np.complex128)


def single_subtracted_gaussian_ket(beta: float, r: float) -> np.ndarray:
    """Normalized a D(beta) S(r)|0> for real beta,r.

    Identity used:
      a D(beta) S(r)|0> = D(beta) S(r) [beta|0> - sinh(r)|1>]
    for the squeezing convention used above.
    """
    sv = squeezed_vacuum_ket(r)
    s1 = squeezed_one_ket(r)
    pre = float(beta) * sv - np.sinh(float(r)) * s1
    norm = np.linalg.norm(pre)
    if norm < 1e-14:
        # At beta=r=0 the heralded state has zero norm; return a huge-loss marker
        # through NaNs handled by the objective.
        return np.full(CUTOFF, np.nan + 0j)
    pre /= norm
    out = displace_real(pre, beta)
    out /= np.linalg.norm(out)
    return out


# -----------------------------------------------------------------------------
# Paper-like independent moment hierarchy
# -----------------------------------------------------------------------------
def matrix_power(op: np.ndarray, power: int) -> np.ndarray:
    if power == 0:
        return I_OP
    return np.linalg.matrix_power(op, power)


def paper_basis(max_degree: int) -> List[Tuple[str, int, np.ndarray]]:
    """Extend exactly the basis pattern used by Eq. (D12) to arbitrary d."""
    ops: List[Tuple[str, int, np.ndarray]] = []
    for n in range(1, max_degree + 1):
        # Non-central representatives: a^(n-k) (a^dag)^k, k<n/2.
        for k in range((n - 1) // 2 + 1):
            p_a = n - k
            p_ad = k
            op = matrix_power(A, p_a) @ matrix_power(AD, p_ad)
            if p_ad == 0:
                name = f"a^{p_a}" if p_a != 1 else "a"
            elif p_ad == 1:
                name = f"a^{p_a} a†"
            else:
                name = f"a^{p_a} (a†)^{p_ad}"
            ops.append((name, n, op))
        # Central Hermitian representative for even degree.
        if n % 2 == 0:
            m = n // 2
            name = "N" if m == 1 else f"N^{m}"
            ops.append((name, n, matrix_power(N_OP, m)))
    return ops


def expected_d6_names() -> set:
    return {
        "a", "a^2", "N", "a^3", "a^2 a†", "N^2", "a^4", "a^3 a†",
        "a^5", "a^4 a†", "a^3 (a†)^2", "N^3", "a^6", "a^5 a†",
        "a^4 (a†)^2",
    }


def moments(ket: np.ndarray, basis) -> Dict[str, complex]:
    return {name: np.vdot(ket, op @ ket) for name, _, op in basis}


def loss_breakdown(ket: np.ndarray, target_m: Dict[str, complex], basis):
    rows = []
    total = 0.0
    normalized = 0.0
    for name, degree, op in basis:
        value = np.vdot(ket, op @ ket)
        target = target_m[name]
        delta = value - target
        c = float(abs(delta) ** 2)
        cn = float(c / (1.0 + abs(target) ** 2))
        total += c
        normalized += cn
        rows.append({
            "observable": name,
            "degree": degree,
            "target_re": float(np.real(target)),
            "target_im": float(np.imag(target)),
            "state_re": float(np.real(value)),
            "state_im": float(np.imag(value)),
            "delta_abs": float(abs(delta)),
            "loss_contribution": c,
            "scale_normalized_contribution": cn,
        })
    return total, normalized / len(basis), rows


def pure_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(abs(np.vdot(a, b)) ** 2, 0.0, 1.0))


def parity_expectation(ket: np.ndarray) -> float:
    return float(np.sum(((-1.0) ** np.arange(len(ket))) * np.abs(ket) ** 2).real)


# -----------------------------------------------------------------------------
# Optimizers and local-curvature diagnostics
# -----------------------------------------------------------------------------
def optimize_odd_cat(target_m, basis):
    def f(alpha):
        ket = cat_ket(float(alpha), "odd")
        return loss_breakdown(ket, target_m, basis)[0]
    res = minimize_scalar(f, bounds=(0.05, 2.5), method="bounded", options={"xatol": 1e-11})
    ket = cat_ket(float(res.x), "odd")
    raw, normed, br = loss_breakdown(ket, target_m, basis)
    return float(res.x), ket, raw, normed, br


def optimize_squeezed_one(target_m, basis):
    def f(r):
        ket = squeezed_one_ket(float(r))
        return loss_breakdown(ket, target_m, basis)[0]
    res = minimize_scalar(f, bounds=(-R_BOUND, R_BOUND), method="bounded", options={"xatol": 1e-11})
    ket = squeezed_one_ket(float(res.x))
    raw, normed, br = loss_breakdown(ket, target_m, basis)
    return float(res.x), ket, raw, normed, br


def optimize_subtracted_slice(target_m, basis, starts=STARTS):
    def objective(x):
        beta, r = float(x[0]), float(x[1])
        ket = single_subtracted_gaussian_ket(beta, r)
        if not np.isfinite(ket).all():
            return 1e100
        value = loss_breakdown(ket, target_m, basis)[0]
        return value if np.isfinite(value) else 1e100

    solutions = []
    for start in starts:
        res = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[(-BETA_BOUND, BETA_BOUND), (-R_BOUND, R_BOUND)],
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 3000, "maxfun": 10000},
        )
        beta, r = map(float, res.x)
        ket = single_subtracted_gaussian_ket(beta, r)
        raw, normed, br = loss_breakdown(ket, target_m, basis)
        solutions.append({
            "start_beta": float(start[0]),
            "start_r": float(start[1]),
            "beta": beta,
            "r": r,
            "loss": raw,
            "normalized_loss": normed,
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(getattr(res, "nit", -1)),
            "ket": ket,
            "breakdown": br,
        })
    solutions.sort(key=lambda x: x["loss"])
    return solutions


def numerical_hessian_2d(fun, x: Sequence[float], h=(2e-4, 2e-4)) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    h0, h1 = map(float, h)
    e0 = np.array([h0, 0.0])
    e1 = np.array([0.0, h1])
    f0 = fun(x)
    H = np.zeros((2, 2), dtype=float)
    H[0, 0] = (fun(x + e0) - 2*f0 + fun(x - e0)) / h0**2
    H[1, 1] = (fun(x + e1) - 2*f0 + fun(x - e1)) / h1**2
    H[0, 1] = H[1, 0] = (
        fun(x + e0 + e1) - fun(x + e0 - e1)
        - fun(x - e0 + e1) + fun(x - e0 - e1)
    ) / (4*h0*h1)
    return H


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None):
    import argparse
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    OUTPUT_DIR = args.output

    OUTPUT_DIR.mkdir(exist_ok=True)
    target = cat_ket(CAT_ALPHA, "even")
    odd_alpha1 = cat_ket(CAT_ALPHA, "odd")

    # Exact basis self-check.
    names6 = {name for name, _, _ in paper_basis(6)}
    if names6 != expected_d6_names() or len(names6) != 15:
        raise RuntimeError(
            "The generated d=6 basis does not reproduce the 15-observable paper basis.\n"
            f"Generated: {sorted(names6)}"
        )

    print("=" * 126)
    print("CAT-STATE MOMENT-HIERARCHY / SYMMETRY-SECTOR SWEEP")
    print("=" * 126)
    print(f"Target = |C_+({CAT_ALPHA:g})>, cutoff={CUTOFF}, degrees={DEGREES}")
    print("Basis continuation exactly reproduces the paper's 15 observables at d=6 (up to ordering).")
    print()

    all_results = []
    all_multistarts = {}
    all_breakdowns = {}

    # Track a continuation start from the measured d=6 basin in both directions.
    continuation_seed = {d: (0.0, CONFIRMED_R_D6) for d in DEGREES}

    header = (
        f"{'d':>2} {'#obs':>5} {'blind':>5} "
        f"{'L odd(1)':>12} {'best odd L':>12} {'a*':>7} "
        f"{'best S1 L':>12} {'r*':>8} "
        f"{'slice L':>12} {'beta*':>9} {'r*':>8} "
        f"{'F+':>9} {'F-':>9} {'<Pi>':>9} {'minEigH':>11}"
    )
    print(header)
    print("-" * len(header))

    for d in DEGREES:
        basis = paper_basis(d)
        target_m = moments(target, basis)

        # How many coordinates are exactly blind for ideal C+(1) vs C-(1)?
        L_odd1, L_odd1_norm, br_odd1 = loss_breakdown(odd_alpha1, target_m, basis)
        blind = sum(row["loss_contribution"] < 1e-12 for row in br_odd1)

        aopt, k_odd, L_best_odd, Ln_best_odd, br_best_odd = optimize_odd_cat(target_m, basis)
        ropt, k_s1, L_s1, Ln_s1, br_s1 = optimize_squeezed_one(target_m, basis)

        # Ensure the measured basin is always explicitly included as a start.
        starts = list(STARTS)
        starts.insert(0, (0.0, ropt))
        sols = optimize_subtracted_slice(target_m, basis, starts=tuple(starts))
        best = sols[0]
        ket = best["ket"]

        # Local Hessian of the raw training objective in the physical 2D slice.
        def slice_objective(x):
            k = single_subtracted_gaussian_ket(float(x[0]), float(x[1]))
            if not np.isfinite(k).all():
                return 1e100
            return loss_breakdown(k, target_m, basis)[0]

        H = numerical_hessian_2d(slice_objective, [best["beta"], best["r"]])
        eigH = np.linalg.eigvalsh(0.5 * (H + H.T))

        Fp = pure_fidelity(target, ket)
        Fm = pure_fidelity(odd_alpha1, ket)
        par = parity_expectation(ket)
        best_odd_fit_alpha, best_odd_fit_F = (None, None)
        # State-level best odd-cat fit of the optimized slice solution.
        def minus_fid(alpha):
            return -pure_fidelity(ket, cat_ket(float(alpha), "odd"))
        fitres = minimize_scalar(minus_fid, bounds=(0.05, 2.5), method="bounded")
        best_odd_fit_alpha = float(fitres.x)
        best_odd_fit_F = float(-fitres.fun)

        row = {
            "degree": d,
            "n_observables": len(basis),
            "n_blind_even_vs_odd_alpha1": blind,
            "blind_fraction": blind / len(basis),
            "ideal_odd_alpha1_loss": L_odd1,
            "ideal_odd_alpha1_normalized_loss": L_odd1_norm,
            "best_odd_cat_alpha": aopt,
            "best_odd_cat_loss": L_best_odd,
            "best_odd_cat_normalized_loss": Ln_best_odd,
            "best_squeezed_one_r": ropt,
            "best_squeezed_one_loss": L_s1,
            "best_squeezed_one_normalized_loss": Ln_s1,
            "slice_beta": best["beta"],
            "slice_r": best["r"],
            "slice_loss": best["loss"],
            "slice_normalized_loss": best["normalized_loss"],
            "slice_F_plus": Fp,
            "slice_F_minus_alpha1": Fm,
            "slice_parity": par,
            "slice_best_odd_cat_alpha": best_odd_fit_alpha,
            "slice_best_odd_cat_fidelity": best_odd_fit_F,
            "hessian_eig_min": float(eigH[0]),
            "hessian_eig_max": float(eigH[1]),
        }
        all_results.append(row)

        # Strip arrays before JSON serialization.
        all_multistarts[str(d)] = [
            {k: v for k, v in s.items() if k not in ("ket", "breakdown")}
            for s in sols
        ]
        all_breakdowns[str(d)] = {
            "ideal_odd_alpha1": br_odd1,
            "best_odd_cat": br_best_odd,
            "best_squeezed_one": br_s1,
            "best_subtracted_slice": best["breakdown"],
        }

        print(
            f"{d:2d} {len(basis):5d} {blind:5d} "
            f"{L_odd1:12.5g} {L_best_odd:12.5g} {aopt:7.4f} "
            f"{L_s1:12.5g} {ropt:8.4f} "
            f"{best['loss']:12.5g} {best['beta']:9.5f} {best['r']:8.4f} "
            f"{Fp:9.5f} {Fm:9.5f} {par:9.5f} {eigH[0]:11.4g}"
        )

    # d=6 self-consistency against the checkpoint-derived basin.
    d6 = next(x for x in all_results if x["degree"] == 6)
    print("\nD=6 CONSISTENCY CHECK")
    print(f"  best S(r)|1> loss from hierarchy scan = {d6['best_squeezed_one_loss']:.9f}")
    print(f"  checkpoint diagnostic loss            = {CONFIRMED_LOSS_D6:.9f}")
    print(f"  absolute difference                   = {abs(d6['best_squeezed_one_loss']-CONFIRMED_LOSS_D6):.3e}")
    print(f"  fitted r                              = {d6['best_squeezed_one_r']:.9f}")
    if abs(d6["best_squeezed_one_loss"] - CONFIRMED_LOSS_D6) > 5e-4:
        print("  WARNING: d=6 does not reproduce the previous diagnostic closely; inspect conventions/cutoff.")
    else:
        print("  PASS: d=6 reproduces the previously identified opposite-parity basin.")

    # Identify whether/where the odd basin ceases to be a local minimum in the 2D physical slice.
    positive = [x["hessian_eig_min"] > 0 for x in all_results]
    transitions = []
    for a, b in zip(all_results[:-1], all_results[1:]):
        if (a["hessian_eig_min"] > 0) != (b["hessian_eig_min"] > 0):
            transitions.append((a["degree"], b["degree"]))

    print("\nLOCAL-BASIN STATUS IN THE REAL SINGLE-SUBTRACTION GAUSSIAN SLICE")
    for x in all_results:
        status = "LOCAL MINIMUM" if x["hessian_eig_min"] > 0 else "SADDLE/FLAT"
        print(
            f"  d={x['degree']:2d}: {status:13s}  min Hessian eig={x['hessian_eig_min']:+.6g}, "
            f"beta={x['slice_beta']:+.5f}, parity={x['slice_parity']:+.6f}, F+={x['slice_F_plus']:.6g}"
        )
    if transitions:
        print("  Sign-change interval(s) in local curvature:", transitions)
    else:
        print("  No sign change of the minimum Hessian eigenvalue over the scanned degrees.")

    # Which newly added degree contributes most at the preceding squeezed-one optimum?
    print("\nINCREMENTAL DISCRIMINATING INFORMATION")
    prev_d = 0
    for x in all_results:
        d = x["degree"]
        basis = paper_basis(d)
        target_m = moments(target, basis)
        ket = squeezed_one_ket(x["best_squeezed_one_r"])
        _, _, br = loss_breakdown(ket, target_m, basis)
        new_rows = [r for r in br if prev_d < r["degree"] <= d]
        new_rows.sort(key=lambda r: r["loss_contribution"], reverse=True)
        top = new_rows[:5]
        print(f"  added degrees {prev_d+1}..{d}: total contribution={sum(r['loss_contribution'] for r in new_rows):.6g}")
        for r in top:
            print(f"      {r['observable']:<18} degree={r['degree']:2d} contribution={r['loss_contribution']:.6g}")
        prev_d = d

    # Save CSV.
    csv_path = OUTPUT_DIR / "hierarchy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    payload = {
        "settings": {
            "cat_alpha": CAT_ALPHA,
            "degrees": list(DEGREES),
            "cutoff": CUTOFF,
            "r_bound": R_BOUND,
            "beta_bound": BETA_BOUND,
            "confirmed_d6_r": CONFIRMED_R_D6,
            "confirmed_d6_loss": CONFIRMED_LOSS_D6,
        },
        "summary": all_results,
        "multistarts": all_multistarts,
        "breakdowns": all_breakdowns,
        "curvature_sign_change_intervals": transitions,
    }
    (OUTPUT_DIR / "hierarchy_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable summary.
    txt_lines = [
        "CAT-STATE MOMENT-HIERARCHY SWEEP",
        f"target |C_+({CAT_ALPHA})>, cutoff={CUTOFF}",
        "",
    ]
    for x in all_results:
        txt_lines.append(
            f"d={x['degree']:2d}, n={x['n_observables']:2d}, blind={x['n_blind_even_vs_odd_alpha1']:2d}, "
            f"L_squeezed1={x['best_squeezed_one_loss']:.9g}, r={x['best_squeezed_one_r']:+.6f}, "
            f"L_slice={x['slice_loss']:.9g}, beta={x['slice_beta']:+.6f}, r={x['slice_r']:+.6f}, "
            f"F+={x['slice_F_plus']:.9g}, F-={x['slice_F_minus_alpha1']:.9g}, "
            f"Pi={x['slice_parity']:+.9g}, minEigH={x['hessian_eig_min']:+.9g}"
        )
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ plots
    ds = np.array([x["degree"] for x in all_results])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.semilogy(ds, [x["ideal_odd_alpha1_loss"] for x in all_results], "o-", label=r"$C_-(1)$")
    ax.semilogy(ds, [x["best_odd_cat_loss"] for x in all_results], "s-", label=r"best $C_-(\alpha)$")
    ax.semilogy(ds, [x["best_squeezed_one_loss"] for x in all_results], "^-", label=r"best $S(r)|1\rangle$")
    ax.semilogy(ds, [x["slice_loss"] for x in all_results], "d-", label=r"best $aD(\beta)S(r)|0\rangle$")
    ax.set_xlabel("maximum moment degree $d$")
    ax.set_ylabel("raw moment-matching loss")
    ax.set_xticks(ds)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_vs_degree.pdf")
    fig.savefig(OUTPUT_DIR / "loss_vs_degree.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(ds, [x["slice_F_plus"] for x in all_results], "o-", label=r"$F_+$")
    ax.plot(ds, [x["slice_F_minus_alpha1"] for x in all_results], "s-", label=r"$F_-$ at $\alpha=1$")
    ax.plot(ds, [(1-x["slice_parity"])/2 for x in all_results], "^-", label=r"$P_{\rm odd}$")
    ax.set_xlabel("maximum moment degree $d$")
    ax.set_ylabel("state-level diagnostic")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(ds)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sector_vs_degree.pdf")
    fig.savefig(OUTPUT_DIR / "sector_vs_degree.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(ds, [x["slice_beta"] for x in all_results], "o-", label=r"$\beta_*$")
    ax.plot(ds, [x["slice_r"] for x in all_results], "s-", label=r"$r_*$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("maximum moment degree $d$")
    ax.set_ylabel("optimized parameter")
    ax.set_xticks(ds)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "parameters_vs_degree.pdf")
    fig.savefig(OUTPUT_DIR / "parameters_vs_degree.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    vals = np.array([x["hessian_eig_min"] for x in all_results])
    ax.plot(ds, vals, "o-")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("maximum moment degree $d$")
    ax.set_ylabel("minimum Hessian eigenvalue")
    ax.set_xticks(ds)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "local_curvature_vs_degree.pdf")
    fig.savefig(OUTPUT_DIR / "local_curvature_vs_degree.png", dpi=220)
    plt.close(fig)

    print("\nSaved numerical and graphical output to:", OUTPUT_DIR.resolve())
    print("Diagnostic completed.")


if __name__ == "__main__":
    main()
