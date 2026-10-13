"""Theoretical cat and three saved QONNs in a single-column 2x2 Wigner figure.

Run from the repository root:
    python -m quannto.utils.strawberryfields.multi_catstate_sf_plotter

Edit MODELS and the common settings below. Mode indices in configuration are
zero-based; automatically generated figure labels are one-based.

The QONN states are reconstructed with the same independent Fock evaluator used
for the state-fidelity tables. In particular, Gaussian layers are composed first
and the photon-subtraction polynomial is evaluated afterwards, so intermediate
Gaussian layers are not repeatedly truncated at the plotting cutoff.
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from quannto.utils.path_utils import figures_dir, models_operators_path
from quannto.utils.strawberryfields.general_tools import state_from_single_mode_rho
from quannto.utils.strawberryfields.quantum_state_evaluation import (
    output_fidelity,
    projected_qonn_ket,
    reduced_output,
    reference_ket,
)
from quannto.utils.strawberryfields.wigner_plotters import _compute_wigner_Z


# -----------------------------------------------------------------------------
# Target and checkpoint convention
# -----------------------------------------------------------------------------
# The training data contain a single coherent input alpha_in=-1. The even-cat
# target has |alpha_cat|=1; its sign is physically irrelevant for phi=0 because
# |alpha>+|-alpha> is invariant under alpha -> -alpha.
INPUT_ALPHA = -1.0 + 0.0j
CAT_ALPHA = 1.0 + 0.0j
PHI = 0.0

# This cutoff belongs to the *checkpoint name* because the moments used during
# training were generated at cutoff 20. It is deliberately not the cutoff used
# below to reconstruct the final Wigner functions.
TRAINING_CAT_CUTOFF = 20
TRAINED_STATS = 15
INPUT_RANGE = (-1, 1)  # Checkpoint filenames use the first endpoint twice.

# A high-cutoff target is used for the fidelity printed in each synthesized
# subplot. The theoretical Wigner itself is drawn at the largest model cutoff.
REFERENCE_CUTOFF = 300

# Order: top right, bottom left, bottom right. Each output cutoff is independent.
# These cutoffs concern only the final projected output state; the independent
# evaluator internally adds the polynomial-degree margin required by the
# photon-subtraction sequence.
MODELS = [
    dict(N=1, L=1, subtractions=[[0, 0]], cutoff=60, mode=0),
    dict(N=1, L=1, subtractions=[[0, 0]], cutoff=60, mode=0),
    dict(N=1, L=1, subtractions=[[0, 0]], cutoff=60, mode=0),
]

XLIM = (-3.3, 3.3)
PLIM = (-2.3, 2.3)
GRID_PTS = 301
OUTPUT_NAME = "wigner_multi_catstate.pdf"

# Single-column layout tuned for APS/PRR column width (~3.35 in)
FIGSIZE = (3.35, 2.78)
# Minimize the outer white border of the exported figure. ``bbox_inches="tight"``
# already crops to the artists; this removes Matplotlib's default 0.1-inch pad.
SAVE_PAD_INCHES = 0.01
TITLE_FS = 6.6
TICK_FS = 6.1
AXISLABEL_FS = 7.0
CBAR_FS = 6.1
CONTOUR_LEVELS = 23
XTICKS = (-2, 0, 2)
YTICKS = (-2, 0, 2)


def _math_scalar(value):
    """Compact Matplotlib-math representation for a real/complex scalar."""
    z = complex(value)
    if abs(z.imag) < 1e-12:
        x = z.real
        if abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
        return f"{x:g}"
    real = f"{z.real:g}" if abs(z.real) >= 1e-12 else ""
    imag_abs = abs(z.imag)
    imag_mag = "" if abs(imag_abs - 1.0) < 1e-12 else f"{imag_abs:g}"
    imag = f"{imag_mag}i"
    if not real:
        return ("-" if z.imag < 0 else "") + imag
    return real + ("+" if z.imag > 0 else "-") + imag


def theoretical_cat_title(alpha=CAT_ALPHA, phi=PHI):
    """Compact two-line title for the target cat-state panel."""
    alpha_text = _math_scalar(alpha)
    if abs(float(phi)) < 1e-12:
        return (
            "Theoretical cat state\n"
            + rf"$|C_+(\alpha)\rangle \propto |\alpha\rangle+|-\alpha\rangle,\ \alpha={alpha_text}$"
        )
    phi_text = _math_scalar(phi)
    return (
        "Theoretical cat state\n"
        + rf"$|C(\alpha,\phi)\rangle \propto |\alpha\rangle + e^{{i\phi}}|-\alpha\rangle,\ "
          rf"\alpha={alpha_text},\ \phi={phi_text}$"
    )


def _compact_subtraction_tex(subtractions):
    r"""Return a compact TeX label for the photon-subtraction pattern.

    Examples:
      [[0, 0]]      -> ``\hat a_1^2``
      [[0, 0, 0]]   -> ``\hat a_1^3``
      [[1, 2]]      -> ``\hat a_2\hat a_3``
      [[1], [1]]    -> ``[\hat a_2][\hat a_2]``
    """
    layer_labels = []
    for layer in subtractions:
        counts = Counter(int(m) + 1 for m in layer)
        pieces = []
        for mode in sorted(counts):
            power = counts[mode]
            if power == 1:
                pieces.append(rf"\hat a_{{{mode}}}")
            else:
                pieces.append(rf"\hat a_{{{mode}}}^{{{power}}}")
        layer_label = " ".join(pieces) if pieces else r"\mathbb{I}"
        layer_labels.append(layer_label)
    if len(layer_labels) == 1:
        return layer_labels[0]
    return " ".join(rf"[{label}]" for label in layer_labels)


def load_model(config):
    """Validate the architecture and both saved operator arrays."""
    N, L = config["N"], config["L"]
    cutoff, mode = config["cutoff"], config.get("mode", 0)
    for name, value, minimum in (("N", N, 1), ("L", L, 1), ("cutoff", cutoff, 2)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(mode, (int, np.integer)) or not 0 <= mode < N:
        raise ValueError("Invalid output mode")

    subs = config["subtractions"]
    if len(subs) != L or any(
        not isinstance(m, (int, np.integer)) or not 0 <= m < N
        for layer in subs for m in layer
    ):
        raise ValueError("Provide one subtraction list per layer, with zero-based mode indices")
    subs = [list(layer) for layer in subs]

    stem = config.get("model_name") or (
        f"catstate_phi{PHI}_trainsize1_stats{TRAINED_STATS}_cut{TRAINING_CAT_CUTOFF}"
        f"_rng{float(INPUT_RANGE[0])}to{float(INPUT_RANGE[0])}"
        f"_N{N}_L{L}_sub{subs}_inNone"
    )

    arrays = []
    for name, shape in (("S_l", (L, 2*N, 2*N)), ("D_l", (L, 2*N))):
        path = models_operators_path(stem, name, "npy")
        arr = np.real_if_close(np.load(path, allow_pickle=False))
        if arr.shape != shape or not np.isfinite(arr).all() or np.iscomplexobj(arr):
            raise ValueError(f"{path}: expected finite real array with shape {shape}, got {arr.shape}")
        arrays.append(arr)

    S_l, D_l = arrays
    omega = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    for S in S_l:
        if not np.allclose(S @ omega @ S.T, omega, atol=1e-6, rtol=1e-6):
            raise ValueError(f"{stem}: saved S is not symplectic in xxpp ordering")

    compact_op = _compact_subtraction_tex(subs)
    title = config.get("title") or rf"N={N}, L={L}, K=$[{compact_op}]$"
    return stem, S_l, D_l, title


def _style_single_column_axes(ax, *, is_bottom_row, is_left_col, xlabel, ylabel):
    ax.set_xticks(list(XTICKS))
    ax.set_yticks(list(YTICKS))
    ax.tick_params(labelsize=TICK_FS, length=2.2, width=0.5, pad=1.5)
    if is_bottom_row:
        ax.set_xlabel(xlabel, fontsize=AXISLABEL_FS, labelpad=1.5)
    else:
        ax.set_xlabel("")
    if is_left_col:
        ax.set_ylabel(ylabel, fontsize=AXISLABEL_FS, labelpad=1.0)
    else:
        ax.set_ylabel("")


def plot_wigner_2x2(states, titles, *, xlim=XLIM, plim=PLIM,
                    grid_lim=4.0, grid_pts=GRID_PTS, use_textbook_axes=True,
                    zmin=None, zmax=None, cmap_name="turbo", show_contours=True,
                    contour_levels=CONTOUR_LEVELS, savepath=None, show=True):
    """Plot four single-mode SF states in row order; return ``(figure, axes)``.

    The figure is optimized for single-column use in the manuscript.
    A common phase-space grid, color normalization and contour scale are used.
    The displayed finite window is never renormalized.
    """
    if len(states) != 4 or len(titles) != 4:
        raise ValueError("Exactly four states and four titles are required")
    if any(state.num_modes != 1 for state in states):
        raise ValueError("Provide single-mode states (reduce multimode states first)")

    grids = [
        _compute_wigner_Z(
            state, 0, grid_lim=grid_lim, grid_pts=grid_pts,
            use_textbook_axes=use_textbook_axes, xlim=xlim, plim=plim,
        )
        for state in states
    ]
    values = [grid[0] for grid in grids]
    if any(not np.isfinite(Z).all() for Z in values):
        raise ValueError("Wigner arrays contain nonfinite values")

    vmin = min(float(Z.min()) for Z in values) if zmin is None else float(zmin)
    vmax = max(float(Z.max()) for Z in values) if zmax is None else float(zmax)
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError("Color limits must be finite and strictly increasing")
    if show_contours and (not isinstance(contour_levels, int) or contour_levels < 2):
        raise ValueError("contour_levels must be an integer >= 2")

    norm = Normalize(vmin, vmax)
    _, x, p, xlabel, ylabel = grids[0]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.845, bottom=0.105, top=0.925, wspace=0.10, hspace=0.1)

    for idx, (ax, Z, title) in enumerate(zip(axes.flat, values, titles)):
        row, col = divmod(idx, 2)
        im = ax.imshow(
            Z, origin="lower", extent=(x[0], x[-1], p[0], p[-1]),
            aspect="equal", cmap=cmap_name, norm=norm, interpolation="nearest",
        )
        if show_contours:
            ax.contour(
                x, p, Z, levels=np.linspace(vmin, vmax, contour_levels),
                colors="k", linewidths=0.38, alpha=0.42,
            )
        title_text = title
        ax.set_title(title_text, fontsize=TITLE_FS, linespacing=1.02, pad=1.8)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(p[0], p[-1])
        _style_single_column_axes(
            ax,
            is_bottom_row=(row == 1),
            is_left_col=(col == 0),
            xlabel=xlabel,
            ylabel=ylabel,
        )

    cax = fig.add_axes([0.875, 0.17, 0.027, 0.66])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.5)
    cbar.ax.text(
        0.5, 1.02, f"{vmax:.2f}", transform=cbar.ax.transAxes,
        ha="center", va="bottom", fontsize=CBAR_FS,
    )
    cbar.ax.text(
        0.5, -0.035, f"{vmin:.2f}", transform=cbar.ax.transAxes,
        ha="center", va="top", fontsize=CBAR_FS,
    )
    cbar.ax.text(
        1.75, 0.5, "W(x,p)", transform=cbar.ax.transAxes,
        rotation=90, ha="center", va="center", fontsize=CBAR_FS + 0.2,
    )

    if savepath is not None:
        path = Path(savepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    if show:
        plt.show()
    return fig, axes


def main(*, show=True):
    if len(MODELS) != 3:
        raise ValueError("MODELS must contain exactly three QONN configurations")

    # Fail early for missing or incompatible checkpoints before costly Fock work.
    loaded = [load_model(config) for config in MODELS]

    # Draw the ideal target at the largest output cutoff used by the selected
    # models, so the four Wigner functions are compared without a lower target
    # truncation. For |alpha|=1 this is already far beyond convergence.
    target_plot_cutoff = max(config["cutoff"] for config in MODELS)
    ket_cat = reference_ket(
        "cat", INPUT_ALPHA, target_plot_cutoff,
        cat_alpha=CAT_ALPHA, phi=PHI,
    )
    rho_cat = np.outer(ket_cat, ket_cat.conj())
    states = [state_from_single_mode_rho(rho_cat, target_plot_cutoff)]
    titles = [theoretical_cat_title(CAT_ALPHA, PHI)]

    print(
        f"Target: even cat with alpha={CAT_ALPHA.real:g}, phi={PHI:g}; "
        f"coherent QONN input alpha_in={INPUT_ALPHA.real:g}",
        flush=True,
    )

    for config, (stem, S_l, D_l, title) in zip(MODELS, loaded):
        print(
            f"Reconstructing {stem}; output Fock cutoff={config['cutoff']}",
            flush=True,
        )

        # Same suffix-conjugated photon-subtraction evaluator as the independent
        # fidelity tables. This avoids sequential intermediate-layer truncation.
        ket, weight = projected_qonn_ket(
            S_l, D_l, config["subtractions"], INPUT_ALPHA, config["cutoff"]
        )
        if not np.isfinite(ket).all() or np.linalg.norm(ket) == 0:
            raise ValueError(f"{stem}: invalid or zero output state")

        rho = reduced_output(ket, mode=config.get("mode", 0))
        states.append(state_from_single_mode_rho(rho, config["cutoff"]))

        target = reference_ket(
            "cat", INPUT_ALPHA, max(REFERENCE_CUTOFF, config["cutoff"]),
            cat_alpha=CAT_ALPHA, phi=PHI,
        )
        fidelity = output_fidelity(rho, target)
        titles.append(title + "\n" + rf"$F={fidelity:.3f}$")

        print("Ideal annihilation weight (not a click probability):", weight)
        print("Squared target-state fidelity:", fidelity)

    output = figures_dir() / OUTPUT_NAME
    result = plot_wigner_2x2(
        states, titles, xlim=XLIM, plim=PLIM,
        grid_pts=GRID_PTS, savepath=output, show=show,
    )
    print(f"Saved {output}")
    return result


if __name__ == "__main__":
    main()
