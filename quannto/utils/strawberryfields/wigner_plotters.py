import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize

from quannto.utils.path_utils import figures_dir
from quannto.utils.strawberryfields.general_tools import fock_probs_from_ket, reduced_rho_from_ket, state_from_ket, state_from_single_mode_rho

def plot_state_diagnostics(
    ket=None,
    N=None,
    cutoff=20,
    mode=0,
    grid_lim=5.0,
    grid_pts=200,
    use_textbook_axes=True,
    title_prefix=None,
):
    """ state = state_from_ket(ket, cutoff=cutoff, N=N)

    if state.num_modes <= mode:
        raise ValueError(f"State has num_modes={state.num_modes}, can't plot mode={mode}.")

    # ----- 1) Fock probabilities of selected mode -----
    # This works for multi-mode too (uses reduced density matrix)
    rho_m = state.reduced_dm(mode)               # (cutoff, cutoff)
    probs = np.real(np.diag(rho_m))              # P(n) """
    probs = fock_probs_from_ket(ket, N=N, cutoff=cutoff, mode=mode)
    n = np.arange(len(probs))

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.bar(n, probs)
    ax.set_title(f"{title_prefix} — Fock probabilities (cutoff={cutoff})")
    ax.set_xlabel("n")
    ax.set_ylabel("P(n)")
    plt.tight_layout()

    # ----- Grid for Wigner + marginals -----
    if use_textbook_axes:
        x_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        p_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        xvec = np.sqrt(2) * x_phys   # convert to SF units
        pvec = np.sqrt(2) * p_phys
        x_plot, p_plot = x_phys, p_phys
        xlabel, ylabel = "x", "p"
    else:
        xvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        pvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        x_plot, p_plot = xvec, pvec
        xlabel, ylabel = "x (SF units)", "p (SF units)"

    # ----- 2) Wigner -----
    reduced_state = state_from_single_mode_rho(reduced_rho_from_ket(ket, cutoff=cutoff, N=N, mode=mode), cutoff=cutoff)
    W = reduced_state.wigner(mode=mode, xvec=xvec, pvec=pvec)  # shape (len(xvec), len(pvec))
    #W = state.wigner(mode=mode, xvec=xvec, pvec=pvec)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(
        W.T,
        extent=[x_plot[0], x_plot[-1], p_plot[0], p_plot[-1]],
        origin="lower",
        aspect="auto",
    )
    ax.set_title(f"{title_prefix} — Wigner W(x,p) (cutoff={cutoff})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, shrink=0.85, label="W(x,p)")
    plt.tight_layout()

    # ----- 3) Quadrature marginals -----
    # These are defined on the same (xvec,pvec) grid in SF units
    """ Px = state.x_quad_values(mode=mode, xvec=xvec, pvec=pvec)
    Pp = state.p_quad_values(mode=mode, xvec=xvec, pvec=pvec) """
    Px = reduced_state.x_quad_values(mode=mode, xvec=xvec, pvec=pvec)
    Pp = reduced_state.p_quad_values(mode=mode, xvec=xvec, pvec=pvec)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x_plot, Px)
    ax[0].set_title("x-quadrature marginal")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("P(x)")

    ax[1].plot(p_plot, Pp)
    ax[1].set_title("p-quadrature marginal")
    ax[1].set_xlabel(ylabel)
    ax[1].set_ylabel("P(p)")
    plt.tight_layout()

    plt.show()
    
def _compute_wigner_Z(state, mode, *, grid_lim, grid_pts, use_textbook_axes):
    # Build xvec, pvec for SF wigner, and x/p axes to display
    if use_textbook_axes:
        x_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        p_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        xvec = np.sqrt(2) * x_phys
        pvec = np.sqrt(2) * p_phys
        x_plot, p_plot = x_phys, p_phys
        xlabel, ylabel = "x", "p"
    else:
        xvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        pvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        x_plot, p_plot = xvec, pvec
        xlabel, ylabel = "x (SF units)", "p (SF units)"

    W = state.wigner(mode=mode, xvec=xvec, pvec=pvec)  # (len(xvec), len(pvec))
    Z = W.T  # for imshow: x horizontal, p vertical
    return Z, x_plot, p_plot, xlabel, ylabel


def plot_wigner_2d_compare(*,
                          # left (theory)
                          state_L=None, ket_L=None, N_L=None, cutoff_L=None, mode_L=0,
                          # right (synthesized)
                          state_R=None, ket_R=None, N_R=None, cutoff_R=None, mode_R=0,
                          # grid/plot
                          grid_lim=4.0, grid_pts=201,
                          use_textbook_axes=True,
                          cmap_name="turbo",
                          show_contours=True,
                          contour_levels=25,
                          # normalization
                          zmin=None, zmax=None,
                          # titles
                          title_L="Theoretical", title_R="Synthesized", suptitle=None,
                          # saving
                          savepath=None):
    """
    Side-by-side 2D Wigner plots with a shared colorbar.
    You can provide each side as either:
      - state_L / state_R (SF state), or
      - (ket_*, cutoff_*, optionally N_*) -> uses your state_from_ket(...)
    """

    # --- build states ---
    if state_L is None:
        if ket_L is None or cutoff_L is None:
            raise ValueError("Left: provide either state_L=... or (ket_L=..., cutoff_L=..., optionally N_L=...).")
        state_L = state_from_ket(ket_L, cutoff=cutoff_L, N=N_L)

    if state_R is None:
        if ket_R is None or cutoff_R is None:
            raise ValueError("Right: provide either state_R=... or (ket_R=..., cutoff_R=..., optionally N_R=...).")
        state_R = state_from_ket(ket_R, cutoff=cutoff_R, N=N_R)

    if state_L.num_modes <= mode_L:
        raise ValueError(f"Left state has num_modes={state_L.num_modes}, can't plot mode_L={mode_L}.")
    if state_R.num_modes <= mode_R:
        raise ValueError(f"Right state has num_modes={state_R.num_modes}, can't plot mode_R={mode_R}.")

    # --- compute Wigners on same grid ---
    ZL, x_plot, p_plot, xlabel, ylabel = _compute_wigner_Z(
        state_L, mode_L, grid_lim=grid_lim, grid_pts=grid_pts, use_textbook_axes=use_textbook_axes
    )
    ZR, x_plot2, p_plot2, _, _ = _compute_wigner_Z(
        state_R, mode_R, grid_lim=grid_lim, grid_pts=grid_pts, use_textbook_axes=use_textbook_axes
    )

    # sanity: grids should match
    if not (np.allclose(x_plot, x_plot2) and np.allclose(p_plot, p_plot2)):
        raise RuntimeError("Left/right grids mismatch (this should not happen).")

    # --- shared normalization (either user-provided or global min/max) ---
    if zmin is None:
        zmin = float(min(np.min(ZL), np.min(ZR)))
    if zmax is None:
        zmax = float(max(np.max(ZL), np.max(ZR)))

    cmap = matplotlib.colormaps[cmap_name]
    norm = Normalize(vmin=zmin, vmax=zmax)

    # --- draw ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharex=True, sharey=True,
                         constrained_layout=True)

    ims = []
    for ax, Z, t in [(axes[0], ZL, title_L), (axes[1], ZR, title_R)]:
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[x_plot[0], x_plot[-1], p_plot[0], p_plot[-1]],
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ims.append(im)

        if show_contours:
            X, P = np.meshgrid(x_plot, p_plot, indexing="xy")
            ax.contour(
                X, P, Z,
                levels=np.linspace(zmin, zmax, contour_levels),
                linewidths=0.6,
                colors="k",
                alpha=0.35,
            )

        ax.set_title(t, fontsize=22)

    # labels/ticks (match your style)
    font_size = 22
    for ax in axes:
        ax.set_yticks([-3, 0, 3])
        ax.set_xticks([-2, 0, 2])
        ax.tick_params(labelsize=font_size)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-3.5, 3.5)

    axes[0].set_xlabel(xlabel, fontsize=font_size)
    axes[0].set_ylabel(ylabel, fontsize=font_size)
    axes[1].set_xlabel(xlabel, fontsize=font_size)
    axes[1].set_ylabel("")  # cleaner; shared y anyway

    # single shared colorbar
    #cbar = fig.colorbar(ims[0], ax=axes, shrink=0.9, pad=0.02)
    cbar = fig.colorbar(ims[0], ax=axes, location="right", shrink=0.9, pad=0.02)
    cbar.set_label("W(x,p)", fontsize=font_size)
    cbar.set_ticks([])

    # optional min/max labels like you did
    cbar.ax.tick_params(labelsize=font_size-2)
    cbar.ax.text(0.5, -0.01, f'−{np.abs(zmin):.2f}', transform=cbar.ax.transAxes,
                 va='top', ha='left', fontsize=font_size)
    cbar.ax.text(0.5, 1.0, f'{zmax:.3f}', transform=cbar.ax.transAxes,
                 va='bottom', ha='left', fontsize=font_size)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=24, y=1.02)

    #plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def plot_wigner_2d(*,
                   state=None,
                   ket=None, N=None, cutoff=None,
                   mode=0,
                   grid_lim=4.0, grid_pts=201,
                   use_textbook_axes=True,
                   cmap_name="turbo",
                   show_contours=True,
                   contour_levels=25,
                   title=None):
    """
    Unified 2D Wigner plotter.
    Use for:
      - theoretical single-mode cat ket: ket=..., N=1, cutoff=...
      - QONN multi-mode final ket: ket=..., N=..., cutoff=..., mode=0
      - already computed SF state: state=...
    """
    if state is None:
        if ket is None or cutoff is None:
            raise ValueError("Provide either state=... or (ket=..., cutoff=..., and optionally N=...).")
        state = state_from_ket(ket, cutoff=cutoff, N=N)

    if state.num_modes <= mode:
        raise ValueError(f"State has num_modes={state.num_modes}, can't plot mode={mode}.")

    # Build xvec, pvec for SF wigner, and x/p axes to display
    if use_textbook_axes:
        x_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        p_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        xvec = np.sqrt(2) * x_phys
        pvec = np.sqrt(2) * p_phys
        x_plot, p_plot = x_phys, p_phys
        xlabel, ylabel = "x", "p"
    else:
        xvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        pvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        x_plot, p_plot = xvec, pvec
        xlabel, ylabel = "x (SF units)", "p (SF units)"

    W = state.wigner(mode=mode, xvec=xvec, pvec=pvec)  # shape (len(xvec), len(pvec))

    # For imshow, we transpose so x corresponds to horizontal axis and p to vertical
    Z = W.T
    zmin, zmax = float(np.min(Z)), float(np.max(Z))
    zmin = -0.05
    zmax = 0.175

    cmap = matplotlib.colormaps[cmap_name]
    norm = Normalize(vmin=zmin, vmax=zmax)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[x_plot[0], x_plot[-1], p_plot[0], p_plot[-1]],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    if show_contours:
        X, P = np.meshgrid(x_plot, p_plot, indexing="xy")
        ax.contour(
            X, P, Z,
            levels=np.linspace(zmin, zmax, contour_levels),
            linewidths=0.6,
            colors="k",
            alpha=0.35,
        )

    font_size = 22
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("W(x,p)", fontsize=font_size)
    #cbar.set_ticks([round(zmin,2), round(0.0,2), round(zmax,2)])
    cbar.set_ticks([])
    cbar.ax.tick_params(labelsize=font_size-2)
    cbar.ax.text(0.5, -0.01, f'−{np.abs(zmin):.2f}', transform=cbar.ax.transAxes, 
        va='top', ha='left', fontsize=font_size)
    cbar.ax.text(0.5, 1.0, f'{zmax:.3f}', transform=cbar.ax.transAxes, 
        va='bottom', ha='left', fontsize=font_size)

    ax.set_yticks([-3, 0, 3])
    ax.set_xticks([-3, 0, 3])
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    
    if title is None:
        title = f"2D Wigner (mode={mode})"
    #ax.set_title(title)
    
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlim(-3, 3)
    plt.ylim(-3.5, 3.5)

    plt.tight_layout()
    plt.savefig(figures_dir() / f"wigner_2d_{title}.pdf")
    plt.show()

def _add_3plane_projections(ax, X, P, Z, *, cmap, norm,
                            levels=30,
                            alpha_xy=0.95, alpha_xw=0.7, alpha_pw=0.7):
    x_min, x_max = float(np.min(X)), float(np.max(X))
    p_min, p_max = float(np.min(P)), float(np.max(P))
    z_min, z_max = float(np.min(Z)), float(np.max(Z))

    # consistent contour levels across all planes
    levs = np.linspace(z_min, z_max, int(levels))

    # offsets slightly outside the data range so they appear as "walls"
    z_offset = z_min - 0.12 * (z_max - z_min + 1e-12)
    p_offset = p_min - 0.10 * (p_max - p_min + 1e-12)
    x_offset = x_min - 0.10 * (x_max - x_min + 1e-12)

    # floor: x–p
    ax.contourf(X, P, Z, zdir="z", offset=z_offset,
                levels=levs, cmap=cmap, norm=norm, alpha=alpha_xy)

    # p-wall: x–W (constant p)
    ax.contourf(X, P, Z, zdir="y", offset=p_offset,
                levels=levs, cmap=cmap, norm=norm, alpha=alpha_xw)

    # x-wall: p–W (constant x)
    ax.contourf(X, P, Z, zdir="x", offset=x_offset,
                levels=levs, cmap=cmap, norm=norm, alpha=alpha_pw)

    # expand bounds to include the walls
    ax.set_zlim(z_offset, z_max)
    ax.set_ylim(p_offset, p_max)
    ax.set_xlim(x_offset, x_max)

def plot_wigner_3d(*,
                   state=None,
                   ket=None, N=None, cutoff=None,
                   mode=0,
                   grid_lim=4.0, grid_pts=161,
                   use_textbook_axes=True,
                   cmap_name="turbo",
                   downsample=2,
                   add_projections=True,
                   projection_levels=40,
                   projection_alpha_xy=0.95,
                   projection_alpha_xw=0.7,
                   projection_alpha_pw=0.7,
                   title=None):
    """
    Unified Wigner plotter.
    Provide either:
      - state=SF_state
    or:
      - ket=..., N=..., cutoff=...
    """
    if state is None:
        if ket is None or N is None or cutoff is None:
            raise ValueError("Provide either `state=...` or (`ket=...`, `N=...`, `cutoff=...`).")
        state = state_from_ket(ket, N, cutoff)

    # Axes conversion: SF uses its own units; if you want textbook axes, scale the grid
    if use_textbook_axes:
        x_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        p_phys = np.linspace(-grid_lim, grid_lim, grid_pts)
        xvec = np.sqrt(2) * x_phys
        pvec = np.sqrt(2) * p_phys
        x_plot, p_plot = x_phys, p_phys
        xlabel, ylabel = "x", "p"
    else:
        xvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        pvec = np.linspace(-grid_lim, grid_lim, grid_pts)
        x_plot, p_plot = xvec, pvec
        xlabel, ylabel = "x (SF units)", "p (SF units)"

    W = state.wigner(mode=mode, xvec=xvec, pvec=pvec)  # shape (len(xvec), len(pvec))

    X, P = np.meshgrid(x_plot, p_plot, indexing="xy")
    Z = W.T

    s = max(int(downsample), 1)
    Xs, Ps, Zs = X[::s, ::s], P[::s, ::s], Z[::s, ::s]

    zmin, zmax = float(np.min(Zs)), float(np.max(Zs))
    norm = Normalize(vmin=zmin, vmax=zmax)  # rainbow mapping across min..max (no white midpoint)
    cmap = matplotlib.colormaps[cmap_name]
    facecolors = cmap(norm(Zs))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Xs, Ps, Zs,
                    facecolors=facecolors,
                    linewidth=0,
                    antialiased=True,
                    shade=False)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Zs)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.78, pad=0.12)
    cbar.set_label("W(x,p)")

    if add_projections:
        _add_3plane_projections(
            ax, Xs, Ps, Zs,
            cmap=cmap, norm=norm,
            levels=projection_levels,
            alpha_xy=projection_alpha_xy,
            alpha_xw=projection_alpha_xw,
            alpha_pw=projection_alpha_pw,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("W(x,p)")
    ax.view_init(elev=32, azim=45)

    if title is None:
        title = f"3D Wigner with projections (mode={mode})"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()