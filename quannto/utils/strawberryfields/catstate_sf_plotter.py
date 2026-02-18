import numpy as np

from quannto.utils.path_utils import figures_dir, models_operators_path
from quannto.utils.strawberryfields.general_tools import cat_even_ket, fidelity_pure_vs_mixed, fidelity_two_pure, mode_view_from_ket, reduced_rho_from_ket, six_order_moments_operators, mode_moments_from_ket
from quannto.utils.strawberryfields.wigner_plotters import *
from quannto.utils.strawberryfields.qonn_builder import run_qonn


N = 3
L = 1
subtractions = [[1,2]]
qnn_label = f'N={N}, L={L}, â in modes {[m+1 for m in subtractions[0]]}'
print(qnn_label)
cat_cutoff = 20
qonn_cutoff = 20
trained_stats = 15
input_alpha = -1.0 + 0.0j
phi = 0.0
qonn_model_name = f'catstate_phi{phi}_trainsize1_stats{trained_stats}_cut{cat_cutoff}_rng-1.0to-1.0_N{N}_L{L}_sub{subtractions}_inNone'

S_l = np.load(models_operators_path(qonn_model_name, "S_l", "npy"))
D_l = np.load(models_operators_path(qonn_model_name, "D_l", "npy"))

layers = []
for S,d in zip(S_l, D_l):
    layers.append({"S": S, "d": d})

title_cat = f"Wigner of cat state (alpha={input_alpha}, cutoff={cat_cutoff})"
title_qonn = f"Wigner of QONN ({qnn_label}) synthesized cat state (alpha={input_alpha}, cutoff={qonn_cutoff})"

ket_cat = cat_even_ket(input_alpha, cat_cutoff, phi=phi)
ket_out, probs = run_qonn(
    N=N, cutoff=qonn_cutoff,
    layers=layers,
    subtractions=subtractions,
    input_alpha=input_alpha   # Input encoding via displacement on mode 0, otherwise None
)
print("Conditional click probs per layer:", probs)

""" rho_cat = state_from_ket(ket_cat, N=1, cutoff=cat_cutoff).reduced_dm(0)
moments_cat = moments_from_rho(rho_cat)
rho_qonn = state_from_ket(ket_out, N=N, cutoff=qonn_cutoff).reduced_dm(0)
moments_qonn = moments_from_rho(rho_qonn) """
cat_stat_ops = six_order_moments_operators(cat_cutoff)
qonn_stat_ops = six_order_moments_operators(qonn_cutoff)
moments_cat = mode_moments_from_ket(ket_cat, cat_stat_ops, cutoff=cat_cutoff, mode=0)
moments_qonn = mode_moments_from_ket(ket_out, qonn_stat_ops, cutoff=qonn_cutoff, mode=0)

plot_wigner_2d_compare(
    ket_L=ket_cat,  N_L=1, cutoff_L=cat_cutoff, mode_L=0,
    ket_R=ket_out,   N_R=N, cutoff_R=qonn_cutoff, mode_R=0,
    grid_lim=4.0, grid_pts=201,
    zmin=-0.05, zmax=0.175,   # or leave None to auto-compute global min/max
    title_L="Theoretical", title_R="Synthesized",
    savepath=figures_dir() / "wigner_compare.pdf",
)

reduced_state_out = reduced_rho_from_ket(ket_out, cutoff=qonn_cutoff, N=N, mode=0)
if cat_cutoff == qonn_cutoff:
    if N==1:
        F0 = fidelity_two_pure(ket_cat, ket_out)
    else:
        F0 = fidelity_pure_vs_mixed(ket_cat, reduced_state_out)
    print("Fidelity on measured mode 0 (cat vs reduced QONN):", F0)

input("Press Enter to continue with losses...")
total_loss = 0
count = 0
for key in moments_cat.keys():
    count += 1
    val_cat = moments_cat[key]
    val_qonn = moments_qonn[key]
    print(f"Moment {count}: expected={val_cat:.6f}, QONN={val_qonn:.6f}, diff={(val_cat - val_qonn)**2:.6e}")
    total_loss += np.abs(val_cat - val_qonn)**2
print("Total moment loss:", total_loss)
input("Press Enter to continue with plotting...")

plot_state_diagnostics(ket=ket_cat, N=1, cutoff=cat_cutoff, mode=0, grid_lim=5, grid_pts=250, title_prefix=rf"Theoretical cat state $\alpha$={input_alpha} $\phi$={phi}")
plot_state_diagnostics(ket=ket_out, N=N, cutoff=qonn_cutoff, mode=0, grid_lim=5, grid_pts=250, title_prefix=rf"QONN ({qnn_label}) synthesized cat state $\alpha$={input_alpha} $\phi$={phi}")

plot_wigner_2d(ket=ket_cat, N=1, cutoff=cat_cutoff, grid_lim=4.5, grid_pts=250, use_textbook_axes=True, title=title_cat)
plot_wigner_2d(ket=ket_out, N=N, cutoff=qonn_cutoff, grid_lim=4.5, grid_pts=250, use_textbook_axes=True, title=title_qonn)
plot_wigner_3d(
    ket=ket_cat, N=1, cutoff=cat_cutoff, mode=0,
    grid_lim=4.0, grid_pts=180,
    cmap_name="turbo",
    downsample=2,
    add_projections=True,
    title=f"Wigner of cat state (alpha={input_alpha}, phi={phi})"
)
plot_wigner_3d(
    ket=ket_out, N=N, cutoff=qonn_cutoff, mode=0,
    grid_lim=4.0, grid_pts=180,
    cmap_name="turbo",
    downsample=2,
    add_projections=True,
    title=f"Wigner of QONN ({qnn_label}) synthesized cat state (alpha={input_alpha}, phi={phi})"
)

