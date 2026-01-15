import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# n-photon–added COHERENT state (NO squeezing):
#   |psi_n(α)> ∝ (a†)^n D(α)|0> = (a†)^n |α>,   α ∈ R
#
# With x = (a + a†)/sqrt(2), the normalized expectation value is:
#   <x>_n(α) = sqrt(2) * α * L_n^{1}(-α^2) / L_n(-α^2)
#
# Here we hard-code Laguerre polynomials for n=0..4 to avoid extra deps.
# ============================================================

def L_n_neg_a2(n, a2):
    """L_n(x) at x=-a^2 for n=0..4."""
    x = -a2
    if n == 0: return 1.0 + 0.0*x
    if n == 1: return 1.0 - x
    if n == 2: return 0.5*x**2 - 2.0*x + 1.0
    if n == 3: return -(1.0/6.0)*x**3 + 1.5*x**2 - 3.0*x + 1.0
    if n == 4: return (1.0/24.0)*x**4 - (2.0/3.0)*x**3 + 3.0*x**2 - 4.0*x + 1.0
    raise ValueError("Implemented only for n=0..4")

def L_n1_neg_a2(n, a2):
    """Associated Laguerre L_n^1(x) at x=-a^2 for n=0..4."""
    x = -a2
    if n == 0: return 1.0 + 0.0*x
    if n == 1: return 2.0 - x
    if n == 2: return 0.5*x**2 - 3.0*x + 3.0
    if n == 3: return -(1.0/6.0)*x**3 + 2.0*x**2 - 6.0*x + 4.0
    if n == 4: return (1.0/24.0)*x**4 - (5.0/6.0)*x**3 + 5.0*x**2 - 10.0*x + 5.0
    raise ValueError("Implemented only for n=0..4")

def x_hat(alpha, n, eps=1e-14):
    r"""
    <x_hat>(alpha) for n-photon addition on coherent states (r=0):

        <x>_n(α) = sqrt(2) * α * L_n^1(-α^2) / L_n(-α^2)
    """
    alpha = np.asarray(alpha, dtype=float)
    a2 = alpha**2
    Ln  = L_n_neg_a2(n, a2)
    Ln1 = L_n1_neg_a2(n, a2)
    return np.sqrt(2.0) * alpha * (Ln1 / (Ln + eps))


# =======================
# Plot formatting: SAME as your previous script
# =======================

os.makedirs("figures", exist_ok=True)

# Alpha range
alphas = np.linspace(-3, 3, 1000)

# We'll map "n" onto your existing (colors, linestyles) arrays by index:
# index 0..4  ->  n = 0..4
n_values = [0, 1, 2, 3, 4]

colors = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = [
    '-',         # index 0
    '--',        # index 1
    '-.',        # index 2
    (0, (5, 2)), # index 3
    (0, (3, 1, 1, 1))  # index 4
]
line_width = 2.7
legend_fontsize = 14

# --------- Single plot (multiple n), keeping your style knobs ----------
fig, ax = plt.subplots()

for n in n_values:
    y = x_hat(alphas, n)
    ax.plot(
        alphas, y,
        color=colors[n],
        linestyle=linestyles[n],
        linewidth=line_width,
        label=(r'$n=0$ (coherent)' if n == 0 else fr'$n={n}$')
    )

ax.set_xlabel(r'$\alpha$', fontsize=15)
ax.set_ylabel(r'$\langle \hat{x} \rangle$', fontsize=15)
ax.set_xlim(left=-2.01, right=2.01)
ax.set_ylim(top=4.01, bottom=-4.01)
ax.set_title(r'$\langle \hat{x} \rangle$ for $n$ photon additions on coherent states', fontsize=15)
ax.grid(True)
ax.legend(fontsize=legend_fontsize)

fig.savefig("figures/coherent_addition_multiple_n.pdf", bbox_inches="tight")
plt.show()
