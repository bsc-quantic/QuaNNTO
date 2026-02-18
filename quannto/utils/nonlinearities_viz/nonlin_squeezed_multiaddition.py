import os
import math
import numpy as np
import matplotlib.pyplot as plt

from quannto.utils.path_utils import figures_dir

# ------------------------------------------------------------
# Photon-added squeezed coherent state:
#   |psi_n>  ∝  (a†)^n S(r) D(α) |0>,   with α,r ∈ R
# Expectation wanted (your convention):
#   x = (a + a†)/sqrt(2)
#
# Using the Gaussian moments you provided for the pre-addition state ρ_{rα}:
#   <a> = e^r α
#   <Δa Δa†> = cosh^2 r
#   <Δa Δa>  = 1/2 sinh(2r) = sinh r cosh r   (NOTE: this fixes the squeezing sign convention)
#   <Δa† Δa> = sinh^2 r
#
# For the photon-added state:
#   <O>_n = < a^n O (a†)^n >_{ρ_{rα}} / < a^n (a†)^n >_{ρ_{rα}}
# and since the products here are anti-normally ordered, we can evaluate moments
# <a^p (a†)^q> via Wick/Isserlis on the (a, a†) Gaussian statistics.
# ------------------------------------------------------------

def central_moments_table(max_pq: int, C: complex, S: complex):
    """
    Build table M[i,j] = <ξ^i (ξ*)^j> for zero-mean complex Gaussian ξ with:
      <ξ ξ*> = C
      <ξ ξ>  = S
      <ξ* ξ*> = S*
    using a Wick pairing recursion.
    """
    M = np.zeros((max_pq + 1, max_pq + 1), dtype=complex)
    M[0, 0] = 1.0 + 0j

    # Fill by increasing total order i+j so dependencies are already computed.
    for total in range(1, 2 * max_pq + 1):
        for i in range(0, min(max_pq, total) + 1):
            j = total - i
            if j < 0 or j > max_pq:
                continue
            if (i + j) % 2 == 1:
                M[i, j] = 0.0 + 0j
                continue

            if i > 0:
                val = 0.0 + 0j
                if i >= 2:
                    val += (i - 1) * S * M[i - 2, j]
                if j >= 1:
                    val += j * C * M[i - 1, j - 1]
                M[i, j] = val
            else:
                # i == 0: only ξ* variables remain, pair ξ* with ξ* using S*
                if j >= 2:
                    M[0, j] = (j - 1) * np.conj(S) * M[0, j - 2]
                else:
                    M[0, j] = 0.0 + 0j

    return M

def moment_ap_aq(mu, p: int, q: int, M):
    """
    Compute <(mu+ξ)^p (mu*+ξ*)^q> given central moments table M[i,j]=<ξ^i (ξ*)^j>.
    Works with mu being a scalar or a numpy array.
    """
    mu = np.asarray(mu, dtype=complex)
    mu_c = np.conj(mu)

    # Precompute powers up to p and q (vectorized).
    mu_pows = [np.ones_like(mu)]
    for _ in range(p):
        mu_pows.append(mu_pows[-1] * mu)

    mu_c_pows = [np.ones_like(mu)]
    for _ in range(q):
        mu_c_pows.append(mu_c_pows[-1] * mu_c)

    res = 0.0 + 0j
    for i in range(p + 1):
        cp = math.comb(p, i)
        mu_factor = mu_pows[p - i]
        for j in range(q + 1):
            cq = math.comb(q, j)
            res += cp * cq * mu_factor * mu_c_pows[q - j] * M[i, j]
    return res

def x_hat_n(alpha, r, n: int):
    """
    <x>(alpha,r) for n-photon addition: |psi_n> ∝ (a†)^n S(r)D(alpha)|0>.
    alpha can be scalar or numpy array. r scalar. n nonnegative int.
    """
    alpha = np.asarray(alpha, dtype=float)
    mu = np.exp(r) * alpha  # <a> = e^r α  (your provided mean)

    # Your provided (central) second moments in ladder-operator form:
    C = (np.cosh(r) ** 2)              # <Δa Δa†>
    S = (np.sinh(r) * np.cosh(r))      # <Δa Δa> = 1/2 sinh(2r)

    # Need central moments up to max(p,q)=n+1
    M = central_moments_table(n + 1, C, S)

    denom = moment_ap_aq(mu, n, n, M)  # <a^n (a†)^n>
    num = (moment_ap_aq(mu, n + 1, n, M) + moment_ap_aq(mu, n, n + 1, M)) / np.sqrt(2)

    # Expectation is real for real alpha,r in this convention
    return (num / denom).real

# ---------------- Plotting ----------------
os.makedirs("figures", exist_ok=True)

alphas = np.linspace(-3, 3, 1000)
r_values = [0.0, 0.1, 0.25, 0.5, 1.0]

colors = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = [
    '-',         # r = 0.0
    '--',        # r = 0.1
    '-.',        # r = 0.25
    (0, (5, 2)), # r = 0.5
    (0, (3, 1, 1, 1))  # r = 1.0
]
line_width = 2.7
legend_fontsize = 12

n_values = [0, 1, 2]
for n in n_values:
    # --------- Plot r >= 0 ---------
    fig1, ax1 = plt.subplots()
    for r, c, ls in zip(r_values, colors, linestyles):
        y = x_hat_n(alphas, r, n)
        ax1.plot(alphas, y, color=c, linestyle=ls, linewidth=line_width,
                 label=fr'$r={r}$')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\langle \hat{x} \rangle$')
    ax1.set_xlim(left=-2.01, right=2.01)
    ax1.set_title(fr'$\langle \hat{{x}} \rangle$ for $n={n}$ photon addition, $r \geq 0$')
    ax1.grid(True)
    ax1.legend(fontsize=legend_fontsize)
    fig1.savefig(figures_dir() / f"addition_xhat_n{n}_rpos.pdf", bbox_inches="tight")

    # --------- Plot r <= 0 ---------
    fig2, ax2 = plt.subplots()
    for r, c, ls in zip(r_values, colors, linestyles):
        r_neg = -r
        y = x_hat_n(alphas, r_neg, n)
        ax2.plot(alphas, y, color=c, linestyle=ls, linewidth=line_width,
                 label=fr'$r={r_neg}$')
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\langle \hat{x} \rangle$')
    ax2.set_xlim(left=-2.01, right=2.01)
    ax2.set_title(fr'$\langle \hat{{x}} \rangle$ for $n={n}$ photon addition, $r \leq 0$')
    ax2.grid(True)
    ax2.legend(fontsize=legend_fontsize)
    fig2.savefig(figures_dir() / f"addition_xhat_n{n}_rneg.pdf", bbox_inches="tight")

plt.show()
