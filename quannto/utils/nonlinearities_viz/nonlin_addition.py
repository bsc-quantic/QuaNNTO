import numpy as np
import matplotlib.pyplot as plt

def x_hat(alpha, r):
    r"""
    <x_hat>(alpha, r) for photon addition:

        <x> = sqrt(2) * alpha * e^r *
              ( alpha^2 e^{2r} + sinh(r) cosh(r) + 2 cosh^2(r) )
              / ( alpha^2 e^{2r} + sinh^2(r) + 1 )
    """
    er = np.exp(r)
    e2r = np.exp(2 * r)
    sh = np.sinh(r)
    ch = np.cosh(r)

    num_inner = alpha**2 * e2r + sh * ch + 2 * ch**2
    denom = alpha**2 * e2r + sh**2 + 1  # always > 0

    return np.sqrt(2) * alpha * er * num_inner / denom

# Alpha range
alphas = np.linspace(-3, 3, 1000)

# Squeezing values to plot
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
legend_fontsize = 14

# --------- Plot r >= 0 ---------
fig1, ax1 = plt.subplots()
for r, c, ls in zip(r_values, colors, linestyles):
    y = x_hat(alphas, r)
    ax1.plot(alphas, y, color=c, linestyle=ls, linewidth=line_width,
             label=fr'$r={r}$')
ax1.set_xlabel(r'$\alpha$', fontsize=15)
#ax1.set_ylabel(r'$\langle \hat{x} \rangle$')
ax1.set_ylim(top=6.01, bottom=-6.01)
ax1.set_xlim(left=-2.01, right=2.01)
ax1.set_title(r'$\langle \hat{x} \rangle$ of single-addition on $|r\alpha\rangle$ with $r \geq 0$', fontsize=15)
ax1.grid(True)
ax1.legend(fontsize=legend_fontsize)

# --------- Plot r <= 0 ---------
fig2, ax2 = plt.subplots()
for r, c, ls in zip(r_values, colors, linestyles):
    r_neg = -r
    if r_neg == 0:
        r_neg = 0.0  # to avoid -0.0 label
    y = x_hat(alphas, r_neg)
    ax2.plot(alphas, y, color=c, linestyle=ls, linewidth=line_width,
             label=fr'$r={r_neg}$')
ax2.set_xlabel(r'$\alpha$', fontsize=15)
#ax2.set_ylabel(r'$\langle \hat{x} \rangle$')
ax2.set_ylim(top=3.01, bottom=-3.01)
ax2.set_xlim(left=-2.01, right=2.01)
ax2.set_title(r'$\langle \hat{x} \rangle$ of single-addition on $|r\alpha\rangle$ with $r \leq 0$', fontsize=15)
ax2.grid(True)
ax2.legend(fontsize=legend_fontsize)
fig1.savefig("figures/addition_nonlinearity_rpos.pdf", bbox_inches="tight")
fig2.savefig("figures/addition_nonlinearity_rneg.pdf", bbox_inches="tight")
plt.show()
