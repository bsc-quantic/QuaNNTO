import numpy as np
import matplotlib.pyplot as plt

def x_hat(alpha, r):
    """
    <x_hat>(alpha, r) = sqrt(2) * ( e^r * alpha
                    + alpha * e^(2r) * sinh(r) / (alpha^2 * e^(2r) + sinh(r)^2) )
    """
    er = np.exp(r)
    e2r = np.exp(2 * r)
    sh = np.sinh(r)

    denom = alpha**2 * e2r + sh**2

    # Avoid division by zero (alpha=0 and r=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(denom == 0, 0.0, alpha * e2r * sh / denom)

    return np.sqrt(2) * (er * alpha + frac)

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

#ax1.set_xlabel(r'$\alpha$', fontsize=15)
ax1.set_xticklabels([])
ax1.set_ylabel(r'$\langle \hat{x} \rangle$', fontsize=15)
ax1.set_ylim(top=5.01, bottom=-5.01)
ax1.set_xlim(left=-2.01, right=2.01)
ax1.set_title(r'$\langle \hat{x} \rangle$ for $r \geq 0$', fontsize=16)
ax1.grid(False)
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

ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
ax2.set_xlabel(r'$\alpha$', fontsize=15)
ax2.set_ylabel(r'$\langle \hat{x} \rangle$', fontsize=15)
ax2.set_ylim(top=1.01, bottom=-1.01)
ax2.set_xlim(left=-2.01, right=2.01)
ax2.set_title(r'$\langle \hat{x} \rangle$ for $r \leq 0$', fontsize=16)
ax2.grid(False)
ax2.legend(fontsize=legend_fontsize)
fig1.tight_layout()
fig2.tight_layout()
fig1.savefig("figures/subtraction-sq-momentum.pdf")
fig2.savefig("figures/subtraction-sq-position.pdf")
plt.show()
