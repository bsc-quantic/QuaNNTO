# cat_dataset_sf.py
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Ket

# ---------- Utilities ----------

def coherent_ket(alpha, cutoff):
    """|alpha> in the Fock basis up to 'cutoff' (standard convention)."""
    n = np.arange(cutoff)
    coeffs = np.exp(-0.5 * abs(alpha)**2) * (alpha**n) / np.sqrt(np.maximum(1, np.array([np.math.factorial(k) for k in n])))
    # factorial(0)=1; the max(1, ...) avoids divide-by-zero in vectorized form (safe anyway)
    coeffs[0] = np.exp(-0.5 * abs(alpha)**2)  # ensure exact for n=0
    return coeffs.astype(complex)

def cat_even_ket(alpha, cutoff, phi=0.0):
    """
    Even-cat ket for fixed phi=0:
        |Cat(α,0)> ∝ |α> + |-α>
    (This also works for general phi if you replace the '+' by '+ e^{i phi} |-α>'.)
    """
    ket_plus  = coherent_ket(+alpha, cutoff)
    ket_minus = coherent_ket(-alpha, cutoff)
    unnorm = ket_plus + ket_minus * np.exp(1j*phi)
    # Normalization: N = 1/sqrt(2(1 + e^{-2|α|^2} cos φ))
    s = np.exp(-2 * (abs(alpha)**2))
    N = 1.0 / np.sqrt(2.0 * (1.0 + s * np.cos(phi)))
    return (N * unnorm).astype(complex)

def ladder_ops(cutoff):
    """Return (a, adag, n_op) in the truncated Fock basis."""
    a = np.zeros((cutoff, cutoff), dtype=complex)
    for n in range(cutoff - 1):
        a[n, n+1] = np.sqrt(n + 1.0)
    adag = a.conj().T
    n_op = adag @ a
    return a, adag, n_op

def moments_from_rho(rho):
    """Compute <a>, <a^2>, <a^3>, <a^† n>, <n> from density matrix rho."""
    cutoff = rho.shape[0]
    a, adag, n_op = ladder_ops(cutoff)
    # Helper: Tr[rho @ O]
    tr = lambda O: np.trace(rho @ O)
    a2  = tr(a @ a)
    n1  = tr(n_op)           # <n>, real by construction
    n2  = tr(n_op @ n_op)
    return dict(a2=a2, n=n1, n2=n2)

# ---------- Dataset builder ----------

def build_cat_dataset(alpha_list, cutoff=20, phi=0.0):
    """
    For each alpha in alpha_list, prepare the even cat state and compute:
    [<a>, <a^2>, <a^3>, <a^\dagger n>, <n>]
    Returns: list of tuples (alpha, np.array([...])).
    """
    inputs_dataset = []
    outputs_dataset = []
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
    for alpha in alpha_list:
        # Prepare the cat ket and load it into the backend
        ket = cat_even_ket(alpha, cutoff, phi=phi)
        prog = sf.Program(1)
        with prog.context as q:
            Ket(ket) | q[0]
        result = eng.run(prog)
        state = result.state
        rho = state.reduced_dm(0)  # single-mode density matrix, shape (cutoff, cutoff)
        obs = moments_from_rho(rho)
        y = np.array([obs["a2"], obs["n"], obs["n2"]], dtype=complex)
        inputs_dataset.append([alpha])
        outputs_dataset.append(y)
    return inputs_dataset, outputs_dataset

# Grid of real alphas in some range
dataset_size = 50
alpha_range = (-1, 1)
cutoff=20
phi=0.0
alphas = np.linspace(alpha_range[0], alpha_range[1], dataset_size)
inputs_dataset, outputs_dataset = build_cat_dataset(alphas, cutoff=cutoff, phi=phi)  # even cat
ds = [inputs_dataset, outputs_dataset]

with open(f"datasets/catstate_phi{phi}_trainsize{dataset_size}_rng{alpha_range[0]}to{alpha_range[-1]}_inputs.npy", "wb") as f:
    np.save(f, np.array(inputs_dataset))
with open(f"datasets/catstate_phi{phi}_trainsize{dataset_size}_rng{alpha_range[0]}to{alpha_range[-1]}_outputs.npy", "wb") as f:
    np.save(f, np.array(outputs_dataset))

# Quick sanity checks for real α, even cat (phi=0):
# <a> ≈ 0, <a^3> ≈ 0; <a^2> ≈ α^2 (edge effects if cutoff too small or |α| large)
for alpha, vec in ds[::5]:  # print a few
    a1, a2, a3, adag_n, n1, n2 = vec
    print(f"alpha={alpha:+.2f}  <a>={a1:.3e}  <a^2>={a2:.3e}  <a^3>={a3:.3e}  <a† n>={adag_n:.3e}  <n>={n1:.3f} <n2>={n2:.3f}")