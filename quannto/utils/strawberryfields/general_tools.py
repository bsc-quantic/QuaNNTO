import numpy as np
import math
import scipy.integrate as si
if not hasattr(si, "simps") and hasattr(si, "simpson"):
    si.simps = si.simpson
import strawberryfields as sf
from strawberryfields.ops import Ket

def annihilation_matrix(cutoff):
    a = np.zeros((cutoff, cutoff), dtype=complex)
    for n in range(cutoff - 1):
        a[n, n + 1] = np.sqrt(n + 1.0)
    return a

def ladder_ops(cutoff):
    """Return (a, adag) in the truncated Fock basis."""
    a = annihilation_matrix(cutoff)
    adag = a.conj().T
    return a, adag

def coherent_ket(alpha, cutoff):
    """|alpha> in the Fock basis up to 'cutoff' (standard convention)."""
    n = np.arange(cutoff)
    coeffs = np.exp(-0.5 * np.abs(alpha)**2) * (alpha**n) / np.sqrt(np.maximum(1, np.array([math.factorial(k) for k in n])))
    # factorial(0)=1; the max(1, ...) avoids divide-by-zero in vectorized form (safe anyway)
    coeffs[0] = np.exp(-0.5 * np.abs(alpha)**2)  # ensure exact for n=0
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

def apply_annihilation_to_ket(ket, cutoff, mode):
    """
    Apply a_mode to a pure ket in truncated Fock basis and renormalize.
    Returns (ket_new, success_prob_conditional).
    """
    dim = ket.size
    N = int(round(np.log(dim) / np.log(cutoff)))
    psi = ket.reshape([cutoff] * N)

    a = annihilation_matrix(cutoff)

    out = np.tensordot(a, psi, axes=([1], [mode]))  # contracts a_in with mode axis
    out = np.moveaxis(out, 0, mode)                 # put output axis back into position
    out_flat = out.reshape(-1)
    norm = np.linalg.norm(out_flat)
    if norm < 1e-14:
        raise RuntimeError(
            f"Photon subtraction produced ~zero state on mode {mode}. "
            "Try increasing cutoff or check the state has photons in that mode."
        )
    p = norm**2  # conditional success prob of the annihilation
    return (out_flat / norm).reshape([cutoff]*N), p

def six_order_moments_operators(cutoff):
    a, adag = ladder_ops(cutoff)
    n_op = adag @ a
    
    a1 = a
    
    a2  = a @ a
    n1  = n_op
    
    a3 = a @ a @ a
    a2adag = a @ a @ adag
    
    n2  = n_op @ n_op
    a4 = a @ a @ a @ a
    a3adag = a @ a @ a @ adag
    
    a5 = a @ a @ a @ a @ a
    a4adag = a @ a @ a @ a @ adag
    a3adag2 = a @ a @ a @ adag @ adag
    
    n3 = n_op @ n_op @ n_op
    a6 = a @ a @ a @ a @ a @ a
    a5adag = a @ a @ a @ a @ a @ adag
    a4adag2 = a @ a @ a @ a @ adag @ adag
    
    return dict(a1=a1, a2=a2, n1=n1, a3=a3, a2adag=a2adag, n2=n2, a4=a4, a3adag=a3adag, a5=a5, a4adag=a4adag, a3adag2=a3adag2, n3=n3, a6=a6, a5adag=a5adag, a4adag2=a4adag2)

def moments_from_rho(rho, moments):
    """Compute the specified statistical moments of a density matrix rho."""
    tr = lambda O: np.trace(rho @ O)
    
    moments_keys = moments.keys()
    moments_expvals = {key : tr(moments[key]) for key in moments_keys}
    
    return moments_expvals

def mode_moments_from_ket(ket, moments, cutoff, mode=0):
    """
    Compute single-mode moments on 'mode' from an N-mode pure ket 
    without building the reduced density matrix.
    """
    dim = ket.size
    N = int(round(np.log(dim) / np.log(cutoff)))
    ket = np.asarray(ket, dtype=complex)
    if ket.ndim == 1:
        ket = ket.reshape([cutoff]*N)
    else:
        assert ket.shape == tuple([cutoff]*N), f"Expected {(cutoff,)*N}, got {ket.shape}"

    # normalize defensively
    psi_flat = ket.reshape(-1)
    nrm = np.linalg.norm(psi_flat)
    if nrm < 1e-14:
        raise ValueError("psi has ~zero norm")
    ket = ket / nrm

    # move the target mode to axis 0 and flatten the rest as "environment"
    ket0 = np.moveaxis(ket, mode, 0).reshape(cutoff, -1)  # (cutoff, env_dim)

    # <O_mode> = Tr( psi0^† (O psi0) )
    def expval(O):
        return np.trace(ket0.conj().T @ (O @ ket0))
    
    moments_keys = moments.keys()
    moments_expvals = {key : expval(moments[key]) for key in moments_keys}
    return moments_expvals

def state_from_ket(ket, N, cutoff):
    """Load a ket into SF and return the state."""
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(N)
    with prog.context as q:
        Ket(ket) | tuple(q[i] for i in range(N))
    return eng.run(prog).state

def mode_view_from_ket(ket, N, cutoff, mode):
    """Return Psi_mode as shape (cutoff, env_dim) from an N-mode ket."""
    psi = np.asarray(ket, dtype=complex)
    if psi.ndim == 1:
        psi = psi.reshape([cutoff]*N)
    psi = psi / np.linalg.norm(psi.reshape(-1))
    return np.moveaxis(psi, mode, 0).reshape(cutoff, -1)

def fock_probs_from_ket(ket, N, cutoff, mode=0):
    """P(n) = diag(rho_mode) without building rho_mode."""
    Psi = mode_view_from_ket(ket, N, cutoff, mode)          # (cutoff, env_dim)
    probs = np.sum(np.abs(Psi)**2, axis=1).real             # sum over environment
    return probs

def reduced_rho_from_ket(ket, N, cutoff, mode):
    Psi = mode_view_from_ket(ket, N, cutoff, mode)          # (cutoff, env_dim)
    rho = Psi @ Psi.conj().T                                 # (cutoff, cutoff)
    return rho / np.trace(rho)

def state_from_single_mode_rho(rho, cutoff):
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (cutoff, cutoff):
        raise ValueError(f"rho must be ({cutoff},{cutoff}), got {rho.shape}")

    # Make sure it's normalized (often already is if it came from reduced_dm)
    tr = np.trace(rho)
    if abs(tr) < 1e-14:
        raise ValueError("rho has ~zero trace")
    rho = rho / tr

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(1)
    with prog.context as q:
        sf.ops.DensityMatrix(rho) | q[0]   # load the state in Fock basis
    return eng.run(prog).state

def catstate_moments(alpha, moments, cutoff=20, phi=0.0):
    """
    For each alpha in alpha_list, prepare the even cat state and compute:
    [<a>, <a^2>, <a^3>, <a^\dagger n>, <n>]
    Returns: list of tuples (alpha, np.array([...])).
    """
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
    # Prepare the cat ket and load it into the backend
    ket = cat_even_ket(alpha, cutoff, phi=phi)
    prog = sf.Program(1)
    with prog.context as q:
        sf.ops.Ket(ket) | q[0]
    result = eng.run(prog)
    state = result.state
    rho = state.reduced_dm(0)  # single-mode density matrix, shape (cutoff, cutoff)
    obs = moments_from_rho(rho, moments)
    moments_expvals = np.array([obs["a1"], obs["a2"], obs["n1"], obs["a3"], obs["a2adag"], obs["n2"], obs["a4"], obs["a3adag"], obs["a5"], obs["a4adag"], obs["a3adag2"], obs["n3"], obs["a6"], obs["a5adag"], obs["a4adag2"]], dtype=complex)

    return moments_expvals

def fidelity_pure_vs_mixed(ket_pure, rho_mix):
    """F = <ket|rho|ket> for pure |ket> and density matrix rho."""
    psi = np.asarray(ket_pure, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    return float(np.real_if_close(np.vdot(psi, rho_mix @ psi)))

def fidelity_two_pure(ket1, ket2):
    """F = <ket1|ket2> for two pure states |ket1> and |ket2>."""
    psi = np.asarray(ket1, dtype=complex).reshape(-1)
    phi = np.asarray(ket2, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    phi = phi / np.linalg.norm(phi)
    return float(np.real_if_close(np.vdot(psi, phi)))
