import numpy as np
from scipy.special import gammaln

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
    """Unnormalized projection of |alpha> on levels 0,...,cutoff-1.

    Keeping the truncated norm lets callers diagnose the missing Fock tail.
    Log-factorials avoid object arrays and overflow at cutoffs above 21.
    """
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, np.integer)) or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    alpha = complex(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha == 0:
        ket = np.zeros(cutoff, dtype=complex)
        ket[0] = 1
        return ket
    n = np.arange(cutoff)
    return np.exp(-abs(alpha)**2 / 2 + n * np.log(alpha) - gammaln(n + 1) / 2)

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
    norm_squared = 2.0 * (1.0 + s * np.cos(phi))
    if not np.isfinite(norm_squared) or norm_squared <= 0:
        raise ValueError("The requested cat superposition has zero or invalid norm")
    N = 1.0 / np.sqrt(norm_squared)
    return (N * unnorm).astype(complex)

def apply_annihilation_to_ket(ket, cutoff, mode):
    """
    Apply a_mode to a pure ket in truncated Fock basis and renormalize.
    Returns (ket_new, squared_norm). This ideal-annihilation weight is not
    a physical heralding probability; it can exceed one.
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
    p = norm**2  # squared norm for the ideal, unbounded operator a
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
        return np.vdot(ket0, O @ ket0)
    
    moments_keys = moments.keys()
    moments_expvals = {key : expval(moments[key]) for key in moments_keys}
    return moments_expvals

def state_from_ket(ket, N, cutoff):
    """Load a ket into SF and return the state."""
    from strawberryfields import Engine, Program
    from strawberryfields.ops import Ket

    eng = Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = Program(N)
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

    from strawberryfields import Engine, Program
    from strawberryfields.ops import DensityMatrix

    eng = Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = Program(1)
    with prog.context as q:
        DensityMatrix(rho) | q[0]   # load the state in Fock basis
    return eng.run(prog).state

def catstate_moments(alpha, moments, cutoff=20, phi=0.0):
    """
    For each alpha in alpha_list, prepare the even cat state and compute:
    [<a>, <a^2>, <a^3>, <a^\dagger n>, <n>]
    Returns: list of tuples (alpha, np.array([...])).
    """
    from strawberryfields import Engine, Program
    from strawberryfields.ops import Ket

    eng = Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
    # Prepare the cat ket and load it into the backend
    ket = cat_even_ket(alpha, cutoff, phi=phi)
    prog = Program(1)
    with prog.context as q:
        Ket(ket) | q[0]
    result = eng.run(prog)
    state = result.state
    rho = state.reduced_dm(0)  # single-mode density matrix, shape (cutoff, cutoff)
    obs = moments_from_rho(rho, moments)
    moments_expvals = np.array([obs["a1"], obs["a2"], obs["n1"], obs["a3"], obs["a2adag"], obs["n2"], obs["a4"], obs["a3adag"], obs["a5"], obs["a4adag"], obs["a3adag2"], obs["n3"], obs["a6"], obs["a5adag"], obs["a4adag2"]], dtype=complex)

    return moments_expvals

def cubicphase_moments(alpha, gamma, cutoff=20):
    """
    Returns a dict of expectation values for {a, a†, , p^2, x^3, p^3}
    after applying Vgate(gamma) to |alpha>, using cutoff n_max.
    """
    # 1) Run the SF program
    from strawberryfields import Engine, Program
    from strawberryfields.ops import Coherent, Vgate

    eng = Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = Program(1)
    
    # 2) Generate the coherent state and apply cubic phase gate V
    with prog.context as q:
        Coherent(alpha)    | q[0]
        Vgate(gamma)  | q[0]
    state = eng.run(prog).state

    rho = state.reduced_dm(0)  # shape = (cutoff, cutoff)

    # 3) Compute statistical moments of the state (FIXME: Refactor moments hardcoding)
    a = np.zeros((cutoff, cutoff), dtype=complex)
    for n in range(cutoff-1):
        a[n, n+1] = np.sqrt(n+1)
    adag = a.conj().T
    
    a_mean = np.trace(rho @ a)
    
    a2 = np.trace(rho @ (a @ a))
    n = np.trace(rho @ (adag @ a))

    a3 = np.trace(rho @ (a @ a @ a))
    adag_n = np.trace(rho @ (adag @ adag @ a))
    
    n2 = np.trace(rho @ (adag @ a @ adag @ a))

    return {
        "a":  a_mean,
        "a2": a2,
        "a3": a3,
        "a†n": adag_n,
        "n": n,
        "n2": n2
    }

def _normalized_ket(ket):
    psi = np.asarray(ket, dtype=complex).reshape(-1)
    norm = np.linalg.norm(psi)
    if not np.all(np.isfinite(psi)) or not np.isfinite(norm) or norm <= 0:
        raise ValueError("ket must be finite and have nonzero norm")
    return psi / norm


def fidelity_pure_vs_mixed(ket_pure, rho_mix):
    """Squared fidelity <psi|rho|psi>, normalizing both input states.

    Truncation convergence must be checked separately before interpreting this
    value as a fidelity of the corresponding infinite-dimensional states.
    """
    psi = _normalized_ket(ket_pure)
    rho = np.asarray(rho_mix, dtype=complex)
    if rho.shape != (psi.size, psi.size) or not np.all(np.isfinite(rho)):
        raise ValueError("rho must be a finite square matrix matching the ket")
    if not np.allclose(rho, rho.conj().T, rtol=0, atol=1e-10):
        raise ValueError("rho must be Hermitian")
    trace = np.trace(rho).real
    if trace <= 0 or not np.isfinite(trace):
        raise ValueError("rho must have positive finite trace")
    rho = rho / trace
    if np.linalg.eigvalsh(rho).min() < -1e-10:
        raise ValueError("rho must be positive semidefinite")
    return float(np.clip(np.vdot(psi, rho @ psi).real, 0.0, 1.0))


def fidelity_two_pure(ket1, ket2):
    """Squared fidelity |<psi|phi>|^2, invariant under global phase."""
    psi, phi = _normalized_ket(ket1), _normalized_ket(ket2)
    if psi.shape != phi.shape:
        raise ValueError("The two kets must have the same dimension")
    return float(np.clip(abs(np.vdot(psi, phi))**2, 0.0, 1.0))
