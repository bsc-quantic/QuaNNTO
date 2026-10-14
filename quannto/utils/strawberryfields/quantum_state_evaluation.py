"""Fock evaluation of saved Gaussian layers and ideal photon subtractions.

No QNN construction, optimizer, random initialization or model deserialization.
The Gaussian Fock coefficients use The Walrus, the converter used by Strawberry
Fields. Earlier subtractions are conjugated through the later Gaussian layers.
This avoids truncating a sequence of intermediate Gaussian gates. A margin equal
to the polynomial degree makes the final projection exact at the requested cutoff
(apart from floating-point arithmetic); convergence to the full state is separate.
"""
from functools import lru_cache

import numpy as np

from .general_tools import coherent_ket, cat_even_ket


CAT_WORDS = ("a", "aa", "ca", "aaa", "aac", "caca", "aaaa", "aaac",
             "aaaaa", "aaaac", "aaacc", "cacaca", "aaaaaa", "aaaaac", "aaaacc")
CUBIC_WORDS = ("a", "aa", "aaa", "cca", "ca", "caca")


def validate_layers(S, d, subtractions, atol=1e-7):
    """Check xxpp symplectic matrices, coherent amplitudes and 0-based modes."""
    S, d = np.asarray(S), np.asarray(d)
    if S.ndim != 3 or S.shape[1] != S.shape[2] or S.shape[1] % 2:
        raise ValueError("S_l must have shape (L, 2N, 2N)")
    L, dim, _ = S.shape
    if L < 1 or dim < 2 or d.shape != (L, dim):
        raise ValueError("D_l must have shape (L, 2N), with N,L >= 1")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(d)):
        raise ValueError("Saved operators contain non-finite values")
    if np.max(np.abs(S.imag)) > atol or np.max(np.abs(d.imag)) > atol:
        raise ValueError("S_l and D_l must be real in the xxpp convention")
    S, d = S.real.astype(float), d.real.astype(float)
    N = dim // 2
    omega = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    if any(not np.allclose(s @ omega @ s.T, omega, atol=atol, rtol=0) for s in S):
        raise ValueError("A saved S_l is not symplectic in xxpp ordering")
    if len(subtractions) != L:
        raise ValueError("One subtraction list is required per layer")
    for layer in subtractions:
        for mode in layer:
            if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)) or not 0 <= mode < N:
                raise ValueError("Subtraction modes must be 0-based integers in [0,N)")
    return S, d


def operators_from_parameters(parameters, N, L):
    """Decode the repo's active-layer parameter layout, including reversed layers.

    Only the manuscript architecture is supported: no initial squeezing/mixing,
    two Givens interferometers, nonnegative squeezing magnitudes and displacement.
    """
    p = np.asarray(parameters)
    if p.ndim != 1 or p.size != L * (2*N*N + 3*N):
        raise ValueError("Parameter count does not match the manuscript architecture")
    if not np.all(np.isfinite(p)) or np.max(np.abs(p.imag)) > 1e-12:
        raise ValueError("Parameters must be finite and real")
    p = p.real

    def passive(coords):
        U = np.eye(N, dtype=complex)
        k = 0
        for i in range(N):
            for j in range(i+1, N):
                theta, phi = coords[k:k+2]
                k += 2
                G = np.eye(N, dtype=complex)
                G[i, i] = np.cos(theta) * np.exp(1j*phi)
                G[j, j] = np.cos(theta)
                G[i, j] = -np.sin(theta)
                G[j, i] = np.sin(theta) * np.exp(1j*phi)
                U = G @ U
        U = U @ np.diag(np.exp(1j*coords[k:k+N]))
        # QuaNNTO uses diag(U.conj(), U) in its ladder representation.
        return np.block([[U.real, U.imag], [-U.imag, U.real]])

    S, d = np.empty((L, 2*N, 2*N)), np.empty((L, 2*N))
    k = 0
    for layer in reversed(range(L)):
        Q1 = passive(p[k:k+N*N]); k += N*N
        Q2 = passive(p[k:k+N*N]); k += N*N
        r = np.abs(p[k:k+N]); k += N
        S[layer] = Q2 @ np.diag(np.exp(np.r_[r, -r])) @ Q1
        d[layer] = p[k:k+2*N]; k += 2*N
    return validate_layers(S, d, [[] for _ in range(L)])


def _linear_action(ket, annihilation, creation, constant):
    """Apply c + sum(u_m a_m + v_m a_m^dagger) without dense operators."""
    out = constant * ket
    D = ket.shape[0]
    for mode, (u, v) in enumerate(zip(annihilation, creation)):
        lo, hi = [slice(None)] * ket.ndim, [slice(None)] * ket.ndim
        lo[mode], hi[mode] = slice(0, D-1), slice(1, D)
        shape = [1] * ket.ndim
        shape[mode] = D-1
        weights = np.sqrt(np.arange(1, D)).reshape(shape)
        out[tuple(lo)] += u * weights * ket[tuple(hi)]
        out[tuple(hi)] += v * weights * ket[tuple(lo)]
    return out


def projected_qonn_ket(S, d, subtractions, alpha, cutoff, max_tensor_bytes=512*1024**2):
    """Return the unnormalized projection of A_L G_L ... A_1 G_1 |alpha,0,...>.

    The squared norm is an ideal-operator weight, not a physical click probability.
    All Gaussian quantities below use hbar=1; coherent amplitudes do not depend
    on hbar. Memory guard estimates six working tensors, excluding JIT overhead.
    """
    S, d = validate_layers(S, d, subtractions)
    N = S.shape[1] // 2
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, np.integer)) or cutoff < 2:
        raise ValueError("cutoff must be an integer >= 2")
    if not np.isfinite(complex(alpha)):
        raise ValueError("Input alpha must be finite")
    degree = sum(map(len, subtractions))
    work_cutoff = cutoff + degree
    estimate = 6 * 16 * work_cutoff**N
    if estimate > max_tensor_bytes:
        raise MemoryError(f"Estimated Fock workspace {estimate/1024**2:.1f} MiB exceeds "
                          f"limit {max_tensor_bytes/1024**2:.1f} MiB; reduce cutoff or raise --max-memory-mib")
    I = np.eye(N)
    T = np.block([[I, 1j*I], [I, -1j*I]]) / np.sqrt(2)
    suffix_S, suffix_d = np.eye(2*N), np.zeros(2*N)
    transformed = [None] * len(S)
    for ell in reversed(range(len(S))):
        B_inverse = np.linalg.inv(T @ suffix_S @ T.conj().T)
        beta = suffix_d[:N] + 1j*suffix_d[N:]
        constant = -B_inverse @ np.r_[beta, beta.conj()]
        transformed[ell] = [(B_inverse[m, :N], B_inverse[m, N:], constant[m])
                            for m in subtractions[ell]]
        suffix_d = suffix_S @ d[ell] + suffix_d
        suffix_S = suffix_S @ S[ell]
    inputs = np.zeros(2*N)
    inputs[0], inputs[N] = complex(alpha).real, complex(alpha).imag
    means = np.sqrt(2) * (suffix_S @ inputs + suffix_d)
    cov = 0.5 * suffix_S @ suffix_S.T
    from thewalrus.quantum import state_vector
    ket = state_vector(means, cov, cutoff=work_cutoff, hbar=1, normalize=False)
    # Rightmost transformed subtraction acts first. Preserve within-mode order.
    for layer in transformed:
        for u, v, c in layer:
            ket = _linear_action(ket, u, v, c)
    ket = np.ascontiguousarray(ket[(slice(0, cutoff),) * N])
    weight = float(np.vdot(ket, ket).real)
    if not np.all(np.isfinite(ket)) or not np.isfinite(weight) or weight <= 0:
        raise ValueError("Circuit produces a zero or non-finite conditional state")
    return ket, weight


def reduced_output(ket, mode=0):
    """Normalized reduced density matrix of the observed mode."""
    ket = np.asarray(ket, dtype=complex)
    if not 0 <= mode < ket.ndim:
        raise ValueError("Invalid output mode")
    view = np.moveaxis(ket, mode, 0).reshape(ket.shape[mode], -1)
    norm = np.vdot(view, view).real
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("State has zero or invalid norm")
    return (view @ view.conj().T) / norm


@lru_cache(maxsize=16)
def moment_operators(task, cutoff):
    """P_D O P_D for the unbounded ladder words, in the training order.

    Building words above the output cutoff prevents artificial top-level
    commutator errors (e.g. a a^dagger must equal n+1 even on |D-1>).
    """
    if task not in ("cat", "cubic"):
        raise ValueError("task must be cat or cubic")
    words = CAT_WORDS if task == "cat" else CUBIC_WORDS
    padded = cutoff + max(map(len, words))
    a = np.diag(np.sqrt(np.arange(1, padded)), 1)
    operators = []
    for word in words:
        op = np.eye(padded)
        for letter in word:
            op = op @ (a if letter == "a" else a.T)
        projected = op[:cutoff, :cutoff].copy()
        projected.setflags(write=False)
        operators.append(projected)
    return tuple(operators)


def output_moments(rho, task):
    return np.array([np.einsum("ij,ji->", rho, o) for o in moment_operators(task, len(rho))])


def reference_ket(task, alpha, cutoff, gamma=0.2, hbar=2.0, cat_alpha=1.0, phi=0.0):
    """Unnormalized reference projection; cubic uses SF's actual Vgate matrix."""
    if task == "cat":
        return cat_even_ket(cat_alpha, cutoff, phi)
    if task != "cubic":
        raise ValueError("task must be cat or cubic")
    if not np.isfinite(gamma) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("gamma must be finite and hbar positive")
    # SF 0.23: V = exp(i gamma x^3 / (3 hbar)); this is the same cached matrix
    # called by the Fock backend's public Vgate, with hbar passed explicitly.
    from quannto.utils.strawberryfields.compat import import_strawberryfields
    import_strawberryfields()
    from strawberryfields.backends.fockbackend.ops import cubicPhase
    return cubicPhase(gamma, hbar, cutoff) @ coherent_ket(alpha, cutoff)


def output_fidelity(rho, reference):
    """Squared fidelity with a larger pure target, without renormalizing its slice.

    The output rho lives in a smaller Fock subspace embedded in the reference
    space. Renormalizing reference[:D] would artificially increase fidelity.
    """
    reference = np.asarray(reference, dtype=complex).reshape(-1)
    D = len(rho)
    if len(reference) < D:
        raise ValueError("Reference cutoff must be at least the output cutoff")
    norm = np.vdot(reference, reference).real
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Target has zero or invalid norm")
    target = reference[:D] / np.sqrt(norm)
    f = np.vdot(target, rho @ target)
    if abs(f.imag) > 1e-10 or not -1e-10 <= f.real <= 1+1e-10:
        raise ValueError("Invalid fidelity; check density matrix normalization")
    return float(np.clip(f.real, 0, 1))
