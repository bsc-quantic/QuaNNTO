"""Independent finite-Fock propagation of QuaNNTO's parameterized gates.

Uses matrix exponentials of physical generators, not Gaussian Fock coefficients,
symplectic decompositions or transformed subtraction polynomials. Intermediate
cutoffs must converge; this is a verification backend, not the training engine.
"""
import numpy as np
from scipy.linalg import expm


def parameter_fock_ket(parameters, N, subtractions, alpha, cutoff):
    """Return an unnormalized ket from sequential gates and annihilations.

    Layout: reversed physical-layer blocks Q1,Q2,|r|,Re(beta),Im(beta).
    The passive physical matrix is U.conj() in QuaNNTO's conventions.
    """
    L = len(subtractions)
    p = np.asarray(parameters, dtype=float)
    if p.shape != (L*(2*N*N+3*N),) or not np.isfinite(p).all():
        raise ValueError('Invalid parameter vector')
    if cutoff < 2 or any(not 0 <= m < N for layer in subtractions for m in layer):
        raise ValueError('Invalid cutoff or subtraction mode')
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    adag = a.T
    ket = np.zeros((cutoff,)*N, complex)
    ket[(0,)*N] = 1

    def local(matrix, mode):
        nonlocal ket
        ket = np.moveaxis(np.tensordot(matrix, ket, axes=(1, mode)), 0, mode)

    def phase(value, mode):
        shape = [1]*N
        shape[mode] = cutoff
        ket[:] *= np.exp(-1j*value*np.arange(cutoff)).reshape(shape)

    def beamsplitter(theta, i, j):
        nonlocal ket
        # K = a_j^dagger a_i - a_i^dagger a_j conserves total occupation.
        moved = np.moveaxis(ket, (i, j), (0, 1))
        flat = moved.reshape(cutoff, cutoff, -1)
        result = np.empty_like(flat)
        for total in range(2*cutoff-1):
            ni = np.arange(max(0, total-cutoff+1), min(cutoff-1, total)+1)
            nj = total-ni
            up = -np.sqrt((ni[:-1]+1)*(total-ni[:-1]))
            generator = np.diag(up, -1)-np.diag(up, 1)
            result[ni, nj] = expm(theta*generator) @ flat[ni, nj]
        ket = np.moveaxis(result.reshape(moved.shape), (0, 1), (i, j))

    def passive(coords):
        pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
        for i, value in enumerate(coords[2*len(pairs):]):
            phase(value, i)
        for index, (i,j) in enumerate(pairs):
            theta, phi = coords[2*index:2*index+2]
            phase(phi, i)
            beamsplitter(theta, i, j)

    local(expm(complex(alpha)*adag-complex(alpha).conjugate()*a), 0)
    width = 2*N*N+3*N
    for ell in range(L):
        block = p[(L-1-ell)*width:(L-ell)*width]
        passive(block[:N*N])
        for i,r in enumerate(abs(block[2*N*N:2*N*N+N])):
            local(expm(.5*r*(adag@adag-a@a)), i)
        passive(block[N*N:2*N*N])
        beta = block[2*N*N+N:2*N*N+2*N]+1j*block[2*N*N+2*N:]
        for i,value in enumerate(beta):
            local(expm(value*adag-value.conjugate()*a), i)
        for i in subtractions[ell]:
            local(a, i)
    return ket
