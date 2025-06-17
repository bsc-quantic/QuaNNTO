import numpy as np
import scipy.linalg as LA

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from functools import lru_cache, partial

def hermitian_matrix(m, N):
    '''
    Transforms the input matrix into a hermitian matrix.
    If the input matrix is real, the resultant matrix is real and symmetric.
    
    :param m: Matrix to be made hermitian
    :return: Hermitian matrix
    '''
    #return 0.5*(m + m.T)
    c = 0
    mat = np.zeros((N,N))
    for i in range(0, N):
        for j in range(i+1):
            mat[j,i] = mat[i,j] = m[c]
            c+=1
    return mat

@lru_cache(maxsize=None)
def _hermitian_index_pairs(N):
    # runs once per distinct N, on the host
    rows, cols = zip(*[(i, j) for i in range(N) for j in range(i+1, N)])
    return jnp.array(rows), jnp.array(cols)

@partial(jax.jit, static_argnums=(1,))
def general_hermitian_matrix(pars: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Builds an N×N Hermitian matrix from pars of length N^2:
      - the first offdiag_sz entries are the real parts of the i<j positions,
      - the next offdiag_sz entries are the imag parts,
      - the final N entries are the diagonal.
    """
    offdiag_sz = N*(N-1)//2

    # split off parameter vector
    offreal = pars[:offdiag_sz]
    offim   = pars[offdiag_sz:2*offdiag_sz]
    diag    = pars[2*offdiag_sz:2*offdiag_sz + N]

    # start from zeros
    mat = jnp.zeros((N, N), dtype=jnp.complex128)

    # set diagonal
    mat = mat.at[jnp.diag_indices(N)].set(diag)

    # fetch (i,j) pairs for i<j
    rows, cols = _hermitian_index_pairs(N)

    # set the upper triangle and its conjugate lower triangle
    mat = mat.at[rows, cols].set(offreal + 1j * offim)
    mat = mat.at[cols, rows].set(offreal - 1j * offim)

    return mat

def givens_rotation(N, i, j, theta, phi):
    """Construct a Givens rotation matrix."""
    G = np.eye(N, dtype=complex)
    G[i, i] = np.cos(theta) * np.exp(1j * phi)
    G[j, j] = np.cos(theta)
    G[i, j] = -np.sin(theta)
    G[j, i] = np.sin(theta) * np.exp(1j * phi)
    return G

def build_general_unitary(N, params):
    """Build an NxN unitary matrix using Givens rotations and phase shifts."""
    assert len(params) == N**2, f"Expected {N**2} parameters, got {len(params)}."
    U = np.eye(N, dtype=complex)
    idx = 0
    # Apply Givens rotations
    for i in range(N):
        for j in range(i + 1, N):
            theta = params[idx]
            phi = params[idx + 1]
            G = givens_rotation(N, i, j, theta, phi)
            U = G @ U  # Multiply on the left
            idx += 2
    # Apply phase shifts on the diagonal
    phases = np.exp(1j * params[idx:idx + N])
    U = U @ np.diag(phases)
    return U

@jax.jit
def unitary_from_hermitian(H: jnp.ndarray) -> jnp.ndarray:
    """
    Given a Hermitian matrix H, returns the unitary U = exp(i H).
    """
    return jsp.expm(1j * H)

def create_omega(N):
    '''
    Creates the omega matrix for N modes needed to verify the uncertainty principle 
    of the quantum system quadratures.
    Used when having an xpxp-ordering of the modes.

    :param N: Dimensions/modes of the quantum system
    :return: Omega matrix of size 2Nx2N
    '''
    omega = np.zeros((2*N,2*N), dtype='complex')
    for i in range(0, 2*N, 2):
        omega[i, i+1] = 1
        omega[i+1, i] = -1
    return omega

def create_J(N):
    '''
    Creates the J matrix for N modes needed to verify the uncertainty principle 
    of the quantum system quadratures. 
    Used when representing the quadratures as xxpp order.

    :param N: Dimensions/modes of the quantum system
    :return: J matrix of size 2Nx2N
    '''
    J = np.zeros((2*N, 2*N), dtype='complex')
    J[0:N, N:2*N] = np.eye(N)
    J[N:2*N, 0:N] = -1*np.eye(N)
    return J

class CanonicalLadderTransformations:
    """
    Convert between canonical (q,p) basis and ladder (a, a†) basis.
    
    to_ladder @ symplectic @ to_ladder†   ↔   unitary blocks for ladder ops
    """

    def __init__(self, dim: int):
        self.dim = dim
        N = dim

        # Build the 2N×2N to_ladder matrix:
        # [  I      i I ]
        # [  I   -i I ]  * (1/√2)
        TL = jnp.zeros((2*N, 2*N), dtype=jnp.complex128)
        I  = jnp.eye(N, dtype=jnp.complex128)

        # top-left block =  I
        TL = TL.at[:N,       :N      ].set(I)
        # top-right block =  i·I
        TL = TL.at[:N,       N:      ].set(1j * I)
        # bot-left block =  I
        TL = TL.at[N:,       :N      ].set(I)
        # bot-right block = -i·I
        TL = TL.at[N:,       N:      ].set(-1j * I)

        # scale by 1/√2
        self.to_ladder = TL * (2 ** -0.5)
        self.to_ladder_dagger = self.to_ladder.conj().T

    @staticmethod
    @jax.jit
    def build_ustar_u(unitary: jnp.ndarray) -> jnp.ndarray:
        """
        Given an N×N unitary U, returns the 2N×2N block
            [ U.conj(),    0  ]
            [    0   ,    U  ].
        """
        N = unitary.shape[0]
        M = jnp.zeros((2*N, 2*N), dtype=unitary.dtype)
        M = M.at[:N, :N].set(unitary.conj())
        M = M.at[N:, N:].set(unitary)
        return M

    @partial(jax.jit, static_argnums=0)
    def to_ladder_op(self, symplectic_transf: jnp.ndarray) -> jnp.ndarray:
        """
        Map a symplectic matrix S in (q,p) → the corresponding ladder‐basis unitary blocks:
            to_ladder @ S @ to_ladder_dagger
        """
        return self.to_ladder @ symplectic_transf @ self.to_ladder_dagger

    @partial(jax.jit, static_argnums=0)
    def to_canonical_op(self, unitary: jnp.ndarray) -> jnp.ndarray:
        """
        Map a ladder‐basis unitary U (on creation/annih operators) 
        back to its action on (q,p):
            to_ladder_dagger @ build_ustar_u(U) @ to_ladder
        """
        ustar_u = CanonicalLadderTransformations.build_ustar_u(unitary)
        return self.to_ladder_dagger @ ustar_u @ self.to_ladder

    
def check_symp_orth(SO):
    '''
    Verifies the symplectic and the orthogonal conditions on the input matrix.
    :param SO: Matrix to be verified
    '''
    N = int(len(SO)/2)
    X_prime = SO[0:N, 0:N]
    Y_prime = SO[0:N, N:]
    cond_1_prime = X_prime@Y_prime.T - Y_prime@X_prime.T
    print(f'Symplecticity condition: {cond_1_prime}')
    print(np.allclose(np.round(cond_1_prime, 4), np.zeros((N, N))))
    
    cond_2_prime = X_prime@X_prime.T + Y_prime@Y_prime.T
    print(f'Orthogonality condition: {cond_2_prime}')
    print(np.allclose(np.round(cond_2_prime, 4), np.eye(N)))
    print()
    
def check_det_and_positive(V):
    '''
    Checks if the input (covariance) matrix is positive and prints its determinant.
    
    :param V: (Covariance) matrix to be verified
    '''
    print('Covariance matrix determinant:')
    print(LA.det(V))
    cm_eigvals = LA.eigvals(V)
    print(f'V >= 0 \n{np.all(np.around(cm_eigvals, decimals=4) >= 0)}\n')
    
def check_uncertainty_pple(V):
    '''
    Checks if the input (covariance) matrix obeys the uncertainty principle printing 
    the eigenvalues of the uncertainty principle matrix.
    
    :param V: (Covariance) matrix to be verified
    '''
    i_J = 0.5j*create_J(int(len(V)/2)) # Normalized symplectic form in xxpp
    uncert_eigvals = LA.eigvals(V + i_J)
    print('V + iJ >= 0: ')
    print(np.all(np.around(uncert_eigvals, decimals=4) >= 0))
    print('Eigenvalues of V + iJ >= 0:')
    print(np.around(uncert_eigvals, decimals=4))
    
def symplectic_eigenvals(V):
    symp_mat = 1j*create_J(len(V) // 2)@V
    symp_eigvals = LA.eigvals(symp_mat)
    return symp_eigvals
    
def reconstruct_stats(exp_vals, N):
    aj = exp_vals[0 : N]
    
    aj_ak_idx = N
    ajdag_ak_idx = N+(N**2 + N)//2
    aj_ak = np.zeros((N, N), dtype='complex')
    ajdag_ak = np.zeros((N, N), dtype='complex')
    for j in range(N):
        for k in range(j,N):
            aj_ak[j, k] = aj_ak[k, j] = exp_vals[aj_ak_idx]
            ajdag_ak[j, k] = exp_vals[ajdag_ak_idx]
            ajdag_ak[k, j] = ajdag_ak[j, k].conjugate()
            aj_ak_idx += 1
            ajdag_ak_idx += 1

    xj = np.array([np.sqrt(2) * ak.real for ak in aj])
    pj = np.array([np.sqrt(2) * ak.imag for ak in aj])
    sj = np.concatenate((xj, pj))
    
    xjxk = np.array([[aj_ak[j,k].real + ajdag_ak[j,k].real + (0.5 if j==k else 0) for k in range(N)] for j in range(N)])
    pjpk = np.array([[-aj_ak[j,k].real + ajdag_ak[j,k].real + (0.5 if j==k else 0) for k in range(N)] for j in range(N)])
    #xjpk = np.array([[aj_ak[j,k].imag + ajdag_ak[j,k].imag + (0.5j if j==k else 0)*norm for k in range(N)] for j in range(N)])
    #pjxk = np.array([[aj_ak[j,k].imag - ajdag_ak[j,k].imag - (0.5j if j==k else 0)*norm for k in range(N)] for j in range(N)])
    xjpk_pkxj = np.array([[aj_ak[j,k].imag + ajdag_ak[j,k].imag for k in range(N)] for j in range(N)])
    #xjpk_pkxj = np.array([[2*aj_ak[j,k].imag for k in range(N)] for j in range(N)])
    sjsk = np.block([[xjxk, xjpk_pkxj],[xjpk_pkxj.T, pjpk]])
    
    V = np.array([[sjsk[j,k] - sj[j]*sj[k] for k in range(2*N)] for j in range(2*N)])
    return sj, V
