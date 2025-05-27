import numpy as np
import scipy.linalg as LA

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

def general_hermitian_matrix(pars, N):
    '''
    Builds a general NxN complex hermitian matrix based on 
    the input N² parameters.
    
    :param pars: Vector of N² parameters
    :param N: Matrix dimension
    :return: Hermitian matrix
    '''
    c = 0
    offdiag_size = int((N-1)*N/2)
    offreal = pars[:offdiag_size]
    offim = pars[offdiag_size : 2*offdiag_size]
    diag = pars[2*offdiag_size : 2*offdiag_size + N]
    mat = np.zeros((N,N), dtype='complex')
    for i in range(0,N):
        for j in range(i,N):
            if i==j:
                mat[i,i] = diag[i]
            else:
                mat[i,j] = offreal[c] + 1j*offim[c]
                mat[j,i] = offreal[c] - 1j*offim[c]
                c += 1
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

def unitary_from_hermitian(H):
    '''
    Creates a unitary matrix from a hermitian one by complex exponentiation.

    :param H: Hermitian matrix
    :return: Unitary matrix of the input matrix
    '''
    unitary = LA.expm(1j*H)
    return unitary

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
    '''
    This class contains the methods to go from/to canonical basis to/from Fock basis.
    '''
    def __init__(self, dim):
        self.to_ladder = np.eye(2*dim, dtype='complex')
        self.to_ladder[0:dim, dim:] = 1j * np.eye(dim)
        self.to_ladder[dim:, 0:dim] = np.eye(dim)
        self.to_ladder[dim:, dim:] *= -1j
        self.to_ladder *= 2**(-0.5)
        self.to_ladder_dagger = self.to_ladder.conj().T
        
    def build_ustar_u(self, unitary):
        dim = len(unitary)
        ustar_u = np.zeros((2*dim, 2*dim), dtype='complex')
        ustar_u[0:dim, 0:dim] = unitary.conj()
        ustar_u[dim:, dim:] = unitary
        return ustar_u
        
    def to_ladder_op(self, symplectic_transf):
        return self.to_ladder @ symplectic_transf @ self.to_ladder_dagger
    
    def to_canonical_op(self, unitary):
        ustar_u = self.build_ustar_u(unitary)
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
    print('Eigenvalues of iJV: ')
    print(symp_eigvals)
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
    
    print("Xj")
    print(xj)
    print("Pj")
    print(pj)
    print("XjXk")
    print(xjxk)
    
    V = np.array([[sjsk[j,k] - sj[j]*sj[k] for k in range(2*N)] for j in range(2*N)])
    return sj, V
