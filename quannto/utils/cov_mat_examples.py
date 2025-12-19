import numpy as np
import scipy.linalg as LA

from .cvquantum_utils import *

# === COVARIANCE MATRICES OF GAUSSIAN QUANTUM STATES ===

def build_squeezed_state(modes):
    r = np.random.rand(modes)
    r_inv = 1.0/r
    z_matrix = np.zeros((2*modes,2*modes))
    np.fill_diagonal(z_matrix, np.concatenate((r, r_inv)))
    return z_matrix

def build_random_cv(modes):
    H = hermitian_matrix(np.random.rand(modes,modes))
    U = unitary_from_hermitian(H)
    
    u_bar = CanonicalLadderTransformations(modes)
    Q = u_bar.to_canonical_op(U)
    
    Z = build_squeezed_state(modes)
    
    S = Q@Z
    
    V = np.real_if_close(np.around(S@S.T, decimals = 5))
    print('\nCovariance matrix:')
    print(V)
    print('\nCovariance matrix determinant:')
    print(LA.det(V))
    return Q,Z,V

def two_mode_squeezed_state(r):
    # Define the matrices A, B, and C
    A = np.cosh(r) * np.identity(2)
    B = A
    C = np.sinh(r) * np.array([[0, 1], [1, 0]])

    # Construct the covariance matrix
    V = np.block([[A, C], [C.T, B]])
    return V

def maximally_entangled_two_mode():
    return np.array([[2.5, 0, 2, 0],
                  [0, 2.5, 0, -2],
                  [2, 0, 2, 0],
                  [0, -2, 0, 2]])

def maximally_entangled_two_mode_2():
    return np.array([[13, 0, 5, 0],
                  [0, 13, 0, -5],
                  [5, 0, 2, 0],
                  [0, -5, 0, 2]])

def cov_mat_1_mode_squeezed(th, r):
    return np.array([[r * np.cos(th)**2 + (1/r) * np.sin(th)**2, (r-(1/r)) * np.cos(th) * np.sin(th)],
                     [(r-(1/r)) * np.cos(th) * np.sin(th), (1/r) * np.cos(th)**2 + r * np.sin(th)**2]])
    
def analytic_energy_1_mode_squeezed(r):
    #return ((r-(1/r))**2)/16 + (((r+(1/r)) - 2)**2)/16
    return 0.25*((r-1/r)**2/(r+1/r-2)) + 0.5*(r+1/r-2)

def cov_mat_gaussian_mixed_state():
    return np.array([[2.0, 0.5, 0.3, 0.1],
                    [0.5, 1.5, 0.2, 0.4],
                    [0.3, 0.2, 1.2, 0.6],
                    [0.1, 0.4, 0.6, 1.8]])

# === COVARIANCE MATRIX OF A NON-GAUSSIAN QUANTUM STATE ===
def non_gaussian_cov_mat():
    return np.array([[1.0, 0.6, 0.4, 0.2],
                    [0.6, 2.0, 0.3, 0.7],
                    [0.4, 0.3, 0.8, 0.5],
                    [0.2, 0.7, 0.5, 1.5]])