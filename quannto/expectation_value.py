import numpy as np
from sympy import symbols, expand
from numba import njit, prange

from .utils import *

def exp_val_ladder_jk(j, k, V, N):
    '''
    Computes the expectation value of two annihilation operators (in mode j and k) of a Gaussian state based 
    on its covariance matrix. This corresponds to the first identity I1.

    :param j: Mode of the first annihilation operator
    :param k: Mode of the second annihilation operator
    :param V: Covariance matrix of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of a pair of annihilation operators of a Gaussian state
    '''
    return 0.25*(V[j,k] - V[N+j, N+k] + 1j*(V[j, N+k] + V[N+j, k]))

def exp_val_ladder_jdagger_k(j, k, V, N):
    '''
    Computes the expectation value of one annihilation and one creation operators (in mode j and k) 
    of a Gaussian state based on its covariance matrix. This corresponds to the second identity I2.

    :param j: Mode of the creation operator
    :param k: Mode of the annihilation operator
    :param V: Covariance matrix of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of one creation and one annihilation operators of a Gaussian state
    '''
    return 0.25*(V[j,k] + V[N+j, N+k] + 1j*(V[j, N+k] - V[N+j, k]) - 2*(1 if j==k else 0))

def exp_val_ladder_jdagger_kdagger(j, k, V, N):
    '''
    Computes the expectation value of two creation operators (in mode j and k) of a Gaussian state based 
    on its covariance matrix. This corresponds to the fourth identity I4.

    :param j: Mode of the first creation operator
    :param k: Mode of the second creation operator
    :param V: Covariance matrix of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of a pair of creation operators of a Gaussian state
    '''
    return 0.25*(V[j,k] - V[N+j, N+k] - 1j*(V[j, N+k] + V[N+j, k]))

def compute_K_exp_vals(V):
    '''
    Computes the expectation value of all combinations of ladder operators pairs of all existing modes
    of a Gaussian state based on its covariance matrix V.
    
    :param V: Covariance matrix of the Gaussian state
    :return: All expectation values of the different configuration of a pair of ladder operators on all modes.
    '''
    N = int(len(V)/2)
    K_exp_vals = np.zeros((4,N,N), dtype='complex')

    # Expectation values of two creation operators for all modes
    for j in range(N):
        for k in range(N):
            K_exp_vals[0,j,k] = exp_val_ladder_jk(j,k,V,N)

    # Expectation values of first creation and then annihilation operators for all modes
    for j in range(N):
        for k in range(N):
            K_exp_vals[1,j,k] = exp_val_ladder_jdagger_k(j,k,V,N)

    # Expectation values of first annihilation and then creation operators for all modes
    K_exp_vals[2] = np.copy(K_exp_vals[1])
    for j in range(N):
        K_exp_vals[2,j,j] += 1

    # Expectation values of two annihilation operators for all modes
    for j in range(N):
        for k in range(N):
            K_exp_vals[3,j,k] = exp_val_ladder_jdagger_kdagger(j,k,V,N)
            
    return K_exp_vals

def to_ladder_expression(sym_exp):
    '''
    Transforms a symbolic expression representing ladder operators acting on a Gaussian quantum state to
    numerical lists of the ladder modes (from 0 to N-1) and the operators type (creation or annihilation).

    :param sym_exp: Symbolic expresion of creation and annihilation operators
    :return: Pair of lists containing the modes the ladder operators act onto and their type
    '''
    ladder_modes = []
    ladder_types = []
    idx = 0
    while idx < len(sym_exp):
        if sym_exp[idx]=='a':
            ladder_types.append(0)
            ladder_modes.append(int(sym_exp[idx+1]))
            idx += 2
        elif sym_exp[idx]=='c':
            ladder_types.append(1)
            ladder_modes.append(int(sym_exp[idx+1]))
            idx += 2
        else:
            for i in range(int(sym_exp[idx]) - 1):
                ladder_types.append(ladder_types[-1])
                ladder_modes.append(ladder_modes[-1])
            idx += 1
            
    return ladder_modes, ladder_types

def ladder_ops_trace_expression(N, layers):
    '''
    Builds the non-Gaussian state expression of a multilayer QNN based on superposition of ladder operators 
    applied to the Gaussian state.

    :param N: Number of modes of the QNN
    :param layers: Number of layers of the QNN
    :return: Pair of lists containing the different expression terms of the modes the ladder operators act onto and their type
    '''
    dim = 2*N
    
    # 1. Create creation (c) and annihilation (a) operators for all modes
    c = symbols(f'c0:{N}', commutative=False)
    a = symbols(f'a0:{N}', commutative=False)
    lad = c + a

    # 2. Build the trace expressions with the ladder operators
    #    2.1. Ladder operators superposition
    S = []
    S_dag = []
    for i in range(layers):
        tr1 = 0
        tr1_dag = 0
        for j in range(2*N):
            tr1 += lad[j]
            tr1_dag += lad[(j+2)%dim]
        S.append(tr1)
        S_dag.append(tr1_dag)

    #    2.2. Last ladder operator corresponding to the last layer
    complete_trace = 1
    for i in range(layers):
        complete_trace = S_dag[layers - 1 - i]*complete_trace*S[layers - 1 - i]

    #    2.3. Arrange the format of the complete expectation value (trace) expression
    complete_trace = expand(complete_trace)
    str_expr = str(complete_trace)
    trace_expressions = str_expr.replace(" ","").replace("*","").split('+')

    # 3. Transform the symbolic trace expression to numerical encoding of the ladder modes and types
    ladder_modes = []
    ladder_types = []
    for tr_expr in trace_expressions:
        tr_modes, tr_types = to_ladder_expression(tr_expr)
        ladder_modes.append(tr_modes)
        ladder_types.append(tr_types)

    return [ladder_modes], [ladder_types]

def include_observable(ladder_modes, ladder_types, observable_modes, observable_types):
    '''
    Adds the ladder operators corresponding to the observable to be measured at the end of the QNN
    to the non-Gaussian state expression.

    :param ladder_modes: List containing the modes of all non-Gaussian terms of the non-Gaussian state expression
    :param ladder_types: List containing the ladder operator types of all non-Gaussian terms of the non-Gaussian state expression
    :param observable_modes: List containing the target modes of the observable(s) to be added
    :param observable_modes: List containing the ladder operator types of the observable(s) to be added
    :return: Pair of lists containing the different terms of the final expression whose expectation value is to be computed
    '''
    obs_ladder_modes = []
    obs_ladder_types = []
    for i in range(len(observable_modes)):
        obs_ladder_modes += [ladder_modes.copy()]
        obs_ladder_types += [ladder_types.copy()]
        
    mid_expr = int(len(ladder_modes[0])/2)
    for i in range(len(observable_modes)):
        for j in range(len(obs_ladder_modes[i])):
            obs_ladder_modes[i][j] = obs_ladder_modes[i][j][:mid_expr] + observable_modes[i] + obs_ladder_modes[i][j][mid_expr:]
            obs_ladder_types[i][j] = obs_ladder_types[i][j][:mid_expr] + observable_types[i] + obs_ladder_types[i][j][mid_expr:]
            
    return obs_ladder_modes, obs_ladder_types

def perfect_matchings(num_ladder_operators):
    ''' 
    Finds all existing perfect matchings in a list of an even number of nodes
    referred to the even number of ladder operators indices applied to a Gaussian state.
    
    :param num_ladder_operators: EVEN number of ladder operators
    :return: List of lists containing all possible perfect matchings of the operators
    '''
    perf_matchings = []
    find_perf_match([i for i in range(num_ladder_operators)], [], perf_matchings)
    return perf_matchings

def find_perf_match(index_list, current_combination, perf_matchings):
    ''' 
    Auxiliary recursive function of perfect_matchings(num_ladder_operators) that creates 
    all existing perfect matchings given an index list and stores them in 
    perf_matchings parameter.
    
    :param index_list: Number of existing indices (or nodes in a complete graph)
    :param current_combination: The perfect matching combination being filled at the moment
    :param perf_matchings: List of lists that will store all perfect matchings at the end of the recursive calls
    '''
    if len(index_list) > 0:
        v1 = index_list.pop(0)
        current_combination.append(v1)
        for i in range(len(index_list)):
            new_combination = current_combination.copy()
            new_idx_list = index_list.copy()
            v2 = new_idx_list.pop(i)
            new_combination.append(v2)
            find_perf_match(new_idx_list, new_combination, perf_matchings)
    else:
        perf_matchings.append(current_combination)
        
@njit
def ladder_exp_val(perf_matchings, ladder_modes, ladder_types, cov_mat_identities):
    '''
    Computes the expected value of the energy when ladder operators are applied
    to a Gaussian state described by its covariance matrix. All perfect matchings
    of the ladder operators sequence are needed.
    
    :param perf_matchings: List containing the lists of all perfect matchings
    :param ladder_modes: List of modes of the expectation value equation (trace of ladder operators on a Gaussian state)
    :param ladder_types: List defining the type of each ladder operator of the expectation value equation matching the modes
    :param cov_mat_identities: All possible expectation values of pairs of dagger operators acting on a Gaussian state
    :return: Corresponding expectation value of the energy
    '''
    energy = np.complex128(0)
    for perf_match in perf_matchings:
        trace_prod = np.complex128(1+0j)
        for i1,i2 in zip(perf_match[0::2], perf_match[1::2]):
            # Determine which identity to pick depending on the pair of ladder ops and the modes
            trace_prod *= cov_mat_identities[ladder_types[i1] + 2*ladder_types[i2], ladder_modes[i1], ladder_modes[i2]]
        energy += trace_prod
    return energy

def symplectic_from_svd(N, Z_params, Q_params_1, Q_params_2):
    '''
    DEPRECATED: Creates a 2Nx2N symplectic matrix from SVD form using the parameters for two symplectic-orthogonal 
    matrices and one diagonal matrix.

    :param N: Half dimension (number of modes) of the symplectic matrix
    :param Z_params: Values of the diagonal matrix
    :param Q_params_1: Values with which the first symplectic-orthogonal matrix is created
    :param Q_params_2: Values with which the second symplectic-orthogonal matrix is created
    :return: Symplectic matrix of the (SVD) form Q2*Z*Q1
    '''
    r = Z_params
    r_inv = 1.0/r
    Z = np.diag(np.concatenate((r, r_inv)))

    u_bar = CanonicalLadderTransformations(N)
    
    H = hermitian_matrix(Q_params_1.reshape((N, N)))
    U = unitary_from_hermitian(H)
    Q1 = u_bar.to_canonical_op(U)
    
    H = hermitian_matrix(Q_params_2.reshape((N, N)))
    U = unitary_from_hermitian(H)
    Q2 = u_bar.to_canonical_op(U)
    
    return Q2@Z@Q1


@njit
def get_symplectic_coefs(N, S, ladder_modes, ladder_types):
    '''
    Computes the coefficient of each term contained in the non-Gaussian state expression given 
    the symplectic matrices of the ladder superposition operators.

    :param N: Number of modes of the system
    :param S: Symplectic matrices of the ladder superposition operators.
    :param ladder_modes: List of terms of the non-Gaussian state expression containing the modes they act onto.
    :param ladder_types: List of terms of the non-Gaussian state expression containing the ladder operator types.
    :return: Symplectic coefficients of each term in the final non-Gaussian state expression.
    '''
    symp_coefs = np.ones((len(ladder_modes), len(ladder_modes[0])))
    if len(S)==0:
        return symp_coefs
    for i in prange(len(ladder_modes)):
        for j in prange(len(ladder_modes[i])):
            middle = int(len(ladder_modes[i][j])/2)
            for k in prange(middle):
                # FOR PHOTON ADDITION ON MODE 1
                symp_coefs[i,j] *= S[k, 0, ladder_modes[i][j][k]+N*ladder_types[i][j][k]]
                symp_coefs[i,j] *= S[k, 0, ladder_modes[i][j][len(ladder_modes[i][j])-k-1] + N*(1 - ladder_types[i][j][len(ladder_modes[i][j])-k-1])]
                # FOR PHOTON SUBTRACTION ON MODE 1
                #symp_coefs[i,j] *= S[k, 0, ladder_modes[i][j][k]+N*(1-ladder_types[i][j][k])]
                #symp_coefs[i,j] *= S[k, 0, ladder_modes[i][j][len(ladder_modes[i][j])-k-1] + N*(ladder_types[i][j][len(ladder_modes[i][j])-k-1])]
            
    return symp_coefs

