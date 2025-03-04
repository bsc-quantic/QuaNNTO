import numpy as np
from sympy import symbols, expand, MatrixSymbol, Add
from numba import njit, prange
import numba as nb
import time

from .utils import *

nb.set_num_threads(6)

def exp_val_ladder_jk(j, k, V, means, N):
    '''
    Computes the expectation value of two annihilation operators (in mode j and k) of a Gaussian state based 
    on its covariance matrix. This corresponds to the first identity I1.

    :param j: Mode of the first annihilation operator
    :param k: Mode of the second annihilation operator
    :param V: Covariance matrix of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of a pair of annihilation operators of a Gaussian state
    '''
    return 0.5*(V[j,k] + means[j]*means[k] - 
                (V[N+j, N+k] + means[N+j]*means[N+k]) + 
                1j*(V[j, N+k] + means[j]*means[N+k] + V[N+j, k] + means[N+j]*means[k]))

def exp_val_ladder_jdagger_k(j, k, V, means, N):
    '''
    Computes the expectation value of one annihilation and one creation operators (in mode j and k) 
    of a Gaussian state based on its covariance matrix. This corresponds to the second identity I2.

    :param j: Mode of the creation operator
    :param k: Mode of the annihilation operator
    :param V: Covariance matrix of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of one creation and one annihilation operators of a Gaussian state
    '''
    return 0.5*(V[j,k] + means[j]*means[k] + 
                V[N+j, N+k] + means[N+j]*means[N+k] + 
                1j*(V[j, N+k] + means[j]*means[N+k] - (V[N+j, k] + means[N+j]*means[k])) - 
                (1 if j==k else 0))

def exp_val_ladder_jdagger_kdagger(j, k, V, means, N):
    '''
    Computes the expectation value of two creation operators (in mode j and k) of a Gaussian state based 
    on its covariance matrix. This corresponds to the fourth identity I4.

    :param j: Mode of the first creation operator
    :param k: Mode of the second creation operator
    :param V: Covariance matrix of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of a pair of creation operators of a Gaussian state
    '''
    return 0.5*(V[j,k] + means[j]*means[k] - 
               (V[N+j, N+k] + means[N+j]*means[N+k]) - 
               1j*(V[j, N+k] + means[j]*means[N+k] + V[N+j, k] + means[N+j]*means[k]))

def compute_K_exp_vals(V, means):
    '''
    Computes the expectation value of all combinations of ladder operators pairs of all existing modes
    of a Gaussian state based on its covariance matrix V.
    
    :param V: Covariance matrix of the Gaussian state
    :return: All expectation values of the different configuration of a pair of ladder operators on all modes.
    '''
    N = len(V)//2
    K_exp_vals = np.zeros((4,N,N), dtype='complex')
    # Expectation values of two creation operators for all modes
    for j in range(N):
        for k in range(N):
            K_exp_vals[0,j,k] = exp_val_ladder_jk(j,k,V,means,N)
    # Expectation values of first creation and then annihilation operators for all modes
    for j in range(N):
        for k in range(N):
            K_exp_vals[1,j,k] = exp_val_ladder_jdagger_k(j,k,V,means,N)
    # Expectation values of first annihilation and then creation operators for all modes
    K_exp_vals[2] = np.copy(K_exp_vals[1])
    for j in range(N):
        K_exp_vals[2,j,j] += 1
    # Expectation values of two annihilation operators for all modes
    for j in range(N):
        for k in range(N):
            K_exp_vals[3,j,k] = exp_val_ladder_jdagger_kdagger(j,k,V,means,N)
    return K_exp_vals

def extract_ladder_expressions(trace_expr):
    '''
    Transforms a symbolic expression representing ladder operators acting on a Gaussian quantum state to
    numerical lists of the ladder modes (from 0 to N-1) and the operators type (creation or annihilation).

    :param trace_expr: Symbolic expression of creation and annihilation operators
    :return: Pair of lists containing the modes the ladder operators act onto and their type
    '''
    if len(trace_expr.args) > 0 and str(trace_expr.args[0]) == 'aux':
        trace_args = trace_expr.args[1:]
    else:
        trace_args = trace_expr.args
        
    ladder_modes = []
    ladder_types = []
    for sym_term in trace_args:
        term = str(sym_term).replace(" ","").replace("*","")
        tr_modes = []
        tr_types = []
        idx = 0
        # TODO: Optimize loop
        while idx < len(term):
            if term[idx]=='a':
                tr_types.append(0)
                tr_modes.append(int(term[idx+1]))
                idx += 2
                if idx < len(term) and term[idx].isdigit():
                    for i in range(int(term[idx]) - 1):
                        tr_types.append(tr_types[-1])
                        tr_modes.append(tr_modes[-1])
                    idx += 1
            elif term[idx]=='c':
                tr_types.append(1)
                tr_modes.append(int(term[idx+1]))
                idx += 2
                if idx < len(term) and term[idx].isdigit():
                    for i in range(int(term[idx]) - 1):
                        tr_types.append(tr_types[-1])
                        tr_modes.append(tr_modes[-1])
                    idx += 1
            else:
                idx += 1
        ladder_modes.append(tr_modes)
        ladder_types.append(tr_types)
    return ladder_modes, ladder_types

def complete_trace_expression(N, layers, photon_additions, n_outputs, include_obs=False, obs='position'):
    '''
    Builds the non-Gaussian state expression of a multi-photon added
    QNN based on superposition of ladder operators applied to the Gaussian state.

    :param N: Number of modes of the QNN
    :param layers: Number of layers of the QNN
    :return: Pair of lists containing the different expression terms of the modes the ladder operators act onto and their type
    '''
    dim = 2*N
    # Displacement vector (complex number 'r' and its conjugate 'i') for each mode
    d_r = symbols(f'r0:{layers*N}', commutative=True)
    d_i = symbols(f'i0:{layers*N}', commutative=True)
    # Symplectic matrix 2Nx2N
    S = MatrixSymbol('S', dim, layers*dim)
    # Creation (c) and annihilation (a) operators for each mode
    c = symbols(f'c0:{N}', commutative=False)
    a = symbols(f'a0:{N}', commutative=False)
    aux = symbols('aux')
    
    sup = 1
    sup_dag = 1
    for l in range(layers):
        for i in range(len(photon_additions)):
            # Displacement terms
            expr = d_r[l*N + photon_additions[i]]
            expr_dag = d_i[l*N + photon_additions[i]]
            for j in range(N):
                # Creation and annihilation terms with their symplectic coefficient
                expr += S[photon_additions[i], l*dim + j]*c[j]
                expr += S[photon_additions[i], l*dim + (N+j)]*a[j]
                expr_dag += S[photon_additions[i], l*dim + j]*a[j]
                expr_dag += S[photon_additions[i], l*dim + (N+j)]*c[j]
            sup *= expr
            sup_dag *= expr_dag

    if include_obs:
        expanded_expr = []
        for i in range(n_outputs):
            if obs == 'position':
                # Position operator
                expr = (1/np.sqrt(2))*(sup_dag*a[i]*sup + sup_dag*c[i]*sup)
            elif obs == 'momentum':
                # Momentum operator
                expr = (1j/np.sqrt(2))*(sup_dag*c[i]*sup - sup_dag*a[i]*sup)
            elif obs == 'number':
                # Number operator
                if len(photon_additions) == 0:
                    expr = sup_dag*c[i]*a[i]*sup + aux
                else:
                    expr = sup_dag*c[i]*a[i]*sup
            expanded_expr.append(expand(expr))
    else:
        expanded_expr = expand(sup_dag*sup)
    return expanded_expr

@njit
def ladder_exp_val(perf_matchings, ladder_modes, ladder_types, N, means_vector, cov_mat_identities):
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
    trace_sum = np.complex128(0)
    for perf_match in perf_matchings:
        trace_prod = np.complex128(1+0j)
        for i1,i2 in zip(perf_match[0::2], perf_match[1::2]):
            # Determine which identity to pick depending on the pair of ladder ops and the modes
            if i1 == -1:
                continue
            elif i1 != i2:
                trace_prod *= cov_mat_identities[ladder_types[i1] + 2*ladder_types[i2], ladder_modes[i1], ladder_modes[i2]]
            else:
                trace_prod *= single_ladder_exp_val(ladder_modes[i1], ladder_types[i2], N, means_vector)
        trace_sum += trace_prod
    return trace_sum

@njit
def single_ladder_exp_val(term_mode, term_type, N, means_vector):
    return (1/np.sqrt(2)) * (means_vector[term_mode] + 1j*(-2*term_type + 1)*means_vector[N+term_mode])

@njit
def get_expectation_value(term_modes, term_types, len_term, perf_matchs, N, K_exp_vals, means_vector):
    if len_term == 0: # CASE Tr[rho]
        return 1
    elif len_term == 1: # CASE Tr[a#rho]
        return single_ladder_exp_val(term_modes[0], term_types[0], N, means_vector)
    elif len_term == 2: # CASE Tr[a#a#rho]
        return K_exp_vals[term_types[0] + 2*term_types[1], term_modes[0], term_modes[1]]
    else:
        return ladder_exp_val(perf_matchs, term_modes, term_types, N, means_vector, K_exp_vals)

@njit(parallel=True)
def compute_terms_in_trace(terms, modes, types, terms_len, lpms, N, K_exp_vals, means_vector):
    sum_tr_values = 0.0 + 0.0*1j
    for i in prange(len(modes)):
        lpms_idx = terms_len[i] - 3 if terms_len[i] > 3 else 0
        sum_tr_values += terms[i] * get_expectation_value(modes[i], types[i], terms_len[i], lpms[lpms_idx][0], N, K_exp_vals, means_vector)
    return sum_tr_values

@njit
def compute_exp_val_loop(unnorm_terms, norm_terms, modes, types, unnorm_terms_len, modes_norm, types_norm, norm_terms_len, lpms, K_exp_vals, means_vector):
    N = len(means_vector) // 2
    # For unnormalized exp val
    unnorm = np.zeros((len(modes)), dtype='complex')
    for outs in prange(len(modes)):
        # TODO: Take care with this condition just made for 0 photon addition
        if len(modes[outs]) == 1:
            unnorm[outs] = compute_terms_in_trace([1], modes[outs], types[outs], unnorm_terms_len[outs], lpms, N, K_exp_vals, means_vector)
        else:
            unnorm[outs] = compute_terms_in_trace(unnorm_terms[outs], modes[outs], types[outs], unnorm_terms_len[outs], lpms, N, K_exp_vals, means_vector)
    
    # For normalization factor
    if len(modes_norm[0]) == 1 and modes_norm[0][0] == -1:
        norm = 1
    else:
        norm = compute_terms_in_trace(norm_terms, modes_norm[0], types_norm[0], norm_terms_len[0], lpms, N, K_exp_vals, means_vector)
    norm_val = unnorm/norm

    return norm_val

@njit
def compute_coefficients_njit(N, layers, n_out, D_concat, S_concat, nb_num_norm, nb_num_unnorm):
    # Build displacement complex vector & its conjugate
    d_r = np.zeros((layers * N), dtype='complex64')
    for l in range(layers):
        for i in range(N):
            d_r[l*N + i] = D_concat[l*2*N + i]+1j*D_concat[l*2*N + N+i]
    d_i = np.conjugate(d_r)
    
    # Values of normalization terms
    norm_vals = np.zeros((len(nb_num_norm)), dtype='complex64')
    for idx in range(len(nb_num_norm)):
        norm_vals[idx] = nb_num_norm[idx](S_concat, d_r, d_i)
    
    # Values of trace terms
    trace_vals = np.zeros((n_out, len(nb_num_unnorm[0])), dtype='complex64')
    for out_idx in range(n_out):
        for idx in range(len(nb_num_unnorm[0])):
            trace_vals[out_idx, idx] = nb_num_unnorm[out_idx][idx](S_concat, d_r, d_i)
    return trace_vals, norm_vals
    
def loop_perfect_matchings(N):
    # This function generates all perfect matchings of a complete graph with loops
    def backtrack(current_matching, remaining_nodes):
        if not remaining_nodes:
            matchings.append(current_matching)
            return
        node = remaining_nodes[0]
        for i in range(len(remaining_nodes)):
            pair = [node, remaining_nodes[i]]
            backtrack(current_matching + pair, remaining_nodes[1:i] + remaining_nodes[i+1:])
    nodes = list(range(N))
    matchings = []
    backtrack([], nodes)
    return matchings

def to_np_array(lists):
    lengths = np.full((len(lists), len(lists[0])), -1) if len(lists[0]) > 0 else np.array([[0]])
    for out_idx in range(len(lists)):
        for term_idx in range(len(lists[out_idx])):
            lengths[out_idx, term_idx] = len(lists[out_idx][term_idx])
    max_length = np.max(lengths)
    arr = np.full((len(lists), len(lists[0]), max_length), -1) if len(lists[0]) > 0 else np.array([[[-1]]])
    for out_idx in range(len(lists)):
        for term_idx in range(len(lists[out_idx])):
            arr[out_idx, term_idx, :len(lists[out_idx][term_idx])] = np.array(lists[out_idx][term_idx])
    return arr, lengths
    