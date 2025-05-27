import numpy as np
from sympy import symbols, expand, MatrixSymbol, Add
from numba import njit, prange
import numba as nb

from .utils import *

nb.set_num_threads(6)

def exp_val_ladder_jk(j, k, V, means, N):
    '''
    Computes the expectation value of two annihilation operators (in mode j and k) of a Gaussian state based 
    on the first two statistical moments of the state. This corresponds to the first identity I1.

    :param j: Mode of the first annihilation operator
    :param k: Mode of the second annihilation operator
    :param V: Covariance matrix of the Gaussian state
    :param means: Means vector of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of a pair of annihilation operators of a Gaussian state
    '''
    return 0.5*(V[j,k] - V[N+j, N+k] + 1j*(V[j, N+k] + V[N+j, k]))

def exp_val_ladder_jdagger_k(j, k, V, means, N):
    '''
    Computes the expectation value of one annihilation and one creation operators (in mode j and k) 
    of a Gaussian state based on the first two statistical moments of the state.
    This corresponds to the second identity I2.

    :param j: Mode of the creation operator
    :param k: Mode of the annihilation operator
    :param V: Covariance matrix of the Gaussian state
    :param means: Means vector of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of one creation and one annihilation operators of a Gaussian state
    '''
    return 0.5*(V[j,k] + V[N+j, N+k] + 1j*(V[j, N+k] - V[N+j, k]) - (1 if j==k else 0))

def exp_val_ladder_jdagger_kdagger(j, k, V, means, N):
    '''
    Computes the expectation value of two creation operators (in mode j and k) of a Gaussian state based 
    on the first two statistical moments of the state. This corresponds to the fourth identity I4.

    :param j: Mode of the first creation operator
    :param k: Mode of the second creation operator
    :param V: Covariance matrix of the Gaussian state
    :param means: Means vector of the Gaussian state
    :param N: Number of modes of the quantum system
    :return: Expectation value of a pair of creation operators of a Gaussian state
    '''
    return 0.5*(V[j,k] - V[N+j, N+k] - 1j*(V[j, N+k] + V[N+j, k]))

def compute_K_exp_vals(V, means):
    '''
    Computes the expectation value of all combinations of ladder operators pairs of all existing modes
    of a Gaussian state based on its covariance matrix and means vector.
    
    :param V: Covariance matrix of the Gaussian state
    :param means: Means vector of the Gaussian state
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
    K_exp_vals[2] = np.copy(K_exp_vals[1].T)
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
    Builds the non-Gaussian state symbolic expression of a multi-photon added
    QNN based on superposition of ladder operators applied to the Gaussian state.

    :param N: Number of modes of the QNN
    :param layers: Number of layers of the QNN
    :param photon_additions: Photon additions made over the modes at each layer
    :param n_outputs: Number of QNN outputs
    :param include_obs: Whether to build the trace expression with or without the observable (default=False)
    :param obs: Observable to be measured (QNN output)
    :return: Symbolic expressions of the different QNN outputs (or normalization factor)
    '''
    dim = 2*N
    # Displacement vector (complex number 'r' and its conjugate 'i') for each mode
    d_r = symbols(f'r0:{layers*N}', commutative=True)
    d_i = symbols(f'i0:{layers*N}', commutative=True)
    # Symplectic matrix 2Nx2N
    S_r = MatrixSymbol('S_r', dim, layers*dim)
    S_i = MatrixSymbol('S_i', dim, layers*dim)
    # Creation (c) and annihilation (a) operators for each mode
    c = symbols(f'c0:{N}', commutative=False)
    a = symbols(f'a0:{N}', commutative=False)
    aux = symbols('aux') if len(photon_additions) == 0 else 0
    
    sup = 1
    sup_dag = 1
    for l in range(layers):
        for i in range(len(photon_additions)):
            # Displacement terms
            expr = d_i[l*N + photon_additions[i]]
            expr_dag = d_r[l*N + photon_additions[i]]
            #expr = c[photon_additions[i]] - d_i[l*N + photon_additions[i]]
            #expr_dag = a[photon_additions[i]] - d_r[l*N + photon_additions[i]]
            for j in range(N):
                # Creation and annihilation terms with their symplectic coefficient
                expr += S_r[N+photon_additions[i], l*dim + j]*a[j]
                expr += S_r[N+photon_additions[i], l*dim + (N+j)]*c[j]
                expr_dag += S_i[N+photon_additions[i], l*dim + j]*c[j]
                expr_dag += S_i[N+photon_additions[i], l*dim + (N+j)]*a[j]
            sup *= expr
            sup_dag *= expr_dag

    if include_obs:
        expanded_expr = []
        if obs == 'witness':
            # NORM
            #expanded_expr.append(expand(sup_dag*(a[0]*c[0])*sup + aux))
            # First-order moments
            for j in range(N):
                #expanded_expr.append(expand(sup_dag*(a[0]*a[j]*c[0])*sup + aux))
                expanded_expr.append(expand(sup_dag*(a[j])*sup + aux))
            # Second-order moments
            for j in range(N):
                for k in range(j,N):
                    #expanded_expr.append(expand(sup_dag*(a[0]*a[j]*a[k]*c[0])*sup + aux))
                    expanded_expr.append(expand(sup_dag*(a[j]*a[k])*sup + aux))
            for j in range(N):
                for k in range(j,N):
                    #expanded_expr.append(expand(sup_dag*(a[0]*c[j]*a[k]*c[0])*sup + aux))
                    expanded_expr.append(expand(sup_dag*(c[j]*a[k])*sup + aux))
            # Third-order moments
            """ for j in range(N):
                for k in range(N):
                    for l in range(N):
                        expanded_expr.append(expand(sup_dag*(a[j]*a[k]*a[l])*sup + aux))
                        expanded_expr.append(expand(sup_dag*(c[j]*a[k]*a[l])*sup + aux))
                        expanded_expr.append(expand(sup_dag*(c[j]*c[k]*a[l])*sup + aux))
                        expanded_expr.append(expand(sup_dag*(c[j]*c[k]*c[l])*sup + aux)) """
        elif obs == 'witness': # FIXME
            expanded_expr.append(expand(sup_dag*(c[0]*a[0]*c[1]*a[1])*sup + aux)) #〈N1N2〉
            expanded_expr.append(expand(sup_dag*(c[0]*a[0])*sup + aux)) #〈N1〉
            expanded_expr.append(expand(sup_dag*(c[1]*a[1])*sup + aux)) #〈N2〉
            #expanded_expr.append(expand(sup_dag*(a[0]*c[0]*a[0]*c[0])*sup + aux)) #〈aN1a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*c[0])*sup + aux)) #〈aa+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*c[0]*a[0]*c[0]*c[0])*sup + aux)) #〈aaN1a+a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*c[0]*c[0])*sup + aux)) #〈aaa+a+〉
            #expanded_expr.append(expand(sup_dag*(c[0]*c[0]*a[0]*a[0])*sup + aux)) #〈a+N1a〉
            #expanded_expr.append(expand(sup_dag*(c[0]*a[0])*sup + aux)) #〈a+a〉
        else:
            for i in range(n_outputs):
                if obs == 'position':
                    # Position operator
                    expr = (sup_dag*a[i]*sup + sup_dag*c[i]*sup)
                elif obs == 'momentum':
                    # Momentum operator
                    expr = (sup_dag*a[i]*sup - sup_dag*c[i]*sup)
                elif obs == 'number':
                    # Number operator
                    expr = sup_dag*c[i]*a[i]*sup + aux
                    """ if len(photon_additions) == 0:
                        expr = sup_dag*c[i]*a[i]*sup + aux
                    else:
                        expr = sup_dag*c[i]*a[i]*sup """
                expanded_expr.append(expand(expr))
    else:
        expanded_expr = expand(sup_dag*sup)
    return expanded_expr

@njit
def wick_expansion_expval(perf_matchings, ladder_modes, ladder_types, N, means_vector, cov_mat_identities):
    '''
    Computes the expected value of the Wick-expanded trace expression terms 
    (sum of product of pairs or singlets of ladder operators applied to a Gaussian state) 
    using loop perfect matchings and the covariance matrix and means vector of the last Gaussian state.
    
    :param perf_matchings: List containing the lists of all loop perfect matchings
    :param ladder_modes: List of modes of the expectation value equation (trace of ladder operators on a Gaussian state)
    :param ladder_types: List defining the type of each ladder operator of the expectation value equation matching the modes
    :param N: Total number of system's modes
    :param means_vector: Expectation values of position and momentum of each mode (xxpp order)
    :param cov_mat_identities: All possible expectation values of pairs of ladder operators acting on a Gaussian state
    :return: Corresponding expectation value of the energy
    '''
    trace_sum = np.complex128(0)
    for perf_match in perf_matchings:
        trace_prod = np.complex128(1+0j)        
        for i1,i2 in zip(perf_match[0::2], perf_match[1::2]):
            term_expval = 0
            # When the pair is -1 -> No more perfect matchings for the term
            if i1 == -1:
                break
            # Different elements in the pair -> Expected value of the pair using covariance matrix and cross relations
            elif i1 != i2:
                term_expval = cov_mat_identities[ladder_types[i1] + 2*ladder_types[i2], ladder_modes[i1], ladder_modes[i2]]
            # Equal elements in the pair (loop) -> Expected value of the singlet using means vector
            else:
                term_expval = single_ladder_exp_val(ladder_modes[i1], ladder_types[i2], N, means_vector)
            trace_prod *= term_expval
        trace_sum += trace_prod
    return trace_sum

@njit
def single_ladder_exp_val(term_mode, term_type, N, means_vector):
    '''
    Computes the expectation value of a single ladder operator over a certain mode based
    on the means vector (position and momentum expectation values).
    
    :param term_mode: Mode over which the ladder operator is acting onto
    :param term_type: Type of ladder operator (creation=1 or annihilation=0)
    :param N: Total number of optical modes
    :param means_vector: Expectation values of position and momentum of each mode (xxpp order)
    :return: Normalized ladder operator expectation value
    '''
    return (1/np.sqrt(2)) * (means_vector[term_mode] + 1j*(-2*term_type + 1)*means_vector[N+term_mode])

@njit
def get_expectation_value(term_modes, term_types, len_term, perf_matchs, N, K_exp_vals, means_vector):
    '''
    Dispatches the expectation value calculation method based on the number of
    terms. Cases: no operators (pure states - return 1), one operator (use means vector), 
    two or more operators (Wick's expansion based on loop perfect matchings).
    
    :param term_modes: Modes of the terms appearing in the expectation value to be computed
    :param term_types: Ladder operator type of the terms
    :param len_term: Total number of terms
    :param perf_matchs: Data structure with needed loop perfect matchings
    :param N: Total number of modes
    :param K_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
    :param means_vector: Position and momentum expectation values of all modes (xxpp order)
    :return: Expectation value of the provided expression
    '''
    if len_term == 0: # CASE Tr[rho] (Pure state)
        return 1
    elif len_term == 1: # CASE Tr[a#rho] (Means vector)
        return single_ladder_exp_val(term_modes[0], term_types[0], N, means_vector)
    else: # CASE Tr[a#a#...rho] -> Wick's expansion
        return wick_expansion_expval(perf_matchs, term_modes, term_types, N, means_vector, K_exp_vals)

@njit(parallel=True)
def compute_terms_in_trace(coefs, modes, types, terms_len, lpms, N, K_exp_vals, means_vector):
    '''
    Computes each term in the trace by calculating the term expression's expectation value
    and multiplying it by its corresponding coefficient.
    
    :param coefs: Terms' coefficients
    :param modes: Array containing the modes of the terms to be computed
    :param types: Array containing the ladder types of the terms (creation or annihilation)
    :param terms_len: Lengths of the terms in the trace
    :param lpms: List of loop perfect matchings regarding the number of operators
    :param N: Total number of system modes
    :param K_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
    :param means_vector: Position and momentum expectation values of all modes (xxpp order)
    :return: Final expectation value of the output
    '''
    sum_tr_values = 0.0 + 0.0*1j
    for i in prange(len(modes)):
        lpms_idx = terms_len[i] - 2 if terms_len[i] > 2 else 0
        exp_val = get_expectation_value(modes[i], types[i], terms_len[i], lpms[lpms_idx][0], N, K_exp_vals, means_vector)
        sum_tr_values += coefs[i] * exp_val
    return sum_tr_values

@njit
def compute_exp_val_loop(N, terms_coefs, norm_coefs, modes, types, unnorm_terms_len, modes_norm, types_norm, norm_terms_len, lpms, K_exp_vals, means_vector):
    '''
    Computes the expectation value of the expression (trace) that defines the QNN operations.
    
    :param N: Total number of system's modes
    :param terms_coefs: Coefficients of the unnormalized terms of the trace expression
    :param norm_coefs: Coefficients of the normalization trace expression
    :param modes: Modes of each term in the trace expression
    :param types: Types of ladder operators of each term in the trace expression
    :param unnorm_terms_len: Lengths of trace expression each term
    :param modes_norm: Modes of each term in the normalization trace expression
    :param types_norm: Types of ladder operators of each term in the normalization expression
    :param norm_terms_len: Lengths of each term in the normalization trace expression
    :param lpms: List of loop perfect matchings regarding the number of operators
    :param K_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
    :param means_vector: Position and momentum expectation values of all modes (xxpp order)
    :return: Normalized expectation value of each output of the QNN
    '''
    # For unnormalized exp val
    unnorm = np.zeros((len(modes)), dtype='complex')
    for outs in prange(len(modes)):
        # TODO: Take care with this condition just made for 0 photon addition and N operator
        if len(modes[outs]) == 1:
            unnorm[outs] = compute_terms_in_trace([1], modes[outs], types[outs], unnorm_terms_len[outs], lpms, N, K_exp_vals, means_vector)
        else:
            unnorm[outs] = compute_terms_in_trace(terms_coefs[outs], modes[outs], types[outs], unnorm_terms_len[outs], lpms, N, K_exp_vals, means_vector)
    
    # For normalization factor
    if len(modes_norm[0]) == 1 and modes_norm[0][0] == -1:
        norm = 1
    else:
        norm = compute_terms_in_trace(norm_coefs, modes_norm[0], types_norm[0], norm_terms_len[0], lpms, N, K_exp_vals, means_vector)
    norm_val = unnorm/norm
  
    return unnorm, norm, norm_val

def loop_perfect_matchings(N):
    '''
    Generates all perfect matchings of a complete graph with loops where
    nodes are labelled from 0 to N-1.
    
    :param N: Number of nodes (number of ladder operators of a given trace term)
    '''
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
    '''
    Creates NumPy arrays of fixed length from nested lists of dynamic sizes
    along with another array containing each list's actual dimensions.
    
    :param lists: Nested list
    :return: Tuple of NumPy array containing the nested lists and the actual
    dimension of the elements inside the lists.
    '''
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
    