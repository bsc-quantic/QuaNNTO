import numpy as np
from sympy import symbols, expand, MatrixSymbol, Matrix
from sympy.parsing.sympy_parser import parse_expr
from numba import njit, prange

from .utils import *

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
    #trace_terms = str(trace_expr).replace(" ","").replace("*","").split('+')
    #trace_terms = str(trace_expr).split('+')
    ladder_modes = []
    ladder_types = []
    #print(trace_terms)
    for sym_term in trace_expr.args:
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
    return [ladder_modes], [ladder_types]

def complete_trace_expression(N, photon_additions, n_outputs, include_obs = False):
    '''
    Builds the non-Gaussian state expression of a multi-photon added
    QNN based on superposition of ladder operators applied to the Gaussian state.

    :param N: Number of modes of the QNN
    :param layers: Number of layers of the QNN
    :return: Pair of lists containing the different expression terms of the modes the ladder operators act onto and their type
    '''
    dim = 2*N
    # Displacement vector (complex number 'r' and its conjugate 'i') for each mode
    d_r = symbols(f'r0:{N}', commutative=True)
    d_i = symbols(f'i0:{N}', commutative=True)
    # Symplectic matrix 2Nx2N
    S = MatrixSymbol('S', dim, dim)
    # Creation (c) and annihilation (a) operators for each mode
    c = symbols(f'c0:{N}', commutative=False)
    a = symbols(f'a0:{N}', commutative=False)
    
    sup = 1
    sup_dag = 1
    for i in range(len(photon_additions)):
        # Displacement terms
        expr = d_r[photon_additions[i]]
        expr_dag = d_i[photon_additions[i]]
        for j in range(N):
            # Creation and annihilation terms with their symplectic coefficient
            expr += S[photon_additions[i], j]*a[j]
            expr += S[photon_additions[i], N+j]*c[j]
            expr_dag += S[photon_additions[i], N+j]*a[j]
            expr_dag += S[photon_additions[i], j]*c[j]
        sup *= expr
        sup_dag *= expr_dag
    
    #print(sup)
    #print(sup_dag)
    # TODO: Generalize for number of outputs (different expanded expression of the expectation value)
    if include_obs:
        expanded_expr = []
        for i in range(n_outputs):
            # Position operator
            #expr = (1/np.sqrt(2)) * (sup_dag*a[i]*sup + sup_dag*c[i]*sup)
            expr = (1/np.sqrt(2))*(sup_dag*a[i]*sup + sup_dag*c[i]*sup)
            # Number operator
            #expr = (sup_dag*c[i]*a[i]*sup)
            expanded_expr.append(expr)
    else:
        expanded_expr = sup_dag*sup
    #str_expr = str(expanded_expr)
    #print(len(str_expr.split("+")))
    print(expanded_expr)
    return expanded_expr

def subs_in_trace_terms(trace_expr, D, symp_mat):
    trace_terms = list(trace_expr.args)
    expr_terms = []
    S = MatrixSymbol('S', len(symp_mat), len(symp_mat))
    S_mat = Matrix(symp_mat)
    N = len(D)//2
    for term in trace_terms:
        sym_term = term
        #print(sym_term)
        for disp_idx in range(N):
            sym_term = sym_term.subs({f"i{disp_idx}": D[disp_idx]-1j*D[N+disp_idx], f"r{disp_idx}": D[disp_idx]+1j*D[N+disp_idx]})
            sym_term = sym_term.subs(S, S_mat)
            #print(sym_term)
        expr_terms.append(sym_term)
    return expr_terms

def single_ladder_exp_val(term_mode, term_type, N, means_vector):
    return (1/np.sqrt(2)) * (means_vector[term_mode] + 1j*(-2*term_type + 1)*means_vector[N+term_mode])


def compute_exp_val_loop(unnorm_expr, norm_expr, modes, types, modes_norm, types_norm, lpms, D, G, K_exp_vals, means_vector):
    unnorm_terms = subs_in_trace_terms(unnorm_expr, D, G)
    norm_terms = subs_in_trace_terms(norm_expr, D, G)
    N = len(G) // 2

    # For unnormalized exp val
    trace_values=[]
    for i in prange(len(modes)):
        for j in prange(len(modes[0])):
            if unnorm_terms[j] != 0:
                trace_values.append(get_expectation_value(modes[i][j], types[i][j], lpms[0], N, K_exp_vals, means_vector))
            else:
                trace_values.append(0)
    exp_val = []
    for (coef, term) in zip(trace_values, unnorm_terms):
        exp_val.append(coef * term.subs(dict(zip(term.free_symbols, [1 for i in range(len(term.free_symbols))])))) 
    unnorm = expand(sum(exp_val))
    
    # For normalization factor
    if len(modes_norm[0]) == 0:
        norm = 1
    else:
        trace_values=[]
        for i in prange(len(modes_norm)):
            for j in prange(len(modes_norm[0])):
                if norm_terms[j] != 0:
                    #TO-DO: Generalize lpms for any number of ladder operators (currently only 3)
                    trace_values.append(get_expectation_value(modes_norm[i][j], types_norm[i][j], lpms[0], N, K_exp_vals, means_vector))
                else:
                    trace_values.append(0)
        exp_val = []
        for (coef, term) in zip(trace_values, norm_terms):
            exp_val.append(coef * term.subs(dict(zip(term.free_symbols, [1 for i in range(len(term.free_symbols))])))) 
        norm = expand(sum(exp_val))
    
    norm_val = np.real_if_close(np.complex128(expand(unnorm/norm)))
    return norm_val

def get_expectation_value(term_modes, term_types, perf_matchs, N, K_exp_vals, means_vector):
    if len(term_modes) == 0: # CASE Tr[rho]
        #print("NUM LADDERS: 0")
        return 1
    elif len(term_modes) == 1: # CASE Tr[a#rho]
        #print("NUM LADDERS: 1")
        return single_ladder_exp_val(term_modes[0], term_types[0], N, means_vector)
    elif len(term_modes) == 2: # CASE Tr[a#a#rho]
        #print("NUM LADDERS: 2")
        return K_exp_vals[term_types[0] + 2*term_types[1], term_modes[0], term_modes[1]]
    else:
        #print(f"NUM LADDERS: {len(term_modes)}")
        return ladder_exp_val(perf_matchs, term_modes, term_types, means_vector, K_exp_vals)
    
def ladder_exp_val(perf_matchings, ladder_modes, ladder_types, means_vector, cov_mat_identities):
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
            if i1!=i2:
                trace_prod *= cov_mat_identities[ladder_types[i1] + 2*ladder_types[i2], ladder_modes[i1], ladder_modes[i2]]
            else:
                trace_prod *= single_ladder_exp_val(ladder_modes[i1], ladder_types[i2], len(means_vector) // 2, means_vector)
        energy += trace_prod
    return energy

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
    

def complete_exp_val(modes, types, perf, N, K, means):
    values = []
    for i in range(len(modes)):
        for j in range(len(modes[0])):
            values.append(get_expectation_value(modes[i][j], types[i][j], perf, N, K, means))
    return values

