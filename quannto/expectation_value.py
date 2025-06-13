import numpy as np
from sympy import symbols, expand, MatrixSymbol
import jax.numpy as jnp
from functools import partial

from .utils import *

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
    if len(trace_expr.args) == 0:
        ladder_modes = [[]]
        ladder_types = [[]]
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
    for l in range(layers-1, -1, -1):
        for i in range(len(photon_additions)):
            # Displacement terms
            expr = d_i[l*N + photon_additions[i]]
            #expr = c[photon_additions[i]] - d_i[l*N + photon_additions[i]]
            for j in range(N):
                # Creation and annihilation terms with their symplectic coefficient
                expr += S_r[N+photon_additions[i], l*dim + j]*a[j]
                expr += S_r[N+photon_additions[i], l*dim + (N+j)]*c[j]
            sup *= expr
    sup_dag = 1
    for l in range(layers):
        for i in range(len(photon_additions)):
            # Displacement terms
            expr_dag = d_r[l*N + photon_additions[i]]
            #expr_dag = a[photon_additions[i]] - d_r[l*N + photon_additions[i]]
            for j in range(N):
                # Creation and annihilation terms with their symplectic coefficient
                expr_dag += S_i[N+photon_additions[i], l*dim + j]*c[j]
                expr_dag += S_i[N+photon_additions[i], l*dim + (N+j)]*a[j]
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
    
def pad_3d_list_of_lists(raw, max_len_inner, pad_value=-1):
    """
    raw:        Python list of length G, each an inner list of variable-length lists of ints
    max_len_inner:  target length for each innermost int-list
    pad_value:  int to pad with
    ---
    returns: jnp.ndarray of shape (G, Mmax, max_len_inner)
    """
    G = len(raw)
    # 1) pad each inner list to max_len_inner
    padded_inners = [
        [sub[:max_len_inner] + [pad_value] * max(0, max_len_inner - len(sub))
         for sub in group]
        for group in raw
    ]
    # 2) find Mmax = max number of sublists in any group
    Mmax = max(len(group) for group in padded_inners)
    # 3) pad each group to Mmax by appending dummy inners
    dummy = [pad_value] * max_len_inner
    padded_groups = [
        grp + [dummy] * (Mmax - len(grp))
        for grp in padded_inners
    ]
    # 4) stack into a single JAX array
    return jnp.array(padded_groups, dtype=jnp.int32)