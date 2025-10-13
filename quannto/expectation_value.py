from __future__ import annotations
import numpy as np
from sympy import symbols, expand, MatrixSymbol
import jax.numpy as jnp
from functools import partial

from .utils import *

# ---------- PRE-CÁLCULO (una sola vez, con SymPy) ----------
import re
from typing import List, Tuple, Sequence
import sympy as sp
from sympy.matrices.expressions.matexpr import MatrixElement
import numpy as np

# ===== TEST =====

_num_re = re.compile(r"^([d])(\d+)$")  # d<number>

def build_meta_from_symbols(N, layers):
    dim = 2*N
    d = symbols(f'd0:{layers*2*N}', commutative=True)
    # Symplectic matrix 2Nx2N
    S_i = MatrixSymbol('S_i', dim, layers*dim)
    """A partir de tus símbolos, calcula offsets/tamaños del vector concatenado."""
    rows, cols = S_i.shape
    assert S_i.shape == (rows, cols)
    size_d = len(d)
    size_S = rows * cols

    off_d  = 0
    off_Si = off_d + size_d
    total  = off_Si + size_S

    return dict(
        rows=rows, cols=cols,
        size_d=size_d, size_S=size_S,
        off_d=off_d, off_Si=off_Si,
        total=total
    )

def _flat_rc_row_major(p: int, q: int, cols: int) -> int:
    return int(p) * int(cols) + int(q)

def _is_matrix_element(x) -> bool:
    # Evita dependencias de versión: MatrixElement tiene .parent, .i, .j
    return hasattr(x, "parent") and hasattr(x, "i") and hasattr(x, "j")

def factor_to_index(f, meta) -> int:
    """Mapea un factor simbólico (rK, iK, S_i[p,q]) a índice en v."""
    rows, cols = meta["rows"], meta["cols"]
    off_d, off_Si = meta["off_d"], meta["off_Si"]
    
    # Caso rK / iK (tus d_r/d_i)    
    if isinstance(f, sp.Symbol):
        m = _num_re.match(f.name)
        if m:
            base, k = m.groups()
            k = int(k)
            if base == "d":
                if not (0 <= k < meta["size_d"]): raise IndexError(f"r{k} fuera de rango")
                return off_d + k

    # Caso MatrixElement: S_i[p,q]
    if _is_matrix_element(f):
        name = str(f.parent)
        p, q = int(f.i), int(f.j)
        if not (0 <= p < rows and 0 <= q < cols):
            raise IndexError(f"Índices fuera de rango para {name}: ({p},{q}) con shape=({rows},{cols})")
        flat = _flat_rc_row_major(p, q, cols)
        if name == "S_i":
            return off_Si + flat

    raise TypeError(f"No sé mapear el factor: {repr(f)}")

# -----------------------------
# Conversión de expresiones
# -----------------------------
def expr_to_index_list_and_coeff(expr, meta) -> Tuple[List[int], float]:
    """
    Devuelve (indices, coeff_numérico).
    - indices: lista de índices (uno por factor simbólico no numérico).
    - coeff: producto de los factores numéricos (float). Si no hay, 1.0.
    """
    # Transforms type sp.Pow to sp.Mul
    if isinstance(expr, sp.Pow):
        base, exp = expr.as_base_exp()
        expr = sp.Mul(*([base]*int(exp)), evaluate=False)
        
    # Separates numerical and symbolic
    if isinstance(expr, sp.Mul):
        nums = [a for a in expr.args if a.is_number]
        non_nums = []
        for a in expr.args:
            if isinstance(a, sp.Pow):
                base, exp = a.as_base_exp()
                non_nums.extend([base]*exp)
            elif not a.is_number:
                non_nums.append(a)
        coeff = float(sp.Mul(*nums)) if nums else 1.0
    elif expr.is_number:
        return [], float(expr)
    else:
        non_nums, coeff = [expr], 1.0

    idxs = [factor_to_index(f, meta) for f in non_nums]
    return idxs, coeff

def expressions_to_index_lists(
    N, layers, exprs: Sequence[sp.Expr], keep_coeffs: bool = True
):
    """
    API principal: convierte cada expresión en su lista de índices.
    - Si keep_coeffs=True, también devuelve array de coeficientes (float32).
    """
    meta = build_meta_from_symbols(N, layers)
    all_idxs: List[List[int]] = []
    coeffs: List[float] = []
    for e in exprs:
        idxs, c = expr_to_index_list_and_coeff(e, meta)
        all_idxs.append(idxs)
        if keep_coeffs:
            coeffs.append(c)

    if keep_coeffs:
        return all_idxs, np.asarray(coeffs, dtype=np.float32), meta
    else:
        return all_idxs, meta

# ===== /TEST ======


def extract_ladder_expressions(trace_expr):
    '''
    Transforms a symbolic expression representing ladder operators acting on a Gaussian quantum state to
    numerical lists of the ladder modes (from 0 to N-1) and the operators type (creation or annihilation).

    :param trace_expr: Symbolic expression of creation and annihilation operators
    :return: Pair of lists containing the modes the ladder operators act onto and their type
    '''
    if isinstance(trace_expr, sp.Add):
        trace_args = list(trace_expr.args)
    elif isinstance(trace_expr, sp.Symbol) or isinstance(trace_expr, sp.Mul):
        trace_args = [trace_expr]
    elif isinstance(trace_expr, sp.core.numbers.One):
        trace_args = []

    ladder_modes = []
    ladder_types = []
    if len(trace_args) == 0:
        ladder_modes = [[]]
        ladder_types = [[]]
    else:
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

def complete_trace_expression(N, layers, ladder_modes, is_addition, n_outputs, include_obs=False, obs='position'):
    '''
    Builds the non-Gaussian state symbolic expression of a multi-photon added
    QNN based on superposition of ladder operators applied to the Gaussian state.

    :param N: Number of modes of the QNN
    :param layers: Number of layers of the QNN
    :param ladder_modes: Photon additions made over the modes at each layer
    :param n_outputs: Number of QNN outputs
    :param include_obs: Whether to build the trace expression with or without the observable (default=False)
    :param obs: Observable to be measured (QNN output)
    :return: Symbolic expressions of the different QNN outputs (or normalization factor)
    '''
    dim = 2*N
    # Displacement vector (complex number 'r' and its conjugate 'i') in Fock for each mode
    d = symbols(f'd0:{layers*dim}', commutative=True)
    # Symplectic matrix in Fock 2Nx2N
    S_i = MatrixSymbol('S_i', dim, layers*dim)
    # Creation (c) and annihilation (a) operators for each mode
    c = symbols(f'c0:{N}', commutative=False)
    a = symbols(f'a0:{N}', commutative=False)
    
    sup = 1
    sup_dag = 1
    for l in range(layers):
        for i in range(len(ladder_modes)):
            # Expectation value indexing in function of the ladder operator
            if is_addition:
                idx = N + ladder_modes[i]
                idx_dag = ladder_modes[i]
            else:
                idx = ladder_modes[i]
                idx_dag = N + ladder_modes[i]
            # Displacement terms
            expr = d[l*dim + idx]
            expr_dag = d[l*dim + idx_dag]
            # Creation and annihilation terms with their symplectic coefficient
            for j in range(N):
                expr += S_i[idx, l*dim + j]*a[j]
                expr += S_i[idx, l*dim + N+j]*c[j]
                expr_dag += S_i[idx_dag, l*dim + j]*a[j]
                expr_dag += S_i[idx_dag, l*dim + N+j]*c[j]
            sup = expr * sup
            sup_dag = sup_dag * expr_dag
    
    if include_obs:
        expanded_expr = []
        if obs == 'witness': # FIXME
            #expanded_expr.append(expand(sup_dag*(c[0]*a[0]*c[1]*a[1])*sup)) #〈N1N2〉
            #expanded_expr.append(expand(sup_dag*(c[0]*a[0])*sup)) #〈N1〉
            #expanded_expr.append(expand(sup_dag*(c[1]*a[1])*sup)) #〈N2〉
            #expanded_expr.append(expand(sup_dag*(a[0]*c[0]*a[0]*c[0])*sup)) #〈aN1a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*c[0])*sup)) #〈aa+〉
            #expanded_expr.append(expand(sup_dag*(c[0]*a[0])*sup)) #〈a+a〉
            #expanded_expr.append(expand(sup_dag*(c[0]*c[0]*a[0]*a[0])*sup)) #〈a+a+aa〉
            #expanded_expr.append(expand(sup_dag*(a[0])*sup)) #〈a〉
            #expanded_expr.append(expand(sup_dag*(c[0])*sup)) #〈a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*c[0])*sup)) #〈a a a+〉
            expanded_expr.append(expand(sup_dag*(a[0]*c[0])*sup)) #〈aa+〉
            #expanded_expr.append(expand(sup_dag*(c[0]*a[0])*sup)) #〈a+a〉
            #expanded_expr.append(expand(sup_dag*(a[0]*c[0]*a[0])*sup)) #〈a a+ a〉
            #expanded_expr.append(expand(sup_dag*(c[0]*a[0]*a[0])*sup)) #〈a+ a a〉
            #expanded_expr.append(expand(sup_dag*(a[0]*c[0]*c[0])*sup)) #〈a a+ a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*c[0]*a[0]*c[0]*c[0])*sup)) #〈aaN1a+a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*a[0]*c[0]*c[0])*sup)) #〈aa a a+a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*c[0]*c[0])*sup)) #〈aaa+a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*c[0]*c[0]*c[0])*sup)) #〈aa a+ a+a+〉
            #expanded_expr.append(expand(sup_dag*(c[0]*c[0]*a[0]*a[0])*sup)) #〈a+N1a〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*a[0]*c[0]*c[0]*c[0])*sup)) #〈aaaa+a+a+〉
            #expanded_expr.append(expand(sup_dag*(c[0]*c[0]*c[0]*a[0]*a[0]*a[0])*sup)) #〈aaaa+a+a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*a[0]*a[0]*c[0]*c[0]*c[0])*sup)) #〈aaa a a+a+a+〉
            #expanded_expr.append(expand(sup_dag*(a[0]*a[0]*a[0]*c[0]*c[0]*c[0]*c[0])*sup)) #〈aaa a+ a+a+a+〉
        elif obs == 'third-order':
            expanded_expr.append(expand(sup_dag*a[0]*sup))
            expanded_expr.append(expand(sup_dag*c[0]*sup))
            expanded_expr.append(expand(sup_dag*a[0]*a[0]*sup))
            expanded_expr.append(expand(sup_dag*c[0]*c[0]*sup))
            expanded_expr.append(expand(sup_dag*a[0]*a[0]*a[0]*sup))
            expanded_expr.append(expand(sup_dag*c[0]*c[0]*c[0]*sup))
            expanded_expr.append(expand(sup_dag*c[0]*a[0]*a[0]*sup))
            expanded_expr.append(expand(sup_dag*c[0]*c[0]*a[0]*sup))
            expanded_expr.append(expand(sup_dag*c[0]*a[0]*sup))
            expanded_expr.append(expand(sup_dag*c[0]*a[0]*c[0]*a[0]*sup))
        else:
            for i in range(n_outputs):
                if obs == 'position' or obs == 'momentum':
                    # 1st order observables
                    expr = sup_dag*a[i]*sup
                elif obs == 'number':
                    # 2nd order observable
                    expr = sup_dag*c[i]*a[i]*sup
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
    max_length_list = 0
    for list in lists:
        if len(list) > max_length_list:
            max_length_list = len(list)
    lengths = np.full((len(lists), max_length_list), -1) if max_length_list > 0 else np.array([[0]])
    for out_idx in range(len(lists)):
        for term_idx in range(len(lists[out_idx])):
            lengths[out_idx, term_idx] = len(lists[out_idx][term_idx])
    max_length = np.max(lengths)
    arr = np.full((len(lists), max_length_list, max_length), -1) if max_length_list > 0 else np.array([[[-1]]])
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