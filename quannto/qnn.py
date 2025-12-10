import numpy as np
import time
import jsonpickle
import time
from functools import reduce
import jax
from jax import lax
import jax.numpy as jnp

from .utils import *
from .expectation_value import *
from .results_utils import *

jax.config.update("jax_enable_x64", True)

class QNN:
    '''
    Class for continuous variables quantum (optics) neural network building, training, evaluation and profiling.
    '''
    def __init__(self, model_name, N, layers, n_in, n_out, ladder_modes=[0], is_addition=True, observable='position',
                 include_initial_squeezing=False, include_initial_mixing=False, is_passive_gaussian=False,
                 in_preprocessors=[], out_preprocessors=[], postprocessors=[]):
        # The number of modes N must be greater or equal to the number of inputs and outputs
        assert N >= n_in
        assert N >= n_out
        
        # QONN's architecture hyperparameters
        self.model_name = model_name
        self.N = N
        self.layers = layers
        self.ladder_modes = ladder_modes
        self.is_addition = is_addition
        self.n_in = n_in
        self.n_out = n_out
        self.observable = observable
        self.in_preprocessors = in_preprocessors
        self.out_preprocessors = out_preprocessors
        self.postprocessors = postprocessors
        self.is_input_reupload = False
        self.include_initial_squeezing = include_initial_squeezing
        self.include_initial_mixing = include_initial_mixing
        self.is_passive_gaussian = is_passive_gaussian
        
        # Some useful constants
        self.oneoversqrt2 = 1/jnp.sqrt(2)
        
        # Quadratures - Fock space transformation utils
        self.u_bar = CanonicalLadderTransformations(N)
        
        # Full expectation value expression of the wavefunction (photon additions + observable to be measured)
        self.trace_expr = complete_trace_expression(self.N, layers, ladder_modes, is_addition, self.n_out, include_obs=True, obs=observable)
        # Normalization expression of the wavefunction related to photon additions
        self.norm_trace_expr = complete_trace_expression(self.N, layers, ladder_modes, is_addition, self.n_out, include_obs=False)
        self.trace_expr.append(self.norm_trace_expr)
        
        # Observables constant coefficient
        self.trace_const = 1 if observable=='number' else self.oneoversqrt2 if observable=='position' else -1j*self.oneoversqrt2 if observable=='momentum' else 0
        
        if observable=='cubicphase' or observable=='catstates':
            self.trace_const = jnp.ones(len(self.trace_expr) - 1)
        
        # Extract ladder operators terms for expectation value expression(s)
        self.modes, self.types = [], []
        for outs in range(len(self.trace_expr)):
            modes, types = extract_ladder_expressions(self.trace_expr[outs])
            self.modes.append(modes)
            self.types.append(types)
        self.np_modes, self.lens_modes = to_np_array(self.modes)
        self.np_types, self.lens_types = to_np_array(self.types)
        self.jax_modes = jnp.array(self.np_modes)
        self.jax_types = jnp.array(self.np_types)
        self.jax_lens = jnp.array(self.lens_modes)
        
        # Compute all needed loop perfect matchings for the Wick's expansion of the trace expression
        max_lpms = np.maximum(np.max(self.lens_modes), 2)
        loop_pms = [loop_perfect_matchings(lens) for lens in range(2, max_lpms+1)]
        self.jax_lpms = pad_3d_list_of_lists(loop_pms, 2*max_lpms)
        
        # Remove all ladder operators from symbolic expressions of the trace and norm
        a = symbols(f'a0:{N}', commutative=False)
        c = symbols(f'c0:{N}', commutative=False)
        ladder_subs = {c[i]: 1 for i in range(self.N)}
        ladder_subs.update({a[i]: 1 for i in range(self.N)})
        self.unnorm_expr_terms_out = []
        for outs in range(len(self.trace_expr)):
            if isinstance(self.trace_expr[outs], sp.Add):
                unnorm_expr_terms = list(self.trace_expr[outs].args)
            else:
                unnorm_expr_terms = [self.trace_expr[outs]]
            unnorm_subs_expr_terms = []
            for term in unnorm_expr_terms:
                new_term = term.subs(ladder_subs)
                unnorm_subs_expr_terms.append(new_term)
            self.unnorm_expr_terms_out.append(unnorm_subs_expr_terms)
        
        self.num_terms_per_trace = jnp.array([len(expr) for expr in self.unnorm_expr_terms_out])
        print(f"Number of terms for each trace: {self.num_terms_per_trace}")
        
        self.exp_vals_inds = jnp.arange(len(self.jax_modes), dtype=jnp.int32)
        self.num_terms_in_trace = jnp.array([len(output_terms) for output_terms in self.modes])
        self.max_terms = jnp.max(self.num_terms_in_trace)
        trace_terms_rngs, _ = to_np_array([[[i for i in range(num_terms)] for num_terms in self.num_terms_in_trace]])
        self.trace_terms_ranges = jnp.array(trace_terms_rngs[0])
        
        # Compile trace expression terms with respect to symplectic and displacement parameters
        trace_terms_coefs_inds = []
        trace_terms_coefs = []
        trace_terms_meta = []
        if len(self.ladder_modes) > 0 and self.layers > 1:
            for subexpr_terms in self.unnorm_expr_terms_out:
                all_inds, coefs, meta = expressions_to_index_lists(N, layers, subexpr_terms)
                num_inds = len(all_inds[0])
                for pad in range(len(subexpr_terms), self.max_terms):
                    all_inds.append([-1]*num_inds)
                trace_terms_coefs_inds.append(all_inds)
                trace_terms_coefs.append(coefs)
                trace_terms_meta.append(meta)
            self.jax_traceterms_coefs_inds = jnp.array(trace_terms_coefs_inds, dtype=jnp.int32)
            self.jax_traceterms_coefs = jnp.array(trace_terms_coefs, dtype=jnp.int32)
        else:
            ones_coefs = np.ones_like(self.trace_terms_ranges)
            ones_coefs[self.trace_terms_ranges == -1] = 0
            self.jax_ones_coefs = jnp.array(ones_coefs)
        
    def build_symp_orth_mat(self, parameters):
        '''
        Creates a symplectic-orthogonal (unitary) matrix with the complex exponential 
        of the most general Hermitian matrix built by N**2 parameters.
        
        :param parameters: Parameters to create the Hermitian matrix for the final SO matrix
        :return: Symplectic-orthogonal matrix made to be applied over the quadratures of the system
        '''
        H = general_hermitian_matrix(parameters, self.N)
        U = unitary_from_hermitian(H)
        return self.u_bar.to_canonical_op(U)
        
    def build_quadratic_gaussians(self, parameters, current_par_idx):
        '''
        Builds the symplectic-orthogonal matrices, Q1 and Q2 (passive optics), 
        and the diagonal symplectic matrix, Z (squeezing), of each QONN layer 
        given the vector of tunable parameters.
        
        :param parameters: Vector of all QONN trainable parameters
        :param current_par_idx: Index of the last unused parameter in the vector
        :return: (Q1, Z, Q2, last unused parameter vector index)
        '''
        # Build passive-optics Q1 and Q2 for the Gaussian transformation
        Q1 = self.build_symp_orth_mat(parameters[current_par_idx : current_par_idx + self.N**2])
        current_par_idx += self.N**2        
        Q2 = self.build_symp_orth_mat(parameters[current_par_idx : current_par_idx + self.N**2])
        current_par_idx += self.N**2
        
        # Build squeezing diagonal matrix Z
        sqz_parameters = jnp.exp(jnp.abs(parameters[current_par_idx : current_par_idx + self.N]))
        sqz_inv = 1.0/sqz_parameters
        Z = jnp.diag(jnp.concatenate((sqz_parameters, sqz_inv)))
        current_par_idx += self.N
        
        return Q1, Z, Q2, current_par_idx
    
    def update_S(S_concat, l, block, S_dim):
        '''
        Updates the matrix S_concat by concatenating the symplectic matrix block after the l column.
        
        :param S_concat: Concatenation of symplectic matrices of each layer
        :param l: Layer index
        :param block: Symplectic matrix block to be concatenated
        :param S_dim: Dimension of each symplectic matrix block
        :return: Updated concatenation of symplectic matrices
        '''
        start = l * S_dim
        end   = (l + 1) * S_dim
        return S_concat.at[:, start:end].set(block)

    def build_QNN(self, parameters):
        '''
        Fills the QONN numerical components using the vector of tunable parameters.
        
        :param parameters: Vector of trainable parameters
        '''
        self.tunable_parameters = jnp.copy(parameters)
        S_dim = 2*self.N
        
        # Initial squeezing
        self.Z0 = jnp.zeros((2*self.N, 2*self.N), dtype=jnp.complex128)
        # Initial passive Gaussian mixer
        self.Q0 = jnp.zeros((2*self.N, 2*self.N), dtype=jnp.complex128)
        
        # Symplectic matrix and displacement vectors for each layer for phase and Fock spaces
        self.S_l = jnp.zeros((self.layers, 2*self.N, 2*self.N), dtype=jnp.complex128)
        self.D_l = jnp.zeros((self.layers, 2*self.N), dtype=jnp.complex128)
        self.S_fock = jnp.zeros((self.layers, 2*self.N, 2*self.N), dtype=jnp.complex128)
        self.D_fock = jnp.zeros((self.layers, 2*self.N), dtype=jnp.complex128)
        
        # Bloch-Messiah decomposition of the symplectic matrix for each layer
        self.Q1_gauss = jnp.zeros((self.layers, 2*self.N, 2*self.N), dtype=jnp.complex128)
        self.Q2_gauss = jnp.zeros((self.layers, 2*self.N, 2*self.N), dtype=jnp.complex128)
        self.Z_gauss = jnp.zeros((self.layers, 2*self.N, 2*self.N), dtype=jnp.complex128)
        # Concatenation of symplectic matrix and displacement vectors of each layer
        self.S_concat = jnp.zeros((2*self.N, 2*(self.layers-1)*self.N), dtype=jnp.complex128)
        self.D_concat = jnp.zeros((2*(self.layers-1)*self.N), dtype=jnp.complex128)
        # Final Gaussian transformation when commuting with photon additions (product of all Gaussians)
        self.G = jnp.eye(S_dim, dtype=jnp.complex128)
        self.G_fock = jnp.eye(S_dim, dtype=jnp.complex128)
        
        current_par_idx = 0
        if self.include_initial_squeezing:
            sqz_parameters = jnp.exp(jnp.abs(parameters[current_par_idx : current_par_idx + self.N]))
            sqz_inv = 1.0/sqz_parameters
            self.Z0 = jnp.diag(jnp.concatenate((sqz_parameters, sqz_inv)))
            current_par_idx += self.N
        
        if self.include_initial_mixing:
            self.Q0 = self.build_symp_orth_mat(parameters[current_par_idx : current_par_idx + self.N**2])
            current_par_idx += self.N**2
        
        for l in range(self.layers-1, -1, -1):
            if self.is_passive_gaussian:
                Q1 = self.build_symp_orth_mat(parameters[current_par_idx : current_par_idx + self.N**2])
                current_par_idx += self.N**2
                self.Q1_gauss = self.Q1_gauss.at[l].set(Q1)
                self.S_l = self.S_l.at[l].set(self.Q1_gauss[l])
            else:
            # Build symplectic-orthogonal matrices for passive Gaussian and diagonal symplectic matrix for squeezing
                Q1, Z, Q2, current_par_idx = self.build_quadratic_gaussians(parameters, current_par_idx)
                self.Q1_gauss = self.Q1_gauss.at[l].set(Q1)
                self.Q2_gauss = self.Q2_gauss.at[l].set(Q2)
                self.Z_gauss = self.Z_gauss.at[l].set(Z)
                # Build final quadratic Gaussian transformation
                self.S_l = self.S_l.at[l].set(self.Q2_gauss[l] @ self.Z_gauss[l] @ self.Q1_gauss[l])
                
            self.G = self.G @ self.S_l[l]
            self.S_fock = self.S_fock.at[l].set(jnp.linalg.inv(self.u_bar.to_ladder_op(self.S_l[l])))
            self.G_fock = self.S_fock[l] @ self.G_fock
            
            # Build displacements (linear Gaussian)
            self.D_l = self.D_l.at[l].set(parameters[current_par_idx : current_par_idx + 2*self.N])
            current_par_idx += 2*self.N
            d_fock = self.D_l[l, 0:self.N] + 1j*self.D_l[l, self.N:2*self.N]
            d_fock_vec = jnp.concat((d_fock, d_fock.conj()))
            self.D_fock = self.D_fock.at[l].set(d_fock_vec)
            
            # Build concatenated symplectic matrices and displacements in Fock space for trace expressions' coefficients
            if l > 0:
                self.S_concat = QNN.update_S(self.S_concat, l - 1, self.G_fock, S_dim)
                start = (l - 1) * S_dim
                end   = l * S_dim
                l_disp = -self.S_fock[l] @ d_fock_vec
                if l < (self.layers - 2):
                    next_slice = self.D_concat[(l + 1) * S_dim : (l + 2) * S_dim]
                    l_disp = l_disp + self.S_fock[l] @ next_slice
                self.D_concat = self.D_concat.at[start:end].set(l_disp)
        
    def apply_linear_gaussian(D, mean_vector):
        '''
        Transforms the means vector of a Gaussian state by adding 
        the corresponding linear gaussian to it.
        
        :param D: Vector of displacements of the system's quadratures
        :param mean_vector: Means vector of the Gaussian state
        :return: Transformed means vector
        '''
        return mean_vector + jnp.sqrt(2) * D
        
    def apply_quadratic_gaussian(G, mean_vector, V):
        '''
        Transforms the means vector and the covariance matrix of the Gaussian state 
        given a quadratic Gaussian operator.
        
        :param G: Symplectic matrix of a quadratic Gaussian operator to be applied
        :param mean_vector: Means vector of the Gaussian state
        :param V: Covariance matrix of the Gaussian state
        :return: Transformed means vector and covariance matrix
        '''
        return G @ mean_vector, G @ V @ G.T
        
    def apply_gaussian_transformations(self, mean_vector, V):
        '''
        Applies all Gaussian transformations of the QONN to the initial quantum state.
        
        :param mean_vector: Means vector of the initial Gaussian state
        :param V: Covariance matrix of the initial Gaussian state
        :return: Transformed means vector and covariance matrix
        '''
        new_means = mean_vector
        new_V = V
        for l in range(self.layers):
            new_means, new_V = QNN.apply_quadratic_gaussian(self.S_l[l], new_means, new_V)
            new_means = QNN.apply_linear_gaussian(self.D_l[l], new_means)
        return new_means, new_V
        
    def compute_coefficients(self):
        '''
        Computes the coefficients of each term of each trace expression.
        
        :return: All coefficients of each term of each trace expression
        '''
        # Full coefficient vector for all trace terms
        full_coef_vector = jnp.concatenate(
            [self.D_concat.ravel(), self.S_concat.ravel()]
        )
        
        def _trace_coefs(trace_term_idx, vec):
            def _coef_comp(coefs_inds, vec):
                def null_coef(_):
                    return 0 + 0j
                def valid_coef(_):
                    return jnp.prod(vec[coefs_inds])
                    
                return lax.cond(
                    coefs_inds[0] == -1,
                    null_coef,
                    valid_coef,
                    operand=None
                )
            sym_coefs = jax.vmap(_coef_comp, in_axes=(0, None))(self.jax_traceterms_coefs_inds[trace_term_idx], vec)
            return self.jax_traceterms_coefs[trace_term_idx] * sym_coefs
            
        return jax.vmap(_trace_coefs, in_axes=(0, None))(self.exp_vals_inds, full_coef_vector)
    
    def exp_val_ladder_jk(self, j, k, V, means):
        '''
        Computes the expectation value of two annihilation operators (in mode j and k) of a Gaussian state based 
        on the first two statistical moments of the state.

        :param j: Mode of the first annihilation operator
        :param k: Mode of the second annihilation operator
        :param V: Covariance matrix of the Gaussian state
        :param means: Means vector of the Gaussian state
        :return: Expectation value of a pair of annihilation operators of a Gaussian state
        '''
        return 0.5*(V[j,k] - V[self.N+j, self.N+k] + 1j*(V[j, self.N+k] + V[self.N+j, k]))

    def exp_val_ladder_jdagger_k(self, j, k, V, means):
        '''
        Computes the expectation value of one annihilation and one creation operators (in mode j and k) 
        of a Gaussian state based on the first two statistical moments of the state.

        :param j: Mode of the creation operator
        :param k: Mode of the annihilation operator
        :param V: Covariance matrix of the Gaussian state
        :param means: Means vector of the Gaussian state
        :return: Expectation value of one creation and one annihilation operators of a Gaussian state
        '''
        delta = jnp.where(j == k, 1, 0)
        return 0.5*(V[j,k] + V[self.N+j, self.N+k] + 1j*(V[j, self.N+k] - V[self.N+j, k]) - delta)
    
    def compute_quad_exp_vals(self, V, means):
        '''
        Computes the expectation value of all combinations of ladder operators pairs of all existing modes
        of a Gaussian state based on its covariance matrix and means vector.
        
        :param V: Covariance matrix of the Gaussian state
        :param means: Means vector of the Gaussian state
        :return: All expectation values of the different configuration of a pair of ladder operators on all modes.
        '''
        js = jnp.arange(self.N)
        ks = jnp.arange(self.N)
        
        K00 = jax.vmap(jax.vmap(self.exp_val_ladder_jk, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))(js, ks, V, means)
        K01 = jax.vmap(jax.vmap(self.exp_val_ladder_jdagger_k, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))(js, ks, V, means)
        K10 = K01.T + jnp.eye(self.N, dtype=K01.dtype)
        K11 = K00.conj()

        return jnp.stack([K00, K01, K10, K11], axis=0)
    
    def finalize_observable_expval(self, expvals):
        '''
        Computes the final expectation value based on the observables selected for the QONN outputs,
        their constants and normalization.
        
        :param expvals: All trace expressions including the normalization term in the last position
        '''
        obs_expvals = expvals[:-1]
        norm = expvals[-1]
        if self.observable == 'position':
            obs_expvals = obs_expvals + obs_expvals.conj()
        elif self.observable == 'momentum':
            obs_expvals = obs_expvals - obs_expvals.conj()
        return self.trace_const * obs_expvals / norm
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_QNN(self, params, inputs_disp):
        '''
        Evaluates the QONN for a given input.
        
        :param params: Vector of tunable parameters of the QONN
        :param inputs_disp: Input values of the QONN loaded as real coherent states
        :return: Normalized expectation values of the observables related to the QONN outputs
        '''
        # 0. Build the QONN components using the tunable parameters
        self.build_QNN(params)
        
        # 1. Prepare initial state: initial vacuum state displaced according to the inputs
        tuned_inputs = inputs_disp
        mean_vector = jnp.zeros((2*self.N,), dtype=tuned_inputs.dtype)
        V = 0.5*jnp.eye(2*self.N)
        mean_vector = QNN.apply_linear_gaussian(tuned_inputs, mean_vector)
            
        # 1.1. Add initial squeezing if specified
        if self.include_initial_squeezing:
            mean_vector, V = QNN.apply_quadratic_gaussian(self.Z0, mean_vector, V)
            
        # 1.2. Add initial mixing if specified
        if self.include_initial_mixing:
            mean_vector, V = QNN.apply_quadratic_gaussian(self.Q0, mean_vector, V)

        # 2. Apply the Gaussian transformation acting as weights matrix and bias vector
        mean_vector, V = self.apply_gaussian_transformations(mean_vector, V)
        
        # 3. Compute the expectation values of all combinations of ladder operators pairs over the final Gaussian state
        K_exp_vals = self.compute_quad_exp_vals(V, mean_vector)

        # 4. Compute coefficients for trace expression and normalization terms
        traces_terms_coefs = self.compute_coefficients() if self.layers > 1 else self.jax_ones_coefs
        
        # 5. Compute the expectation values acting as outputs
        exp_vals = self.compute_exp_val_loop(traces_terms_coefs, K_exp_vals, mean_vector)
        
        # 6. Multiply by trace coefficients and normalize (last expectation value)
        return self.finalize_observable_expval(exp_vals)
    
    def wick_expansion_expval(self, trace_idx, tr_term_idx, quadratic_exp_vals, means_vector):
        """
        Vectorized over all perfect‐matchings for a given (trace_idx, tr_term_idx).
        
        :param trace_idx: Index of the trace expression to be evaluated
        :param tr_term_idx: Index of the term to be handled in the trace expression
        :param quadratic_exp_vals: Expectation values of all combinations of ladder operators pairs
        :param means_vector: Position and momentum expectation values of all modes (xxpp order)
        :return: Expectation value of the trace expression term using Wick's expansion
        """

        # 1. Identify the set of loop perfect matchings for the current term
        lpms_idx = self.jax_lens[trace_idx][tr_term_idx] - 2
        all_pms = self.jax_lpms[lpms_idx]

        # 2. Reshape the loop perfect matchings in pair format
        p1 = all_pms[:, ::2]
        p2 = all_pms[:, 1::2]

        # 3. Get ladder modes and types for the current term
        modes_row = self.jax_modes[trace_idx, tr_term_idx]
        types_row = self.jax_types[trace_idx, tr_term_idx]

        def term_prod(p1_row, p2_row):
            def valid_pms(_):
                # 3.1. Identify valid pairs (i.e. exclude padding positions)
                valid = p1_row >= 0 # False for matchings with -1

                # 3.2. Get modes and types for each pair
                m1 = modes_row[p1_row.clip(0)]
                m2 = modes_row[p2_row.clip(0)]
                t1 = types_row[p1_row.clip(0)]
                t2 = types_row[p2_row.clip(0)]

                # 3.3. Compute mean and covariance terms for each pair
                mean_term = self.oneoversqrt2 * (means_vector[m1] + 1j*(-2*t1 + 1)*means_vector[self.N+m1])
                cov_term  = quadratic_exp_vals[t1 + 2*t2, m1, m2]

                # 3.4. Determine expectation value based on whether it is a loop or a pair
                pair_val = jnp.where(p1_row != p2_row, cov_term, mean_term)

                # 3.5. Exclude invalid pairs from the product
                pair_val = jnp.where(valid, pair_val, 1.0 + 0.0j)

                # 3.6. Compute product of all pairs in the loop perfect matching set
                return jnp.prod(pair_val)
            
            def null_pm(_):
                return 0 + 0j
            
            return lax.cond(
                p1_row[0] == -1,
                null_pm,
                valid_pms,
                operand=None
            )

        # 4. Compute products for all existing sets of loop perfect matchings
        prods = jax.vmap(term_prod, in_axes=(0,0))(p1, p2)

        # 5. Sum each set's loop perfect matching product to get final expectation value
        finalsum = jnp.sum(prods)
        return finalsum
    
    def get_expectation_value(self, trace_idx, tr_term_idx, quadratic_exp_vals, means_vector):
        '''
        Dispatches the expectation value calculation method based on the number of
        ladder operators in the term expression.
        Cases: no operators (pure Gaussian state - return 1), one operator (use means vector), 
        two or more operators (Wick's expansion based on loop perfect matchings).
        
        :param trace_idx: Index of the trace expression to be evaluated
        :param tr_term_idx: Index of the term to be handled in the trace expression
        :param quadratic_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
        :param means_vector: Position and momentum expectation values of all modes (xxpp order)
        :return: Expectation value of the provided expression
        '''
        len_term = self.jax_lens[trace_idx, tr_term_idx]

        # Case -1: Non-existent term (padded) → return 0
        def casenull(_):
            return 0.0+0.0j

        # Case 0: No ladder op Tr[ρ] → return 1
        def case0(_):
            return 1.0+0.0j

        # Case 1: Single ladder op Tr[a# ρ] → means vector
        def case1(_):
            mode = self.jax_modes[trace_idx, tr_term_idx, 0]
            typ  = self.jax_types[trace_idx, tr_term_idx, 0]
            return self.oneoversqrt2 * (means_vector[mode] + 1j*(-2*typ + 1)*means_vector[self.N+mode])

        # Case 2: Pair or more ladder ops Tr[a#a#... ρ] → Wick expansion
        def case2(_):
            return self.wick_expansion_expval(
                trace_idx, tr_term_idx, quadratic_exp_vals, means_vector
            )
        
        return lax.cond(
            len_term == -1,
            casenull,
            lambda _: lax.cond(
                len_term == 0,
                case0,
                lambda _: lax.cond(
                    len_term == 1,
                    case1,
                    case2,
                    operand=None
                ),
                operand=None
            ),
            operand=None
        )
        
    def compute_terms_in_trace(self, trace_idx, terms_coefs, quadratic_exp_vals, means_vector):
        '''
        Computes each term in the trace by calculating the term expression's expectation value
        and multiplying it by its corresponding coefficient.
        
        :param trace_idx: Index of the trace expression to be evaluated
        :param terms_coefs: Coefficients of each term in the trace expression
        :param quadratic_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
        :param means_vector: Position and momentum expectation values of all modes (xxpp order)
        :return: Final expectation value of the output
        '''
        expvals = jax.vmap(self.get_expectation_value, in_axes=(None, 0, None, None))(trace_idx, self.trace_terms_ranges[trace_idx], quadratic_exp_vals, means_vector)
        tr_value = jnp.sum(terms_coefs * expvals)
        return tr_value
    
    def compute_exp_val_loop(self, exp_vals_terms_coefs, quadratic_exp_vals, means_vector):
        '''
        Computes the expectation value of the expression (trace) that defines the QONN operations.
    
        :param exp_vals_terms_coefs: Coefficients of the unnormalized terms of the trace expression
        :param quadratic_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
        :param means_vector: Position and momentum expectation values of all modes (xxpp order)
        :return: Unnormalized expectation value of each output of the QONN
        '''
        exp_vals = jax.vmap(self.compute_terms_in_trace, in_axes=(0, 0, None, None))(self.exp_vals_inds, exp_vals_terms_coefs, quadratic_exp_vals, means_vector)
        return exp_vals

    @partial(jax.jit, static_argnums=(0,4))
    def train_QNN(self, parameters, inputs_dataset, outputs_dataset, loss_function):
        '''
        Evaluates all dataset items with the QONN current state and computes the loss function 
        values for each item. This function is the one to be minimized and refers to one epoch.
        
        :param parameters: QONN tunable parameters
        :param inputs_dataset: Inputs of the dataset to be learned
        :param outputs_dataset: Outputs to be learned
        :param loss_function: Function to evaluate the loss of the QONN predictions
        :return: Losses of the QONN predictions
        '''
        # Shuffle dataset
        B = inputs_dataset.shape[0]
        perm = jax.random.permutation(jax.random.PRNGKey(42), B)

        # Evaluate all dataset
        x = inputs_dataset[perm]
        y = outputs_dataset[perm]
        batched_eval = jax.vmap(lambda xi: self.eval_QNN(parameters, xi), in_axes=0, out_axes=0)
        y_hat = batched_eval(x)

        # Compute loss
        return loss_function(y, y_hat)
    
    def print_qnn(self):
        '''
        Prints all QONN items needed to physically build it.
        '''
        print(f"PARAMETERS:\n{self.tunable_parameters}")
        for layer in range(self.layers):
            if self.include_initial_squeezing:
                print(f"=== INITIAL SQUEEZING ===\nZ0 = {self.Z0}")
            if self.include_initial_mixing:
                print(f"=== INITIAL PASSIVE (PRE-MIXER) ===\nQ0 = {self.Q0}")
            if self.is_passive_gaussian:
                print(f"=== PASSIVE LAYER {layer+1} ===\nQ1 = {self.Q1_gauss[layer]}")
            else:
                print(f"=== ACTIVE LAYER {layer+1} ===\nQ1 = {self.Q1_gauss[layer]}\nZ = {self.Z_gauss[layer]}\nQ2 = {self.Q2_gauss[layer]}")
                print(f"Symplectic matrix:\n{self.S_l[layer]}")
            check_symp_orth(self.S_l[layer])
            print(f"Displacement vector:\n{self.D_l[layer]}")
            if layer > 0:
                print(f"Symplectic coefficients:\n{self.S_concat[:,(layer-1)*2*self.N : layer*2*self.N]}")
                print(f"Displacement coefficients:\n{self.D_concat[(layer-1)*2*self.N : layer*2*self.N]}")
        
    def test_model(self, testing_dataset, loss_function):
        '''
        Makes predictions of the given QNN using the input testing dataset.
        
        :param testing_dataset: List of inputs and outputs to be tested
        :param loss_function: Loss function to compute the testing set losses
        :return: QNN predictions of the testing set
        '''
        test_inputs = reduce(lambda x, func: func(x), self.in_preprocessors, testing_dataset[0])
        test_outputs = reduce(lambda x, func: func(x), self.out_preprocessors, testing_dataset[1])
        
        qnn_outputs = jax.vmap(self.eval_QNN, in_axes=(None, 0))(self.tunable_parameters, test_inputs)
        
        mean_error = loss_function(test_outputs, qnn_outputs)
        print(f"LOSS VALUE FOR TESTING SET: {mean_error}")
        print("\n==========\n")
        
        return reduce(lambda x, func: func(x), self.postprocessors, qnn_outputs)
    
    def evaluate_model(self, test_inputs):
        '''
        Makes predictions of the given QNN using the input testing dataset.
        
        :param test_inputs: Inputs to be evaluated by the QNN
        :return: Post-processed QNN predictions of the input set
        '''
        test_inputs = reduce(lambda x, func: func(x), self.in_preprocessors, test_inputs)
        
        # Evaluate all testing set
        qnn_outputs = np.real_if_close(
            jax.vmap(self.eval_QNN, in_axes=(None, 0))(self.tunable_parameters, test_inputs), tol=1e6
        )
        
        return reduce(lambda x, func: func(x), self.postprocessors, qnn_outputs)
    
    def save_model(self, filename):
        '''
        Saves the QONN model to a JSON pickle file.
        
        :param filename: Path to save the QONN model
        '''
        f = open("models/"+filename, 'w')
        f.write(jsonpickle.encode(self))
        f.close()

    def load_model(filename):
        '''
        Loads a QONN model from a JSON pickle file.
        
        :param filename: Path to the QONN model file
        :return: QONN model
        '''
        with open(filename, 'r') as f:
            qnn_str = jsonpickle.decode(f.read())
        return qnn_str
    
