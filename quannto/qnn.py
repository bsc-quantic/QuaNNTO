import numpy as np
import time
import jsonpickle
import time
from sympy import lambdify
from functools import reduce
import jax
from jax import lax
import jax.numpy as jnp

from .utils import *
from .expectation_value import *
from .results_utils import *

jax.config.update("jax_enable_x64", True)

class ProfilingQNN:
    '''
    Data structure containing the times for each part of the training process 
    of a QNN with a particular number of modes and layers.
    '''
    def __init__(self, N, layers):
        self.N = N
        self.layers = layers
        self.build_qnn_times = []
        self.input_prep_times = []
        self.gauss_times = []
        self.K_exp_vals_times = []
        self.ladder_superpos_times = []
        self.nongauss_times = []
        self.epoch_times = []

    def avg_benchmark(self):
        self.avg_times = {}
        self.avg_times["Build QNN"] = sum(self.build_qnn_times)/len(self.build_qnn_times)
        self.avg_times["Input prep"] = sum(self.input_prep_times)/len(self.input_prep_times)
        self.avg_times["Gaussian op"] = sum(self.gauss_times)/len(self.gauss_times)
        self.avg_times["Pairs exp-vals"] = sum(self.K_exp_vals_times)/len(self.K_exp_vals_times)
        self.avg_times["Non-gauss coefs"] = sum(self.ladder_superpos_times)/len(self.ladder_superpos_times)
        self.avg_times["Non-gaussianity"] = sum(self.nongauss_times)/len(self.nongauss_times)
        return self.avg_times
    
    def avg_epochs(self):
        avg_epochs = sum(self.epoch_times) / len(self.epoch_times)
        print(f'TOTAL EPOCHS: {len(self.epoch_times)}')
        print(f'AVERAGE EPOCH TIME: {avg_epochs}')
        return avg_epochs
    
    def clear_times(self):
        self.build_qnn_times = []
        self.input_prep_times = []
        self.gauss_times = []
        self.K_exp_vals_times = []
        self.ladder_superpos_times = []
        self.nongauss_times = []
        self.avg_epochs = []
    
class QNN:
    '''
    Class for continuous variables quantum (optics) neural network building, training, evaluation and profiling.
    '''
    def __init__(self, model_name, N, layers, n_in, n_out, photon_add=[0], observable='position', is_input_reupload=False,
                 in_preprocessors=[], out_preprocessors=[], postprocessors=[]):
        # The number of modes N must be greater or equal to the number of inputs and outputs
        assert N >= n_in
        assert N >= n_out
        
        # QONN's architecture hyperparameters
        self.model_name = model_name
        self.N = N
        self.layers = layers
        self.photon_add = photon_add
        self.n_in = n_in
        self.n_out = n_out
        self.observable = observable
        self.in_preprocessors = in_preprocessors
        self.out_preprocessors = out_preprocessors
        self.postprocessors = postprocessors
        self.is_input_reupload = is_input_reupload
        
        # Some useful constants
        self.oneoversqrt2 = 1/np.sqrt(2)
        
        # Quadratures - Fock space transformation utils
        self.u_bar = CanonicalLadderTransformations(N)
        
        # Benchmarking utils for QONN training
        self.qnn_profiling = ProfilingQNN(N, layers)
        
        # Observable constant coefficient
        self.trace_const = 1 if observable=='number' else (1/np.sqrt(2)) if observable=='position' else (-1j/np.sqrt(2)) if observable=='momentum' else 0
        
        # Full expectation value expression of the wavefunction (photon additions + observable to be measured)
        self.trace_expr = complete_trace_expression(self.N, layers, photon_add, self.n_out, include_obs=True, obs=observable)
        # Normalization expression of the wavefunction related to photon additions
        self.norm_trace_expr = complete_trace_expression(self.N, layers, photon_add, self.n_out, include_obs=False)
        
        # Extract ladder operators terms for expectation value expression(s)
        self.modes, self.types = [], []
        for outs in range(len(self.trace_expr)):
            modes, types = extract_ladder_expressions(self.trace_expr[outs])
            self.modes.append(modes)
            self.types.append(types)
        # Extract ladder operators terms for normalization expression
        modes_norm, types_norm = extract_ladder_expressions(self.norm_trace_expr)
        self.modes.append(modes_norm)
        self.types.append(types_norm)
        
        self.np_modes, self.lens_modes = to_np_array(self.modes)
        self.np_types, self.lens_types = to_np_array(self.types)
        self.jax_modes = jnp.array(self.np_modes)
        self.jax_types = jnp.array(self.np_types)
        self.jax_lens = jnp.array(self.lens_modes)
        
        # Compute all needed loop perfect matchings for the Wick's expansion of the trace expression
        max_lpms = np.maximum(np.max(self.lens_modes), 2)
        loop_pms = [loop_perfect_matchings(lens) for lens in range(2, max_lpms+1)]
        self.jax_lpms = pad_3d_list_of_lists(loop_pms, 2*max_lpms)
        
        # Remove all ladder operators from symbolic expressions of the trace and normalization
        a = symbols(f'a0:{N}', commutative=False)
        c = symbols(f'c0:{N}', commutative=False)
        ladder_subs = {c[i]: 1 for i in range(self.N)}
        ladder_subs.update({a[i]: 1 for i in range(self.N)})
        
        self.unnorm_expr_terms_out = []
        for outs in range(len(self.trace_expr)):
            unnorm_expr_terms = list(self.trace_expr[outs].args) if (len(photon_add) > 0 or (observable!='number' and observable!='witness')) else list(self.trace_expr[outs].args[1:])
            unnorm_subs_expr_terms = []
            for term in unnorm_expr_terms:
                new_term = term.subs(ladder_subs)
                unnorm_subs_expr_terms.append(new_term)
            self.unnorm_expr_terms_out.append(unnorm_subs_expr_terms)
        
        norm_expr_terms = list(self.norm_trace_expr.args)
        if len(norm_expr_terms) == 0:
            self.unnorm_expr_terms_out.append([1])
        else:
            self.norm_subs_expr_terms = []
            for norm_term in norm_expr_terms:
                new_term = norm_term.subs(ladder_subs)
                self.norm_subs_expr_terms.append(new_term)
            self.unnorm_expr_terms_out.append(self.norm_subs_expr_terms)
        self.num_terms_per_trace = jnp.array([len(expr) for expr in self.unnorm_expr_terms_out])
        print(f"Number of terms for each trace: {self.num_terms_per_trace}")
        
        d = symbols(f'r0:{layers*N}', commutative=True)
        d_conj = symbols(f'i0:{layers*N}', commutative=True)
        dim = 2*N
        S_r = MatrixSymbol('S_r', dim, layers*dim)
        S_i = MatrixSymbol('S_i', dim, layers*dim)
        
        self.exp_vals_inds = jnp.arange(len(self.jax_modes), dtype=jnp.int32)
        self.num_terms_in_trace = jnp.array([len(output_terms) for output_terms in self.modes])
        self.max_terms = jnp.max(self.num_terms_in_trace)
        trace_terms_rngs, _ = to_np_array([[[i for i in range(num_terms)] for num_terms in self.num_terms_in_trace]])
        self.trace_terms_ranges = jnp.array(trace_terms_rngs[0])
        
        # Compile trace expression terms with respect to symplectic and displacement parameters
        t1 = time.time()
        self.unnorm_fns = []
        for expr_list in self.unnorm_expr_terms_out:
            # 1) Introduce padding for those traces not reaching the maximum number of terms
            for pad in range(len(expr_list), self.max_terms):
                expr_list.append(0)
            # 2) Turn the list of sympy scalars into a Python tuple
            expr_tuple = tuple(expr_list)  # length = m_i

            # 3) Lambdify that tuple: returns a Python tuple of JAX scalars
            f_vec = lambdify(
                (S_r, S_i, d, d_conj),
                expr_tuple,
                modules={'jax': jnp, 'numpy': jnp}
            )
            #   now f_vec(Sr, Si, d_r, d_i) → Python tuple (v0, v1, …, v_{m_i−1})
            #   each v_k is a JAX scalar (dtype complex64/128).

            # 4) Wrap in a one‐line function that stacks into a 1D array:
            def f_stack(Sr, Si, d_r, d_i, _f=f_vec):
                # NOTE: Python tuple → stack to JAX array of shape (m_i,)
                return jnp.stack(_f(Sr, Si, d_r, d_i), axis=0)

            # 5) JIT‐compile that stacked function once
            f_jit = jax.jit(f_stack)
            self.unnorm_fns.append(f_jit)
        t2 = time.time()
        print(f"Time to lambdify trace terms: {t2-t1}")
        
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
        
    def build_quadratic_gaussians(self,
                                  parameters:      jnp.ndarray,
                                  current_par_idx: jnp.ndarray  # tracer-compatible int
                                 ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Builds (Q1, Z, Q2) and returns updated current_par_idx,
        slicing `parameters` via lax.dynamic_slice so it works under jit.
        """

        N2 = self.N * self.N      # number of parameters for each orth mat
        # 1) Slice out the next N2 params for Q1:
        p1 = lax.dynamic_slice(parameters,
                               start_indices=(current_par_idx,),
                               slice_sizes=(N2,))
        current_par_idx = current_par_idx + N2

        # 2) Slice out the next N2 params for Q2:
        p2 = lax.dynamic_slice(parameters,
                               start_indices=(current_par_idx,),
                               slice_sizes=(N2,))
        current_par_idx = current_par_idx + N2

        # 3) Build squeezing diagonal Z from the next N params:
        sqN = self.N
        p3  = lax.dynamic_slice(parameters,
                                start_indices=(current_par_idx,),
                                slice_sizes=(sqN,))
        current_par_idx = current_par_idx + sqN

        # Now build your matrices
        Q1 = self.build_symp_orth_mat(p1)
        Q2 = self.build_symp_orth_mat(p2)

        sqz_parameters = jnp.exp(jnp.abs(p3))        # shape (N,)
        sqz_inv        = 1.0 / sqz_parameters       # shape (N,)
        Z              = jnp.diag(jnp.concatenate([sqz_parameters, sqz_inv]))

        return Q1, Z, Q2, current_par_idx
    
    def build_reuploading_disp(self, inputs):
        '''
        Substitutes the displacement operators of all QONN layers by the input values
        when the input reuploading is enabled in the QONN.
        
        :param inputs: Inputs to be re-uploaded in the displacement operators 
        '''
        S_dim = 2*self.N
        for l in range(self.layers-1, -1, -1):
            self.D_l[l, 0:len(inputs)] = np.sqrt(2) * inputs
            if l == (self.layers - 1):
                self.D_concat[l*S_dim:(l+1)*S_dim] = self.D_l[l].copy()
            else:
                self.D_concat[l*S_dim:(l+1)*S_dim] = self.D_l[l] + self.S_l[l] @ self.D_concat[(l+1)*S_dim:(l+2)*S_dim]
        
    def apply_linear_gaussian(D, mean_vector):
        '''
        Transforms the means vector of the Gaussian state by adding 
        the corresponding linear gaussian to it.
        
        :param D: Vector of displacements of the system's quadratures
        '''
        return mean_vector + D
        
    def apply_quadratic_gaussian(G, mean_vector, V):
        '''
        Transforms the means vector and the covariance matrix of the Gaussian state 
        given a quadratic Gaussian operator.
        
        :param G: Symplectic matrix of a quadratic Gaussian operator to be applied
        '''
        return G @ mean_vector, G @ V @ G.T
        
    def apply_gaussian_transformations(S_stack: jnp.ndarray,
                                    D_stack: jnp.ndarray,
                                    mean0:     jnp.ndarray,
                                    V0:        jnp.ndarray
                                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Runs all layers of the Gaussian net on (mean0, V0) via one XLA scan.
        
        S_stack.shape == (L, 2N, 2N)
        D_stack.shape == (L,   2N,)
        """

        def body(carry, sd):
            mean, cov = carry
            S, D      = sd        # shapes (2N,2N) and (2N,)

            # quadratic step
            mean, cov = QNN.apply_quadratic_gaussian(S, mean, cov)
            # linear (displacement) step
            mean      = QNN.apply_linear_gaussian(D, mean)

            return (mean, cov), None

        # scan over the leading axis of (S_stack, D_stack)
        (final_mean, final_cov), _ = lax.scan(
            body,
            (mean0, V0),
            (S_stack, D_stack),
        )
        return final_mean, final_cov

    def build_disp_coefs(self, D_concat):
        """
        Returns a 1‐D JAX complex array of shape (layers * N,),
        where each block of length N is
          D_concat[layer*2N : layer*2N + N] 
          + 1j * D_concat[layer*2N + N : layer*2N + 2N].
        """
        # 1) Reshape into (layers, 2*N)
        D = D_concat.reshape((self.layers, 2 * self.N))

        # 2) Split into real part (first N cols) and imag part (last N cols)
        real_part = D[:, :self.N]        # shape (layers, N)
        imag_part = D[:, self.N:]        # shape (layers, N)

        # 3) Form the complex displacement matrix
        disp_matrix = real_part + 1j * imag_part  # shape (layers, N), dtype=complex

        # 4) Flatten to a 1‐D array of length layers * N
        return disp_matrix.reshape((self.layers * self.N,))
        
    def compute_coefficients(self, S_concat, D_concat):
        d_r = self.build_disp_coefs(D_concat)
        d_i = jnp.conjugate(d_r)
        S_r = S_concat
        S_i = jnp.conjugate(S_r)

        def f_switch(i, S_r, S_i, d_r, d_i):
            # Internally compile all f’s as branches
            return lax.switch(i, self.unnorm_fns, S_r, S_i, d_r, d_i)
        
        batched_f = jax.vmap(f_switch, in_axes=(0, None, None, None, None))
        
        return batched_f(self.exp_vals_inds, S_r, S_i, d_r, d_i)
    
    def exp_val_ladder_jk(self, j, k, V):
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
        return 0.5*(V[j,k] - V[self.N+j, self.N+k] + 1j*(V[j, self.N+k] + V[self.N+j, k]))

    def exp_val_ladder_jdagger_k(self, j, k, V):
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
        delta = jnp.where(j == k, 1, 0)
        return 0.5*(V[j,k] + V[self.N+j, self.N+k] + 1j*(V[j, self.N+k] - V[self.N+j, k]) - delta)

    def exp_val_ladder_jdagger_kdagger(self, j, k, V):
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
        return 0.5*(V[j,k] - V[self.N+j, self.N+k] - 1j*(V[j, self.N+k] + V[self.N+j, k]))
    
    def compute_quad_exp_vals(self, V):
        '''
        Computes the expectation value of all combinations of ladder operators pairs of all existing modes
        of a Gaussian state based on its covariance matrix and means vector.
        
        :param V: Covariance matrix of the Gaussian state
        :param means: Means vector of the Gaussian state
        :return: All expectation values of the different configuration of a pair of ladder operators on all modes.
        '''
        js = jnp.arange(self.N)
        ks = jnp.arange(self.N)
        
        K00 = jax.vmap(jax.vmap(self.exp_val_ladder_jk, in_axes=(None, 0, None)), in_axes=(0, None, None))(js, ks, V)
        K01 = jax.vmap(jax.vmap(self.exp_val_ladder_jdagger_k, in_axes=(None, 0, None)), in_axes=(0, None, None))(js, ks, V)
        K10 = K01.T + jnp.eye(self.N, dtype=K01.dtype)
        K11 = jax.vmap(jax.vmap(self.exp_val_ladder_jdagger_kdagger, in_axes=(None, 0, None)), in_axes=(0, None, None))(js, ks, V)

        return jnp.stack([K00, K01, K10, K11], axis=0)
    
    def build_QNN_jax(self, parameters: jnp.ndarray):
        """
        Pure-functional build of all layer quantities via one lax.scan,
        using dynamic_slice and dynamic_update_slice for any dynamic indices.
        Returns (Q1_gauss, Q2_gauss, Z_gauss, S_l, S_concat, D_l, D_concat, trace_const).
        """
        N     = self.N
        L     = self.layers
        S_dim = 2 * N

        # Initial empty arrays
        Q1_gauss = jnp.zeros((L, S_dim, S_dim), dtype=jnp.complex128)
        Q2_gauss = jnp.zeros_like(Q1_gauss)
        Z_gauss  = jnp.zeros_like(Q1_gauss)
        S_l      = jnp.zeros_like(Q1_gauss)
        S_concat = jnp.zeros((S_dim, L * S_dim), dtype=jnp.complex128)
        D_l      = jnp.zeros((L, S_dim), dtype=jnp.complex128)
        D_concat = jnp.zeros((L * S_dim,), dtype=jnp.complex128)

        G   = jnp.eye(S_dim, dtype=jnp.complex128)
        idx = 0  # this will be a traced int

        init_carry = (idx, G,
                      Q1_gauss, Q2_gauss, Z_gauss,
                      S_l,      S_concat,
                      D_l,      D_concat)

        def body(carry, layer_idx):
            idx, G, Q1_g, Q2_g, Z_g, S_l_g, S_concat_g, D_l_g, D_concat_g = carry

            # 1) Bloch–Messiah decomposition for this layer
            Q1, Zmat, Q2, idx1 = self.build_quadratic_gaussians(parameters, idx)
            Q1_g = Q1_g.at[layer_idx].set(Q1)
            Q2_g = Q2_g.at[layer_idx].set(Q2)
            Z_g  = Z_g .at[layer_idx].set(Zmat)

            # 2) Net symplectic and update S_l
            SL      = Q2 @ Zmat @ Q1
            S_l_g   = S_l_g.at[layer_idx].set(SL)
            G_new   = SL @ G

            # 3) Build the Fock-space block and dynamically update S_concat
            blk     = self.u_bar.to_ladder_op(G_new)          # shape (S_dim, S_dim)
            start   = layer_idx * S_dim
            # dynamic_update_slice handles tracer‐valued start:
            S_concat_g = lax.dynamic_update_slice(S_concat_g, blk, (0, start))

            # 4) Linear displacements (if enabled)
            idx2 = idx1
            if not self.is_input_reupload:
                # dynamic slice of 2N parameters
                d_slice = lax.dynamic_slice(parameters,
                                            (idx1,),
                                            (2 * N,))
                idx2 = idx1 + 2 * N

                d_l   = jnp.sqrt(2.0) * d_slice           # (S_dim,)
                D_l_g = D_l_g.at[layer_idx].set(d_l)

                # back-recursion into D_concat
                next_start = (layer_idx + 1) * S_dim
                next_slice = lax.dynamic_slice(D_concat_g,
                                               (next_start,),
                                               (S_dim,))     # (S_dim,)
                # choose d_l for last layer, else d_l + SL @ next_slice
                val = jnp.where(layer_idx == L-1,
                                d_l,
                                d_l + SL @ next_slice)
                # dynamic_update_slice into the 1-D D_concat_g
                D_concat_g = lax.dynamic_update_slice(D_concat_g,
                                                      val,
                                                      (start,))

            new_carry = (idx2, G_new,
                         Q1_g, Q2_g, Z_g,
                         S_l_g, S_concat_g,
                         D_l_g, D_concat_g)
            return new_carry, None

        # scan over layer indices [0,1,...,L-1]
        (idx_final, G_final,
         Q1_gauss, Q2_gauss, Z_gauss,
         S_l,      S_concat,
         D_l,      D_concat), _ = lax.scan(
            body,
            init_carry,
            jnp.arange(L, dtype=jnp.int64),
        )

        # optional trace constant
        """ if self.observable == 'witness':
            trace_const = jnp.ones((N * (N + 1) + N,), dtype=jnp.complex128)
        else:
            trace_const = jnp.empty((0,), dtype=jnp.complex128) """

        return (Q1_gauss, Q2_gauss, Z_gauss,
                S_l,       S_concat,
                D_l,       D_concat,)
                #trace_const)
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_QNN(self, params, input):
        '''
        Evaluates the QONN for a given input.
        
        :param input: Input to be predicted by the QONN
        :return: EXpectation values of the observables related to QONN outputs
        '''
        # 0. Build the QONN components using the tunable parameters
        #build_start = time.time()
        Q1_gauss, Q2_gauss, Z_gauss, S_l, S_concat, D_l, D_concat = self.build_QNN_jax(params)
        #self.qnn_profiling.build_qnn_times.append(time.time() - build_start)
        
        # 1. Prepare initial state: initial vacuum state displaced according to the inputs
        #input_prep_start = time.time()
        V = 0.5*jnp.eye(2*self.N, dtype=jnp.complex128)
        mean_vector = jnp.zeros((2*self.N,), dtype=jnp.complex128)
        mean_vector = QNN.apply_linear_gaussian(jnp.sqrt(2) * input, mean_vector)
        
        # 1.1. When using input reuploading: build displacement with inputs
        if self.is_input_reupload:
            self.build_reuploading_disp(input)
        #self.qnn_profiling.input_prep_times.append(time.time() - input_prep_start)

        # 2. Apply the Gaussian transformation acting as weights matrix and bias vector
        #gauss_start = time.time()
        #mean_vector, V = self.apply_gaussian_transformations(mean_vector, V)
        mean_vector, V = QNN.apply_gaussian_transformations(S_l, D_l, mean_vector, V)
        #self.qnn_profiling.gauss_times.append(time.time() - gauss_start)
        
        # 3. Compute the expectation values of all possible ladder operators pairs over the final Gaussian state
        #K_exp_vals_start = time.time()
        K_exp_vals = self.compute_quad_exp_vals(V)
        #self.qnn_profiling.K_exp_vals_times.append(time.time() - K_exp_vals_start)

        # 4. Compute coefficients for trace expression and normalization terms
        #ladder_superpos_start = time.time()
        traces_terms_coefs = self.compute_coefficients(S_concat, D_concat)
        #self.qnn_profiling.ladder_superpos_times.append(time.time() - ladder_superpos_start)
        
        # 5. Compute the expectation values acting as outputs
        #nongauss_start = time.time()
        exp_vals = self.compute_exp_val_loop(traces_terms_coefs, K_exp_vals, mean_vector)
        #self.qnn_profiling.nongauss_times.append(time.time() - nongauss_start)
        
        # 6. Multiply by trace coefficients and normalize (last expectation value)
        return self.trace_const * exp_vals[:-1] / exp_vals[-1]
    
    def single_ladder_exp_val(self, trace_idx, tr_term_idx, means_vector):
        '''
        Computes the expectation value of a single ladder operator over a certain mode based
        on the means vector (position and momentum expectation values).
        
        :param means_vector: Expectation values of position and momentum of each mode (xxpp order)
        :return: Normalized ladder operator expectation value
        '''
        term_mode = self.jax_modes[trace_idx][tr_term_idx][0]
        term_type = self.jax_types[trace_idx][tr_term_idx][0]
        return self.oneoversqrt2 * (means_vector[term_mode] + 1j*(-2*term_type + 1)*means_vector[self.N+term_mode])
    
    def wick_expansion_expval(self, trace_idx, tr_term_idx, quadratic_exp_vals, means_vector):
        """
        Vectorized over all perfect‐matchings for a given (trace_idx, tr_term_idx).
        """

        # 1) pull out the block of matchings for this trace
        #    shape (Pmax, max_pm_len)
        lpms_idx = self.jax_lens[trace_idx][tr_term_idx] - 2
        all_pms = self.jax_lpms[lpms_idx]

        # 2) reshape into two arrays of shape (Pmax, Kmax):
        #    even positions 0,2,4... are i1; 1,3,5... are i2
        p1 = all_pms[:, ::2]   # shape (Pmax, Kmax)
        p2 = all_pms[:, 1::2]  # shape (Pmax, Kmax)

        # 3) pull out the per‐term mode/type vectors (shape (Kmax,))
        modes_row = self.jax_modes[trace_idx, tr_term_idx]  # (Kmax,)
        types_row = self.jax_types[trace_idx, tr_term_idx]  # (Kmax,)

        def term_prod(p1_row, p2_row):
            def valid_pms(_):
                # p1_row, p2_row: shape (Kmax,)
                # 3a) valid‐pair mask
                valid = p1_row >= 0 # False for matchings with -1

                # 3b) gather mode/type indices
                m1 = modes_row[p1_row.clip(0)]  # clip to avoid OOB on padding
                m2 = modes_row[p2_row.clip(0)]
                t1 = types_row[p1_row.clip(0)]
                t2 = types_row[p2_row.clip(0)]

                # 3c) compute covariance‐based term and mean‐based term
                cov_term  = quadratic_exp_vals[t1 + 2*t2, m1, m2]
                mean_term = self.oneoversqrt2 * (means_vector[m1] + 1j*(-2*t1 + 1)*means_vector[self.N+m1])

                # 3d) select: equal‐index pairs use mean_term, unequal use cov_term
                pair_val = jnp.where(p1_row != p2_row, cov_term, mean_term)

                # 3e) mask out padding positions: set to 1 so prod ignores them
                pair_val = jnp.where(valid, pair_val, 1.0 + 0.0j)

                # 3f) product across Kmax
                return jnp.prod(pair_val)
            
            def null_pm(_):
                return 0 + 0j
            
            return lax.cond(
                p1_row[0] == -1,
                null_pm,
                valid_pms,
                operand=None
            )

        # 4) vmapped over all Pmax matchings (axis 0)
        prods = jax.vmap(term_prod, in_axes=(0,0))(p1, p2)  # shape (Pmax,)

        # 5) sum over all matchings
        finalsum = jnp.sum(prods)
        return finalsum
    
    def get_expectation_value(self, trace_idx, tr_term_idx, quadratic_exp_vals, means_vector):
        '''
        Dispatches the expectation value calculation method based on the number of
        terms. Cases: no operators (pure states - return 1), one operator (use means vector), 
        two or more operators (Wick's expansion based on loop perfect matchings).
        
        :param quadratic_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
        :param means_vector: Position and momentum expectation values of all modes (xxpp order)
        :return: Expectation value of the provided expression
        '''
        len_term = self.jax_lens[trace_idx, tr_term_idx]

        # case0: Tr[ρ] → return 1
        def case0(_):
            return 1.0+0.0j   # or jnp.array(1.0, dtype=...)

        # case1: single ladder → call single_ladder_exp_val
        def case1(_):
            mode = self.jax_modes[trace_idx, tr_term_idx, 0]
            typ  = self.jax_types[trace_idx, tr_term_idx, 0]
            #return self.single_ladder_exp_val(trace_idx, tr_term_idx, means_vector) #FIXME: Indices fixed to 0 but not for PMs
            return self.oneoversqrt2 * (means_vector[mode] + 1j*(-2*typ + 1)*means_vector[self.N+mode])

        # case2: wick expansion
        def case2(_):
            return self.wick_expansion_expval(
                trace_idx, tr_term_idx, quadratic_exp_vals, means_vector
            )

        # first branch off len_term == 0 vs len_term > 0
        return lax.cond(
            len_term == 0,
            case0,
            lambda _: lax.cond(
                len_term == 1,
                case1,
                case2,
                operand=None
            ),
            operand=None
        )
        
    def compute_terms_in_trace(self, trace_idx, coefs, quadratic_exp_vals, means_vector):
        '''
        Computes each term in the trace by calculating the term expression's expectation value
        and multiplying it by its corresponding coefficient.
        
        :param coefs: Terms' coefficients
        :param quadratic_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
        :param means_vector: Position and momentum expectation values of all modes (xxpp order)
        :return: Final expectation value of the output
        '''
        expvals = jax.vmap(self.get_expectation_value, in_axes=(None, 0, None, None))(trace_idx, self.trace_terms_ranges[trace_idx], quadratic_exp_vals, means_vector)
        tr_value = jnp.sum(coefs * expvals)
        return tr_value
    
    def compute_exp_val_loop(self, terms_coefs, quadratic_exp_vals, means_vector):
        '''
        Computes the expectation value of the expression (trace) that defines the QNN operations.
        
        :param N: Total number of system's modes
        :param terms_coefs: Coefficients of the unnormalized terms of the trace expression
        :param norm_coefs: Coefficients of the normalization trace expression
        :param quadratic_exp_vals: Expectation values of all combinations of ladder operators pairs over system's modes
        :param means_vector: Position and momentum expectation values of all modes (xxpp order)
        :return: Normalized expectation value of each output of the QNN
        '''
        exp_vals = jax.vmap(self.compute_terms_in_trace, in_axes=(0, 0, None, None))(self.exp_vals_inds, terms_coefs, quadratic_exp_vals, means_vector)
        return exp_vals
    
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
        # BATCH EVALUATION
        epoch_start_time = time.time()
        shuffled_inds = np.random.permutation(len(inputs_dataset))
        shuffled_inputs_dataset = inputs_dataset[shuffled_inds]
        t0_lower = time.time()
        lowered = self.eval_QNN.lower(self, parameters, shuffled_inputs_dataset[0])
        #print("===== StableHLO =====")
        #print(lowered.compiler_ir(dialect='stablehlo'))
        ir_time = time.time() - t0_lower
        print("IR Time: ", ir_time)
        input('w7')
        hlo = lowered.compiler_ir(dialect="hlo")
        print(f"HLO text is {len(hlo.as_hlo_text())/1e6} MB")
        input('w8')
        
        t0_compile = time.time()
        compiled = lowered.compile()
        compile_time = time.time() - t0_compile
        print("First compile time: ", compile_time)
        """ qnn_outputs = np.real_if_close(
            jax.vmap(self.eval_QNN, in_axes=(None, 0))(parameters, shuffled_inputs_dataset), tol=1e6
        ) """
        self.qnn_profiling.epoch_times.append(time.time() - epoch_start_time)
        
        return loss_function(outputs_dataset[shuffled_inds], qnn_outputs)

    """ def train_QNN(self, parameters, inputs_dataset, outputs_dataset, loss_function):
        '''
        Evaluates all dataset items with the QONN current state and computes the loss function 
        values for each item. This function is the one to be minimized and refers to one epoch.
        
        :param parameters: QONN tunable parameters
        :param inputs_dataset: Inputs of the dataset to be learned
        :param outputs_dataset: Outputs to be learned
        :param loss_function: Function to evaluate the loss of the QONN predictions
        :return: Losses of the QONN predictions
        '''
        # BATCH EVALUATION
        self.tunable_parameters = jnp.copy(parameters)
        epoch_start_time = time.time()
        shuffled_inds = np.random.permutation(len(inputs_dataset))
        shuffled_inputs_dataset = inputs_dataset[shuffled_inds]
        qnn_outputs = np.real_if_close(
            jax.vmap(self.eval_QNN, in_axes=(None, 0))(parameters, shuffled_inputs_dataset), tol=1e6
        )
        self.qnn_profiling.epoch_times.append(time.time() - epoch_start_time)
        
        return loss_function(outputs_dataset[shuffled_inds], qnn_outputs) """
    
    def train_symp_rank(self, parameters):
        '''
        Tunes the QONN parameters in order to maximize the symplectic rank, 
        i.e. to maximize the non-Gaussianity in a pure quantum system.
        
        :param parameters: QONN tunable parameters
        :return: Symplectic eigenvalues of the final QONN state
        '''
        inputs = parameters[-2*self.N:]
        witness_vals = self.eval_QNN(parameters[:-self.N], inputs)
        
        s, V_rec = reconstruct_stats(witness_vals, self.N)
        symp_eigvals = np.real_if_close(symplectic_eigenvals(V_rec))
        abs_smyp_eigvals = np.abs(symp_eigvals)
        print(abs_smyp_eigvals)
        print('SUM: ', np.sum(abs_smyp_eigvals))
        print('0.5 EIGVAL: ', np.sum(np.isclose(symp_eigvals, 0.5, atol=1e-4)))
        symp_rank = np.sum(0.5 - abs_smyp_eigvals)
        return symp_rank
    
    def print_qnn(self):
        '''
        Prints all QONN items needed to physically build it.
        '''
        print(f"PARAMETERS:\n{self.tunable_parameters}")
        for layer in range(self.layers):
            print(f"=== LAYER {layer+1} ===\nQ1 = {self.Q1_gauss[layer]}\nZ = {self.Z_gauss[layer]}\nQ2 = {self.Q2_gauss[layer]}")
            print(f"Symplectic matrix:\n{self.S_l[layer]}")
            check_symp_orth(self.S_l[layer])
            if not self.is_input_reupload:
                print(f"Displacement vector:\n{self.D_l[layer]}")
            print(f"Symplectic coefficients:\n{self.S_concat[:,layer*2*self.N : (layer+1)*2*self.N]}")
            if not self.is_input_reupload:
                print(f"Displacement coefficients:\n{self.D_concat[layer*2*self.N : (layer+1)*2*self.N]}")
        
    def test_model(self, testing_dataset, loss_function):
        '''
        Makes predictions of the given QNN using the input testing dataset.
        
        :param qnn: QNN to be tested
        :param testing_dataset: List of inputs and outputs to be tested
        :param loss_function: Loss function to compute the testing set losses
        :return: QNN predictions of the testing set
        '''
        test_inputs = reduce(lambda x, func: func(x), self.in_preprocessors, testing_dataset[0])
        test_outputs = reduce(lambda x, func: func(x), self.out_preprocessors, testing_dataset[1])
        
        # Evaluate all testing set
        qnn_outputs = np.real_if_close(
            jax.vmap(self.eval_QNN, in_axes=(None, 0))(self.tunable_parameters, test_inputs), tol=1e6
        )
        
        mean_error = loss_function(test_outputs, qnn_outputs)
        print(f"LOSS VALUE FOR TESTING SET: {mean_error}")
        
        return reduce(lambda x, func: func(x), self.postprocessors, qnn_outputs)
    
    def save_model(self, filename):
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
    
