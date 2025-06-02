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
loss_values = []
best_loss_values = [10]

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

    def avg_benchmark(self):
        self.avg_times = {}
        self.avg_times["Build QNN"] = sum(self.build_qnn_times)/len(self.build_qnn_times)
        self.avg_times["Input prep"] = sum(self.input_prep_times)/len(self.input_prep_times)
        self.avg_times["Gaussian op"] = sum(self.gauss_times)/len(self.gauss_times)
        self.avg_times["Pairs exp-vals"] = sum(self.K_exp_vals_times)/len(self.K_exp_vals_times)
        self.avg_times["Non-gauss coefs"] = sum(self.ladder_superpos_times)/len(self.ladder_superpos_times)
        self.avg_times["Non-gaussianity"] = sum(self.nongauss_times)/len(self.nongauss_times)
        return self.avg_times
    
    def clear_times(self):
        self.build_qnn_times = []
        self.input_prep_times = []
        self.gauss_times = []
        self.K_exp_vals_times = []
        self.ladder_superpos_times = []
        self.nongauss_times = []
    
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
        
        # Symplectic matrix and displacement vectors for each layer
        self.S_l = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.D_l = np.zeros((self.layers, 2*self.N))
        # Bloch-Messiah decomposition of the symplectic matrix for each layer
        self.Q1_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Q2_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Z_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        # Concatenation of symplectic matrix and displacement vectors of each layer
        self.S_concat = np.zeros((2*self.N, 2*self.layers*self.N), dtype='complex')
        self.D_concat = np.zeros((2*self.layers*self.N))
        # Final Gaussian transformation when commuting with photon additions (product of all Gaussians)
        self.G = np.eye(2*self.N)
        
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
        
        # Compile trace expression terms with respect to symplectic and displacement parameters
        t1 = time.time()
        self.nb_num_unnorm = []
        for outs in range(len(self.unnorm_expr_terms_out)):
            self.nb_num_unnorm.append([jax.jit(lambdify((S_r, S_i, d, d_conj), unnorm_trm, modules={'jax': jnp, 'numpy': jnp})) for unnorm_trm in self.unnorm_expr_terms_out[outs]])
        t2 = time.time()
        print(f"Time to lambdify trace terms: {t2-t1}")
        
        self.exp_vals_inds = jnp.arange(len(self.jax_modes), dtype=jnp.int32)
        self.num_terms_in_trace = jnp.array([len(output_terms) for output_terms in self.modes])
        trace_terms_rngs, _ = to_np_array([[[i for i in range(num_terms)] for num_terms in self.num_terms_in_trace]])
        self.trace_terms_ranges = jnp.array(trace_terms_rngs[0])
        print(self.num_terms_in_trace)
        print(self.trace_terms_ranges)
        input("sdaf")
        #self.trace_terms_ranges = jnp.array([jnp.array([i for i in range(num_terms)]) for num_terms in self.num_terms_in_trace])
        #uniq_num_terms = jnp.unique(self.num_terms_in_trace)
        #self.terms_ranges = {int(i): jnp.arange(i, dtype=jnp.int32) for i in uniq_num_terms}
        #self.trace_terms_ranges = {int(trace): self.terms_ranges[int(self.num_terms_in_trace[int(trace)])] for trace in self.exp_vals_inds}
        
    def build_symp_orth_mat(self, parameters):
        '''
        Creates a symplectic-orthogonal (unitary) matrix with the complex exponential 
        of the most general Hermitian matrix built by N**2 parameters.
        
        :param parameters: Parameters to create the Hermitian matrix for the final SO matrix
        :return: Symplectic-orthogonal matrix made to be applied over the quadratures of the system
        '''
        H = general_hermitian_matrix(parameters, self.N)
        U = unitary_from_hermitian(H)
        return np.real_if_close(self.u_bar.to_canonical_op(U))
        
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
        sqz_parameters = np.e**np.abs(parameters[current_par_idx : current_par_idx + self.N])
        sqz_inv = 1.0/sqz_parameters
        Z = np.diag(np.concatenate((sqz_parameters, sqz_inv)))
        current_par_idx += self.N
        
        return Q1, Z, Q2, current_par_idx

    def build_QNN(self, parameters):
        '''
        Fills the QONN numerical components using the vector of tunable parameters.
        
        :param parameters: Vector of trainable parameters
        '''
        self.tunable_parameters = jnp.copy(parameters)
        S_dim = 2*self.N
        self.G = np.eye(S_dim)
        current_par_idx = 0
        for l in range(self.layers-1, -1, -1):
            # Build symplectic-orthogonal (unitary) matrices and diagonal symplectic matrix
            self.Q1_gauss[l], self.Z_gauss[l], self.Q2_gauss[l], current_par_idx = self.build_quadratic_gaussians(parameters, current_par_idx)

            # Build final quadratic Gaussian transformation
            self.S_l[l] = self.Q2_gauss[l] @ self.Z_gauss[l] @ self.Q1_gauss[l]
            self.G = self.S_l[l] @ self.G
            # Build concatenated symplectic matrices in Fock space for trace expressions' coefficients
            self.S_concat[:, l*S_dim:(l+1)*S_dim] = self.u_bar.to_ladder_op(self.G)
            
            # Build displacements (linear Gaussian)
            if not self.is_input_reupload:
                self.D_l[l] = np.sqrt(2) * parameters[current_par_idx : current_par_idx + 2*self.N]
                current_par_idx += 2*self.N
                if l == (self.layers - 1):
                    self.D_concat[l*S_dim:(l+1)*S_dim] = self.D_l[l].copy()
                else:
                    self.D_concat[l*S_dim:(l+1)*S_dim] = self.D_l[l] + self.S_l[l] @ self.D_concat[(l+1)*S_dim:(l+2)*S_dim]
        if self.observable == 'witness':
            self.trace_const = np.array([1, 1, parameters[current_par_idx]])
            current_par_idx += 1
    
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
        
    def apply_linear_gaussian(self, D):
        '''
        Transforms the means vector of the Gaussian state by adding 
        the corresponding linear gaussian to it.
        
        :param D: Vector of displacements of the system's quadratures
        '''
        self.mean_vector[0:len(D)] += D
        
    def apply_quadratic_gaussian(self, G):
        '''
        Transforms the means vector and the covariance matrix of the Gaussian state 
        given a quadratic Gaussian operator.
        
        :param G: Symplectic matrix of a quadratic Gaussian operator to be applied
        '''
        self.V = G @ self.V @ G.T
        self.mean_vector = G @ self.mean_vector
        
    def apply_gaussian_transformations(self):
        '''
        Applies all Gaussian transformations of the QONN to the initial quantum state.
        '''
        for l in range(self.layers):
            self.apply_quadratic_gaussian(self.S_l[l])
            self.apply_linear_gaussian(self.D_l[l])

    def build_disp_coefs(self):
        '''
        Creates the complex vector of displacements of each mode from the displacement parameters.
        '''
        d = np.zeros((self.layers * self.N), dtype='complex')
        for l in range(self.layers):
            for i in range(self.N):
                d[l*self.N + i] = self.D_concat[l*2*self.N + i] + 1j*self.D_concat[l*2*self.N + self.N+i]
        return d
    
    def compute_coefficients(self):
        '''
        Calculates the coefficients related to each term in the trace and the normalization expressions
        that appear after the commutation of the non-Gaussian operators 
        using the displacement and the symplectic coefficients of the Gaussian operators.
        '''
        # Build displacement complex vector & its conjugate
        d_r = self.build_disp_coefs()
        d_i = np.conjugate(d_r)
        # Transform symplectic matrix to ladder basis & compute its transpose conjugate
        S_r = self.S_concat
        S_i = np.conjugate(S_r)

        # Values of trace terms
        trace_vals = np.zeros((len(self.nb_num_unnorm), len(self.nb_num_unnorm[0])), dtype='complex')
        for out_idx in range(len(self.num_terms_per_trace)):
            if len(self.nb_num_unnorm[out_idx]) == 0:
                trace_vals[out_idx, 0] = 1
            for idx in range(self.num_terms_per_trace[out_idx]):
                trace_vals[out_idx, idx] = self.nb_num_unnorm[out_idx][idx](S_r, S_i, d_r, d_i)
        return trace_vals
    
    def eval_QNN(self, input):
        '''
        Evaluates the QONN for a given input.
        
        :param input: Input to be predicted by the QONN
        :return: EXpectation values of the observables related to QONN outputs
        '''
        # 1. Prepare initial state: initial vacuum state displaced according to the inputs
        input_prep_start = time.time()
        self.V = 0.5*np.eye(2*self.N)
        self.mean_vector = np.zeros(2*self.N)
        self.apply_linear_gaussian(np.sqrt(2) * input)
        # 1.1. When using input reuploading: build displacement with inputs
        if self.is_input_reupload:
            self.build_reuploading_disp(input)
        self.qnn_profiling.input_prep_times.append(time.time() - input_prep_start)

        # 2. Apply the Gaussian transformation acting as weights matrix and bias vector
        gauss_start = time.time()
        self.apply_gaussian_transformations()
        self.qnn_profiling.gauss_times.append(time.time() - gauss_start)
        
        # 3. Compute the expectation values of all possible ladder operators pairs over the final Gaussian state
        K_exp_vals_start = time.time()
        K_exp_vals = compute_K_exp_vals(self.V, self.mean_vector)
        self.qnn_profiling.K_exp_vals_times.append(time.time() - K_exp_vals_start)

        # 4. Compute coefficients for trace expression and normalization terms
        ladder_superpos_start = time.time()
        traces_terms_coefs = self.compute_coefficients()
        self.qnn_profiling.ladder_superpos_times.append(time.time() - ladder_superpos_start)
        """ print("TRACE TERMS COEFS:")
        print(traces_terms_coefs)
        print("QUADRATIC EXP VALS:")
        print(K_exp_vals)
        print("TRACES LENS:")
        print(self.jax_lens) """
        # 5. Compute the expectation values acting as outputs
        nongauss_start = time.time()
        unnorm_val = self.compute_exp_val_loop(traces_terms_coefs, K_exp_vals, self.mean_vector)
        self.qnn_profiling.nongauss_times.append(time.time() - nongauss_start)
        
        print("UNNORMALIZED EXPECTATION VALUES:")
        #print(unnorm_val[:-1])
        print(unnorm_val[1:-1])
        
        print("NORMALIZED EXPECTATION VALUES:")
        #print(unnorm_val[:-1] / unnorm_val[-1])
        print(unnorm_val[1:-1] / unnorm_val[0])
        
        print("NORM")
        print(unnorm_val[-1])
            
        print("MEANS VECTOR AND COV MAT OF THE GAUSSIAN STATE:")
        print(self.mean_vector)
        print(np.round(self.V, 4))
        check_uncertainty_pple(self.V)
        symplectic_eigenvals(self.V)
        print()
        
        #s, V = reconstruct_stats(unnorm_val[:-1] / unnorm_val[-1], self.N)
        s, V = reconstruct_stats(unnorm_val[1:-1]/unnorm_val[0], self.N)
        print("FINAL STATE MEANS AND COV MAT:")
        print(s)
        print(np.round(np.real_if_close(V), 4))
        check_uncertainty_pple(V)
        symplectic_eigenvals(V)
        print(self.qnn_profiling.avg_benchmark())
        input("ASDF")
        return self.trace_const * np.real_if_close(unnorm_val[:-1] / unnorm_val[-1], tol=1e6)
    
    #==================
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
        tr_value = jnp.vdot(coefs, expvals)
        return tr_value
    
    @partial(jax.jit, static_argnums=(0,))
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
    
    #=====================

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
        build_start = time.time()
        self.build_QNN(parameters)
        self.qnn_profiling.build_qnn_times.append(time.time() - build_start)
        
        shuffle_indices = np.random.permutation(len(inputs_dataset))
        qnn_outputs = np.full_like(outputs_dataset, 0)
        for dataset_idx in shuffle_indices:
            qnn_outputs[dataset_idx] = self.eval_QNN(inputs_dataset[dataset_idx])
        return loss_function(outputs_dataset, qnn_outputs)
    
    def train_ent_witness(self, parameters):
        '''
        Tunes the QONN parameters in order to minimize an entanglement witness 
        aiming for maximally entangled states.
        
        :param parameters: QONN tunable parameters
        :return: Entanglement witness expectation value
        '''
        #input_disp = parameters[0:self.N] # TODO: Consider parameterize momentum too
        
        build_start = time.time()
        self.build_QNN(parameters)
        self.qnn_profiling.build_qnn_times.append(time.time() - build_start)
        
        witness_vals = self.eval_QNN(parameters[-self.N:])
        #print("VAL")
        #print(witness_vals[0] - witness_vals[1]*witness_vals[2])
        #self.print_qnn()
        
        return -(witness_vals[0] - witness_vals[1]*witness_vals[2])
    
    def print_qnn(self):
        '''
        Prints all QONN items needed to physically build it.
        '''
        print(f"PARAMETERS:\n{self.tunable_parameters}")
        for layer in range(self.layers):
            print(f"=== LAYER {layer+1} ===\nQ1 = {self.Q1_gauss[layer]}\nZ = {self.Z_gauss[layer]}\nQ2 = {self.Q2_gauss[layer]}")
            print(f"Symplectic matrix:\n{self.S_l[layer]}")
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
        
        #error = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
        qnn_outputs = np.full_like(test_outputs, 0)
        
        # Evaluate all testing set
        for k in range(len(test_inputs)):
            qnn_outputs[k] = self.eval_QNN(test_inputs[k])
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
    
