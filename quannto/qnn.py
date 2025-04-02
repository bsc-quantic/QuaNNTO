import numpy as np
import time
import jsonpickle
import time
import scipy.optimize as opt
from sympy import lambdify
from functools import partial, reduce

from .utils import *
from .expectation_value import *
from .results_utils import *
from .loss_functions import mse

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
        self.trace_const = 1 if observable=='number' else (1/np.sqrt(2)) if observable=='position' else (1j/np.sqrt(2)) if observable=='momentum' else 0
        
        # Full expectation value expression of the wavefunction (photon additions + observable to be measured)
        self.trace_expr = complete_trace_expression(self.N, layers, photon_add, self.n_out, include_obs=True, obs=observable)
        # Normalization expression of the wavefunction related to photon additions
        self.norm_trace_expr = complete_trace_expression(self.N, layers, photon_add, self.n_out, include_obs=False)
        
        # Extract ladder operators terms from expectation value expression
        self.modes, self.types = [], []
        for outs in range(len(self.trace_expr)):
            modes, types = extract_ladder_expressions(self.trace_expr[outs])
            self.modes.append(modes)
            self.types.append(types)
        self.np_modes, self.lens_modes = to_np_array(self.modes)
        self.np_types, self.lens_types = to_np_array(self.types)
        
        # Extract ladder operators terms from normalization expression
        modes_norm, types_norm = extract_ladder_expressions(self.norm_trace_expr)
        self.modes_norm, self.types_norm = [modes_norm], [types_norm]
        self.np_modes_norm, self.lens_modes_norm = to_np_array(self.modes_norm)
        self.np_types_norm, self.lens_types_norm = to_np_array(self.types_norm)
        
        # Compute all needed loop perfect matchings for the Wick's expansion of the trace expression
        max_lpms = np.max(self.lens_modes)
        self.lpms = [to_np_array([loop_perfect_matchings(lens)]) for lens in range(2, max_lpms+1)]
        if len(self.lpms) == 0:
            self.lpms = [to_np_array([loop_perfect_matchings(2)])]
        self.np_lpms = nb.typed.List([lpms for (lpms, _) in self.lpms])
        self.lens_lpms = nb.typed.List([lens for (_, lens) in self.lpms])
        
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
        print(f"Trace terms per output: {len(self.unnorm_expr_terms_out[0])}")
        
        norm_expr_terms = list(self.norm_trace_expr.args)
        self.norm_subs_expr_terms = []
        for norm_term in norm_expr_terms:
            new_term = norm_term.subs(ladder_subs)
            self.norm_subs_expr_terms.append(new_term)
        print(f"Normalization factor terms: {len(self.norm_subs_expr_terms)}")
        
        d = symbols(f'r0:{layers*N}', commutative=True)
        d_conj = symbols(f'i0:{layers*N}', commutative=True)
        dim = 2*N
        S_r = MatrixSymbol('S_r', dim, layers*dim)
        S_i = MatrixSymbol('S_i', dim, layers*dim)
        
        # Compile trace expression terms with respect to symplectic and displacement parameters
        t1 = time.time()
        self.nb_num_unnorm = []
        for outs in range(len(self.unnorm_expr_terms_out)):
            self.nb_num_unnorm.append([nb.njit(lambdify((S_r, S_i, d, d_conj), unnorm_trm, modules='numpy')) for unnorm_trm in self.unnorm_expr_terms_out[outs]])
        t2 = time.time()
        print(f"Time to lambdify unnormalized terms: {t2-t1}")
        self.nb_num_norm = [nb.njit(lambdify((S_r, S_i, d, d_conj), norm_trm, modules='numpy')) for norm_trm in self.norm_subs_expr_terms]
        t3 = time.time()
        print(f"Time to lambdify normalization terms: {t3-t2}")
        
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
        self.tunable_parameters = np.copy(parameters)
        S_dim = 2*self.N
        self.G = np.eye(S_dim)
        current_par_idx = 0
        for l in range(self.layers-1, -1, -1):
            # Build symplectic-orthogonal (unitary) matrices and diagonal symplectic matrix
            self.Q1_gauss[l], self.Z_gauss[l], self.Q2_gauss[l], current_par_idx = self.build_quadratic_gaussians(parameters, current_par_idx)

            # Build final quadratic Gaussian transformation
            self.S_l[l] = self.Q2_gauss[l] @ self.Z_gauss[l] @ self.Q1_gauss[l]
            self.G = self.S_l[l] @ self.G
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
            self.trace_const = parameters[current_par_idx]
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
        # Transform symplectic matrix to ladder basis & compute its conjugate
        S_r = self.S_concat
        S_i = np.conjugate(S_r)
        
        # Values of normalization terms
        norm_vals = np.zeros((len(self.nb_num_norm)), dtype='complex')
        for idx in range(len(self.modes_norm[0])):
            norm_vals[idx] = self.nb_num_norm[idx](S_r, S_i, d_r, d_i)
        
        # Values of trace terms
        trace_vals = np.zeros((len(self.nb_num_unnorm), len(self.nb_num_unnorm[0])), dtype='complex')
        for out_idx in range(len(self.modes)):
            for idx in range(len(self.modes[out_idx])):
                trace_vals[out_idx, idx] = self.nb_num_unnorm[out_idx][idx](S_r, S_i, d_r, d_i)
        return trace_vals, norm_vals
    
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
        terms_coefs, norm_coefs = self.compute_coefficients()
        self.qnn_profiling.ladder_superpos_times.append(time.time() - ladder_superpos_start)
        
        # 5. Compute the expectation values acting as outputs
        nongauss_start = time.time()
        norm_exp_val = compute_exp_val_loop(self.N, terms_coefs, norm_coefs,
                                            self.np_modes, self.np_types, self.lens_modes,
                                            self.np_modes_norm, self.np_types_norm, self.lens_modes_norm, 
                                            self.np_lpms, K_exp_vals, self.mean_vector)
        self.qnn_profiling.nongauss_times.append(time.time() - nongauss_start)
        
        return self.trace_const * np.real_if_close(norm_exp_val, tol=1e6)

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
    
def test_model(qnn, testing_dataset, loss_function):
    '''
    Makes predictions of the given QNN using the input testing dataset.
    
    :param qnn: QNN to be tested
    :param testing_dataset: List of inputs and outputs to be tested
    :param loss_function: Loss function to compute the testing set losses
    :return: QNN predictions of the testing set
    '''
    test_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, testing_dataset[0])
    test_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, testing_dataset[1])
    
    #error = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
    qnn_outputs = np.full_like(test_outputs, 0)
    
    # Evaluate all testing set
    for k in range(len(test_inputs)):
        qnn_outputs[k] = qnn.eval_QNN(test_inputs[k])
    mean_error = loss_function(test_outputs, qnn_outputs)
    print(f"LOSS VALUE FOR TESTING SET: {mean_error}")
    
    return reduce(lambda x, func: func(x), qnn.postprocessors, qnn_outputs)
    
def build_and_train_model(name, N, layers, n_inputs, n_outputs, photon_additions, observable, is_input_reupload, 
                          train_set, valid_set, loss_function=mse, hopping_iters=2, in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    '''
    Creates and trains a QNN model with the given hyperparameters and dataset by optimizing the 
    tunable parameters of the QNN.
    
    :param name: Name of the model
    :param N: Number of neurons per layer (modes of the quantum system)
    :param layers: Number of layers
    :param n_inputs: Number of QONN inputs
    :param n_outputs: Number of QONN outputs
    :param photon_additions: Photon additions over system modes' that are desired (per layer)
    :param observable: Name of the observable to be measured ('position', 'momentum', 'number' or 'witness')
    :param is_input_reupload: Boolean variable telling whether the QONN has to have input-reuploading or not
    :param train_set: List of inputs and outputs to be learned
    :param valid_set: Validation set to be evaluated every epoch testing the generalization of the QONN
    :param loss_function: Function that computes the loss between the predicted and the expected value (default 'mse')
    :param hopping_iters: Number of Basinhopping iterations
    :param in_preprocs: List of preprocessors for the dataset inputs
    :param out_preprocs: List of preprocessors for the dataset outputs
    :param postprocs: List of postprocessors (postprocessing of QONN yielded data)
    :param init_pars: Initialization parameters for the QNN
    :param save: Boolean determining whether to save the model (default=True)
    :return: Trained QNN model
    '''
    if type(init_pars) == type(None):
        #init_pars = np.random.rand(2*((N**2 + N) //2) + 3*N + layers*(2*((N**2 + N) // 2) + N)) if is_input_reupload else np.random.rand(2*((N**2 + N) //2) + 3*N + layers*(2*((N**2 + N) // 2) + N + 2*N))
        #init_pars = np.random.rand(2*N**2 + 3*N + layers*(2*N**2 + N)) if is_input_reupload else np.random.rand(2*N**2 + 3*N + layers*(2*N**2 + 3*N))
        n_pars = layers*(2*N**2 + N) if is_input_reupload else layers*(2*N**2 + 3*N)
        init_pars = np.random.rand(n_pars) if observable!='witness' else np.random.rand(n_pars + 1)
    else:
        aux_pars = 0 if observable!='witness' else 1
        if is_input_reupload:
            #assert len(init_pars) == 2*((N**2 + N) //2) + N + layers*(2*((N**2 + N) // 2) + N)
            #assert len(init_pars) == 2*N**2 + N + layers*(2*N**2 + N)
            assert len(init_pars) == layers*(2*N**2 + N) + aux_pars
        else:
            #assert len(init_pars) == 2*((N**2 + N) //2) + 3*N + layers*(2*((N**2 + N) // 2) + N + 2*N)
            assert len(init_pars) == layers*(2*N**2 + 3*N) + aux_pars
    
    def callback(xk):
        '''
        Prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QONN tunable parameters
        '''
        e = training_QNN(xk)
        print(f'Training loss: {e}')
        val_e = validate_QNN(xk)
        print(f'Validation loss: {val_e}')
        loss_values.append(e)
        validation_loss.append(val_e)
        
    def callback_hopping(x,f,accept):
        '''
        Prints the best Basinhopping error achieved so far and the current one. 
        If the current error is less than the best one so far, it substitutes the best loss values.
        
        :param x: QONN tunable parameters
        :param f: Current Basinhopping error achieved
        :param accept: Determines whether the current error achieved is accepted (stopping iterations)
        '''
        global best_loss_values
        global loss_values
        global best_validation_loss
        global validation_loss
        print(f"Best basinhopping iteration error so far: {best_loss_values[-1]}")
        print(f"Current basinhopping iteration error: {f}\n==========\n")
        if best_validation_loss[-1] > validation_loss[-1]:# and best_loss_values[-1] > loss_values[-1]:
            best_loss_values = loss_values.copy()
            best_validation_loss = validation_loss.copy()
        loss_values = [9999]
        validation_loss = [9999]
    
    qnn = QNN("model_N" + str(N) + "_L" + str(layers) + "_" + name, N, layers, n_inputs, n_outputs,
              photon_additions, observable, is_input_reupload, in_preprocs, out_prepocs, postprocs)
    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])
    
    valid_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, valid_set[0])
    valid_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, valid_set[1])
    
    global best_loss_values
    best_loss_values = [9999]
    global loss_values
    loss_values = [9999]
    global best_validation_loss
    best_validation_loss = [9999]
    global validation_loss
    validation_loss = [9999]
    
    training_QNN = partial(qnn.train_QNN, inputs_dataset=train_inputs, outputs_dataset=train_outputs, loss_function=loss_function)
    validate_QNN = partial(qnn.train_QNN, inputs_dataset=valid_inputs, outputs_dataset=valid_outputs, loss_function=loss_function)
    
    training_start = time.time()
    minimizer_kwargs = {"method": "L-BFGS-B", "callback": callback}
    result = opt.basinhopping(training_QNN, init_pars, niter=hopping_iters, minimizer_kwargs=minimizer_kwargs, callback=callback_hopping)
    print(f'Total training time: {time.time() - training_start} seconds')
    
    print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}')
    print(result.fun)
    
    qnn.build_QNN(result.x)

    qnn.print_qnn()
    print(qnn.qnn_profiling.avg_benchmark())
    
    if save:
        qnn.qnn_profiling.clear_times()
        qnn.save_model(qnn.model_name + ".txt")
        
    if best_loss_values[0] == 9999:
        best_loss_values = best_loss_values[1:]
        best_validation_loss = best_validation_loss[1:]
    
    return qnn, best_loss_values, best_validation_loss
