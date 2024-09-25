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
    def __init__(self, N, layers):#, num_observables):
        self.N = N
        self.layers = layers
        #self.num_observables = num_observables
        self.build_qnn_times = []
        self.input_prep_times = []
        self.gauss_times = []
        self.K_exp_vals_times = []
        #self.ladder_superpos_times = []
        self.nongauss_times = []

    def avg_benchmark(self):
        self.avg_times = {}
        self.avg_times["Build QNN"] = sum(self.build_qnn_times)/len(self.build_qnn_times)
        self.avg_times["Input prep"] = sum(self.input_prep_times)/len(self.input_prep_times)
        self.avg_times["Gaussian op"] = sum(self.gauss_times)/len(self.gauss_times)
        self.avg_times["Pairs exp-vals"] = sum(self.K_exp_vals_times)/len(self.K_exp_vals_times)
        #self.avg_times["Non-gauss coefs"] = sum(self.ladder_superpos_times)/len(self.ladder_superpos_times)
        self.avg_times["Non-gaussianity"] = sum(self.nongauss_times)/len(self.nongauss_times)
        return self.avg_times
    
    def clear_times(self):
        self.build_qnn_times = []
        self.input_prep_times = []
        self.gauss_times = []
        self.K_exp_vals_times = []
        #self.ladder_superpos_times = []
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
        
        self.model_name = model_name
        self.N = N
        self.layers = layers
        self.photon_add = photon_add
        self.n_in = n_in
        self.n_out = n_out
        self.in_preprocessors = in_preprocessors
        self.out_preprocessors = out_preprocessors
        self.postprocessors = postprocessors
        self.is_input_reupload = is_input_reupload
        
        self.Q1_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Q2_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Z_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.S_l = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.D_l = np.zeros((self.layers, 2*self.N))
        self.S_concat = np.zeros((2*self.N, 2*self.layers*self.N))
        self.D_concat = np.zeros((2*self.layers*self.N))
        self.G = np.eye(2*self.N)
        
        # Normalization expression related to a single photon addition on first mode for each QNN layer
        self.norm_trace_expr = complete_trace_expression(self.N, layers, photon_add, self.n_out, include_obs=False)

        # Full trace expression including the photon additions and the observable to be measured
        self.trace_expr = complete_trace_expression(self.N, layers, photon_add, self.n_out, include_obs=True, obs=observable)
        
        self.modes, self.types = [], []
        for outs in range(self.n_out):
            modes, types = extract_ladder_expressions(self.trace_expr[outs])
            self.modes.append(modes)
            self.types.append(types)
        self.np_modes, self.lens_modes = to_np_array(self.modes)
        self.np_types, self.lens_types = to_np_array(self.types)
                
        modes_norm, types_norm = extract_ladder_expressions(self.norm_trace_expr)
        self.modes_norm, self.types_norm = [modes_norm], [types_norm]
        self.np_modes_norm, self.lens_modes_norm = to_np_array(self.modes_norm)
        self.np_types_norm, self.lens_types_norm = to_np_array(self.types_norm)
        
        max_lpms = np.max(self.lens_modes)
        self.lpms = [to_np_array([loop_perfect_matchings(lens)]) for lens in range(3, max_lpms+1)]
        if len(self.lpms) == 0:
            self.lpms = [to_np_array([loop_perfect_matchings(3)])]
        self.np_lpms = nb.typed.List([lpms for (lpms, _) in self.lpms])
        self.lens_lpms = nb.typed.List([lens for (_, lens) in self.lpms])
        
        a = symbols(f'a0:{N}', commutative=False)
        c = symbols(f'c0:{N}', commutative=False)
        ladder_subs = {c[i]: 1 for i in range(self.N)}
        ladder_subs.update({a[i]: 1 for i in range(self.N)})
        
        self.unnorm_expr_terms_out = []
        for outs in range(self.n_out):
            unnorm_expr_terms = list(self.trace_expr[outs].args) if len(photon_add) > 0 or observable!='number' else list(self.trace_expr[outs].args[1:])
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
        
        d_r = symbols(f'r0:{layers*N}', commutative=True)
        d_i = symbols(f'i0:{layers*N}', commutative=True)
        dim = 2*N
        S = MatrixSymbol('S', dim, layers*dim)
        
        self.num_unnorm = []
        for outs in range(self.n_out):
            self.num_unnorm.append([lambdify((S, d_r, d_i), unnorm_trm, modules='numpy') for unnorm_trm in self.unnorm_expr_terms_out[outs]])
        self.num_norm = [lambdify((S, d_r, d_i), norm_trm, modules='numpy') for norm_trm in self.norm_subs_expr_terms]
        """ self.nb_unnorm = nb.typed.List.empty_list(nb.types.ListType(nb.types.complex64(nb.types.Array(nb.types.float64, 2, 'C'), nb.types.Array(nb.types.complex64, 1, 'C'), nb.types.Array(nb.types.complex64, 1, 'C')).as_type()))
        self.nb_norm = nb.typed.List.empty_list(nb.types.complex64(nb.types.Array(nb.types.float64, 2, 'C'), nb.types.Array(nb.types.complex64, 1, 'C'), nb.types.Array(nb.types.complex64, 1, 'C')).as_type())
        self.nb_num_unnorm = []
        for outs in range(self.n_out):
            self.nb_num_unnorm.append([nb.njit(f) for f in self.num_unnorm[outs]])
        self.nb_num_norm = [nb.njit(f) for f in self.num_norm]
        for outs in range(self.n_out):
            nb_unnorm_out = nb.typed.List.empty_list(nb.types.complex64(nb.types.Array(nb.types.float64, 2, 'C'), nb.types.Array(nb.types.complex64, 1, 'C'), nb.types.Array(nb.types.complex64, 1, 'C')).as_type())
            for f in self.nb_num_unnorm[outs]:
                nb_unnorm_out.append(f)
            self.nb_unnorm.append(nb_unnorm_out)
        for f in self.nb_num_norm:
            self.nb_norm.append(f) """
            
        self.u_bar = CanonicalLadderTransformations(N)
        self.qnn_profiling = ProfilingQNN(N, layers)

    def build_QNN(self, parameters):
        self.tunable_parameters = np.copy(parameters)
        self.set_parameters(parameters)

    def set_parameters(self, parameters):
        S_dim = 2*self.N
        self.G = np.eye(S_dim)
        current_par_idx = 0
        for l in range(self.layers-1, -1, -1):
            # Build passive-optics Q1 and Q2 for the Gaussian transformation
            H = hermitian_matrix(parameters[current_par_idx : current_par_idx + ((self.N**2 + self.N) // 2)], self.N)
            #H = general_hermitian_matrix(parameters[current_par_idx : current_par_idx + self.N**2], self.N)
            U = unitary_from_hermitian(H)
            self.Q1_gauss[l] = np.real_if_close(self.u_bar.to_canonical_op(U))
            current_par_idx += ((self.N**2 + self.N) // 2)
            #current_par_idx += self.N**2
            
            H = hermitian_matrix(parameters[current_par_idx : current_par_idx + ((self.N**2 + self.N) // 2)], self.N)
            #H = general_hermitian_matrix(parameters[current_par_idx : current_par_idx + self.N**2], self.N)
            U = unitary_from_hermitian(H)
            self.Q2_gauss[l] = np.real_if_close(self.u_bar.to_canonical_op(U))
            current_par_idx += ((self.N**2 + self.N) // 2)
            #current_par_idx += self.N**2
            
            # Build squeezing diagonal matrix Z
            sqz_parameters = np.abs(parameters[current_par_idx : current_par_idx + self.N])
            sqz_inv = 1.0/sqz_parameters
            self.Z_gauss[l] = np.diag(np.concatenate((sqz_parameters, sqz_inv)))
            current_par_idx += self.N

            # Build final Gaussian transformation
            self.S_l[l] = self.Q2_gauss[l] @ self.Z_gauss[l] @ self.Q1_gauss[l]
            self.G = self.G @ self.S_l[l]
            self.S_concat[:, l*S_dim:(l+1)*S_dim] = self.G.copy()
            
            # Build displacements
            if not self.is_input_reupload:
                self.D_l[l] = parameters[current_par_idx : current_par_idx + 2*self.N]
                current_par_idx += 2*self.N
                self.D_concat[l*S_dim:(l+1)*S_dim] = self.D_l[l]
        
    def displacement_operator(self, factors):
        self.mean_vector[0:len(factors)] += factors
        
    def gaussian_transformation(self):
        for l in range(self.layers):
            self.V = self.S_l[l] @ self.V @ self.S_l[l].T
            self.mean_vector = self.S_l[l] @ self.mean_vector
            self.displacement_operator(self.D_l[l])

    def eval_QNN(self, input):
        # 1. Prepare initial state: initial vacuum state displaced according to the inputs
        input_prep_start = time.time()
        self.V = 0.5*np.eye(2*self.N)
        self.mean_vector = np.zeros(2*self.N)
        self.displacement_operator(input)
        self.qnn_profiling.input_prep_times.append(time.time() - input_prep_start)
        
        # 1.1. When using input reuploading: build the symplectic coefficients for ladder operators' superposition
        # TODO: Modify the profiling name
        #ladder_superpos_start = time.time()
        if self.is_input_reupload:
            for l in range(self.layers):
                self.D_l[l][0:len(input)] = input
                self.D_concat[l*2*self.N:l*2*self.N + len(input)] = input
        #self.qnn_profiling.ladder_superpos_times.append(time.time() - ladder_superpos_start)

        # 2. Apply the Gaussian transformation acting as weights matrix
        gauss_start = time.time()
        self.gaussian_transformation()
        self.qnn_profiling.gauss_times.append(time.time() - gauss_start)
        
        # 3. Compute the expectation values of all possible ladder operators pairs over the final Gaussian state
        K_exp_vals_start = time.time()
        K_exp_vals = compute_K_exp_vals(self.V, self.mean_vector)
        self.qnn_profiling.K_exp_vals_times.append(time.time() - K_exp_vals_start)
        
        # 5. Compute the observables' normalized expectation value of the non-Gaussianity applied to the final Gaussian state
        # TODO: Generalize for multilayer
        nongauss_start = time.time()
        d_r, d_i = create_displacements(self.D_concat, self.N, self.layers)
        norm_exp_val = compute_exp_val_loop(self.num_unnorm, self.num_norm,
                                            self.np_modes, self.np_types, self.lens_modes,
                                            self.np_modes_norm, self.np_types_norm, self.lens_modes_norm, 
                                            self.np_lpms, self.S_concat, d_r, d_i, K_exp_vals, self.mean_vector)
        self.qnn_profiling.nongauss_times.append(time.time() - nongauss_start)

        #compute_times(self)
        #print(f"OUTCOME: {np.real_if_close(norm_exp_val, tol=1e6)}")
        return np.real_if_close(norm_exp_val, tol=1e6)

    def train_QNN(self, parameters, inputs_dataset, outputs_dataset, loss_function):
        build_start = time.time()
        self.build_QNN(parameters)
        self.qnn_profiling.build_qnn_times.append(time.time() - build_start)
        
        shuffle_indices = np.random.permutation(len(inputs_dataset))
        qnn_outputs = np.full_like(outputs_dataset, 0)
        for dataset_idx in shuffle_indices:
            qnn_outputs[dataset_idx] = self.eval_QNN(inputs_dataset[dataset_idx])
        return loss_function(outputs_dataset, qnn_outputs)
    
    def print_qnn(self):
        for layer in range(self.layers):
            print(f"Layer {layer+1}:\nQ1 = {self.Q1_gauss[layer]}\nZ = {self.Z_gauss[layer]}\nQ2 = {self.Q2_gauss[layer]}")
            if not self.is_input_reupload:
                print(f"D = {self.D_l[layer]}")
        
    def save_model(self, filename):
        f = open("models/"+filename, 'w')
        f.write(jsonpickle.encode(self))
        f.close()
    
def load_model(filename):
    with open(filename, 'r') as f:
        qnn_str = jsonpickle.decode(f.read())
    return qnn_str
    
def test_model(qnn, testing_dataset, loss_function):
    '''
    Makes predictions of the given QNN using the input testing dataset.
    
    :param qnn: QNN to be tested
    :param testing_dataset: List of inputs and outputs to be tested
    :return: QNN predictions of the testing set
    '''
    test_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, testing_dataset[0])
    test_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, testing_dataset[1])
    
    #error = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
    #qnn_outputs = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
    qnn_outputs = np.full_like(test_outputs, 0)
    
    # Evaluate all testing set
    for k in range(len(test_inputs)):
        qnn_outputs[k] = qnn.eval_QNN(test_inputs[k])
    mean_error = loss_function(test_outputs, qnn_outputs)
    print(f"LOSS VALUE FOR TESTING SET: {mean_error}")
    
    return reduce(lambda x, func: func(x), qnn.postprocessors, qnn_outputs)
    
def build_and_train_model(name, N, layers, n_inputs, n_outputs, photon_additions, observable, is_input_reupload, 
                          train_set, valid_set, loss_function=mse, in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    '''
    Creates and trains a QNN model with the given hyperparameters and dataset by optimizing the 
    tunable parameters of the QNN.
    
    :param name: Name of the model
    :param N: Number of neurons per layer (modes of the quantum system)
    :param layers: Number of layers
    :param observable_modes: Observable operator modes
    :param observable_types: Observable operator ladder types
    :param is_input_reupload: Boolean determining whether the model has input reuploading
    :param train_set: List of inputs and outputs to be learned
    :param init_pars: Initialization parameters for the QNN
    :param save: Boolean determining whether to save the model (default=True)
    :return: Trained QNN model
    '''
    if type(init_pars) == type(None):
        init_pars = np.random.rand(layers*(2*((N**2 + N) // 2) + N)) if is_input_reupload else np.random.rand(layers*(2*((N**2 + N) // 2) + N + 2*N))
        #init_pars = np.random.rand(layers*(2*N**2 + N)) if is_input_reupload else np.random.rand(layers*(2*N**2 + 3*N))
    else:
        if is_input_reupload:
            assert len(init_pars) == layers*(2*((N**2 + N) // 2) + N)
            #assert len(init_pars) == layers*(2*N**2 + N)
        else:
            assert len(init_pars) == layers*(2*((N**2 + N) // 2) + N + 2*N)
            #assert len(init_pars) == layers*(2*N**2 + 3*N)
    
    def callback(xk):
        '''
        Callback function that prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QNN tunable parameters
        '''
        e = training_QNN(xk)
        print(f'Training loss: {e}')
        val_e = validate_QNN(xk)
        print(f'Validation loss: {val_e}')
        loss_values.append(e)
        validation_loss.append(val_e)
        
    def callback_hopping(x,f,accept):
        global best_loss_values
        global loss_values
        global best_validation_loss
        global validation_loss
        print(f"Best basinhopping iteration error so far: {best_loss_values[-1]}\n")
        print(f"Current basinhopping iteration error: {f}")
        if best_validation_loss[-1] > validation_loss[-1]:# and best_loss_values[-1] > loss_values[-1]:
            best_loss_values = loss_values.copy()
            best_validation_loss = validation_loss.copy()
        loss_values = [9999]
        validation_loss = [9999]
    
    qnn = QNN("model_N" + str(N) + "_L" + str(layers) + "_" + name, N, layers, n_inputs, n_outputs,
              photon_additions, observable, is_input_reupload, in_preprocs, out_prepocs, postprocs)
    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])
    """ print("TRAIN DATASET:")
    for (inp, inp_prep, outp, outp_prep) in zip(train_set[0], train_inputs, train_set[1], train_outputs):
        print(inp, inp_prep, outp, outp_prep)
     """
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
    #result = opt.minimize(training_QNN, init_pars, method='L-BFGS-B', callback=callback)
    minimizer_kwargs = {"method": "L-BFGS-B", "callback": callback}
    result = opt.basinhopping(training_QNN, init_pars, niter=2, minimizer_kwargs=minimizer_kwargs, callback=callback_hopping)
    print(f'Total training time: {time.time() - training_start} seconds')
    
    print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}')
    print(result.fun)
    
    qnn.build_QNN(result.x)
    
    """ if n_inputs == 1:
        qnn_outputs = test_model(qnn, train_set, loss_function)
        plot_qnn_train_results(qnn, train_set[0], train_set[1], qnn_outputs, best_loss_values)
    """
    #show_times(qnn)
    qnn.print_qnn()
    
    if save:
        qnn.qnn_profiling.clear_times()
        qnn.save_model(qnn.model_name + ".txt")
        
    if best_loss_values[0] == 9999:
        best_loss_values = best_loss_values[1:]
        best_validation_loss = best_validation_loss[1:]
    
    return qnn, best_loss_values, best_validation_loss

