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
        self.n_in = n_in
        self.n_out = n_out
        self.in_preprocessors = in_preprocessors
        self.out_preprocessors = out_preprocessors
        self.postprocessors = postprocessors
        self.is_input_reupload = is_input_reupload
        
        # Normalization expression related to a single photon addition on first mode for each QNN layer
        self.norm_trace_expr = complete_trace_expression(self.N, photon_add, self.n_out, include_obs=False)

        # Full trace expression including the photon additions and the observable to be measured
        # TODO: Generalize for any number of outputs
        self.trace_expr = complete_trace_expression(self.N, photon_add, self.n_out, include_obs=True, obs=observable)
        print("EXPECTATION VALUE EXPRESSION:")
        print(self.trace_expr)
        
        print("NORM EXPRESSION:")
        print(self.norm_trace_expr)
        
        # TODO: Generalize for any number of perfect matchings needed for the exp val expression
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
            unnorm_expr_terms = list(self.trace_expr[outs].args)
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
            
        d_r = symbols(f'r0:{N}', commutative=True)
        d_i = symbols(f'i0:{N}', commutative=True)
        dim = 2*N
        S = MatrixSymbol('S', dim, dim)
        
        num_unnorm = []
        for outs in range(self.n_out):
            num_unnorm.append([lambdify((S, d_r, d_i), unnorm_trm, modules='numpy') for unnorm_trm in self.unnorm_expr_terms_out[outs]])
        num_norm = [lambdify((S, d_r, d_i), norm_trm, modules='numpy') for norm_trm in self.norm_subs_expr_terms]
        self.nb_unnorm = nb.typed.List.empty_list(nb.types.ListType(nb.types.float64(nb.types.Array(nb.types.float64, 2, 'C'), nb.types.Array(nb.types.complex64, 1, 'C'), nb.types.Array(nb.types.complex64, 1, 'C')).as_type()))
        self.nb_norm = nb.typed.List.empty_list(nb.types.float64(nb.types.Array(nb.types.float64, 2, 'C'), nb.types.Array(nb.types.complex64, 1, 'C'), nb.types.Array(nb.types.complex64, 1, 'C')).as_type())
        nb_num_unnorm = []
        for outs in range(self.n_out):
            nb_num_unnorm.append([nb.njit(f) for f in num_unnorm[outs]])
        nb_num_norm = [nb.njit(f) for f in num_norm]
        for outs in range(self.n_out):
            nb_unnorm_out = nb.typed.List.empty_list(nb.types.float64(nb.types.Array(nb.types.float64, 2, 'C'), nb.types.Array(nb.types.complex64, 1, 'C'), nb.types.Array(nb.types.complex64, 1, 'C')).as_type())
            for f in nb_num_unnorm[outs]:
                nb_unnorm_out.append(f)
            self.nb_unnorm.append(nb_unnorm_out)
        for f in nb_num_norm:
            self.nb_norm.append(f)
            
        self.u_bar = CanonicalLadderTransformations(N)
        self.qnn_profiling = ProfilingQNN(N, layers)

    def build_QNN(self, parameters):
        self.tunable_parameters = np.copy(parameters)
        self.set_parameters(parameters)

    def set_parameters(self, parameters):
        self.Q1_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Q2_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Z_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.G_l = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.D_l = np.zeros((self.layers, 2*self.N))
        self.G = np.eye(2*self.N)
        
        current_par_idx = 0
        for l in range(self.layers):
            # Build passive-optics Q1 and Q2 for the Gaussian transformation
            #H = hermitian_matrix(parameters[current_par_idx : current_par_idx + self.N**2].reshape((self.N, self.N)))
            H = hermitian_matrix(parameters[current_par_idx : current_par_idx + ((self.N**2 + self.N) // 2)], self.N)
            U = unitary_from_hermitian(H)
            self.Q1_gauss[l] = np.real_if_close(self.u_bar.to_canonical_op(U))
            #current_par_idx += self.N**2
            current_par_idx += ((self.N**2 + self.N) // 2)
            
            #H = hermitian_matrix(parameters[current_par_idx : current_par_idx + self.N**2].reshape((self.N, self.N)))
            H = hermitian_matrix(parameters[current_par_idx : current_par_idx + ((self.N**2 + self.N) // 2)], self.N)
            U = unitary_from_hermitian(H)
            self.Q2_gauss[l] = np.real_if_close(self.u_bar.to_canonical_op(U))
            #current_par_idx += self.N**2
            current_par_idx += ((self.N**2 + self.N) // 2)
            
            # Build squeezing diagonal matrix Z
            sqz_parameters = np.abs(parameters[current_par_idx : current_par_idx + self.N])
            sqz_inv = 1.0/sqz_parameters
            self.Z_gauss[l] = np.diag(np.concatenate((sqz_parameters, sqz_inv)))
            current_par_idx += self.N

            # Build final Gaussian transformation
            self.G_l[l] = self.Q2_gauss[l] @ self.Z_gauss[l] @ self.Q1_gauss[l]
            #self.G = self.G_l[l] @ self.G
            
            # Build displacements
            if not self.is_input_reupload:
                self.D_l[l] = parameters[current_par_idx : current_par_idx + 2*self.N]
                current_par_idx += 2*self.N
    
    def displacement_operator(self, factors):
        self.mean_vector[0:len(factors)] += factors

    def gaussian_transformation(self):
        for l in range(self.layers):
            self.V = self.G_l[l] @ self.V @ self.G_l[l].T
            self.mean_vector = self.G_l[l] @ self.mean_vector
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
        nongauss_start = time.time()
        norm_exp_val = compute_exp_val_loop(self.nb_unnorm, self.nb_norm,
                                            self.np_modes, self.np_types, self.lens_modes,
                                            self.np_modes_norm, self.np_types_norm, self.lens_modes_norm, 
                                            self.np_lpms, self.D_l[0], self.G_l[0], K_exp_vals, self.mean_vector)
        self.qnn_profiling.nongauss_times.append(time.time() - nongauss_start)
        
        #compute_times(self)
        #print(f"OUTCOME: {np.real_if_close(norm_exp_val, tol=1e6)}")
        return np.real_if_close(norm_exp_val, tol=1e6)

    def train_QNN(self, parameters, inputs_dataset, outputs_dataset):
        build_start = time.time()
        self.build_QNN(parameters)
        self.qnn_profiling.build_qnn_times.append(time.time() - build_start)
        
        shuffle_indices = np.random.permutation(len(inputs_dataset))
        mse = 0
        for dataset_idx in shuffle_indices:
            qnn_outputs = self.eval_QNN(inputs_dataset[dataset_idx])
            #print(f"Desired: {outputs_dataset[dataset_idx]} Obtained: {qnn_outputs}")
            #qnn_outputs = reduce(lambda x, func: func(x), self.postprocessors, qnn_outputs)
            mse += ((outputs_dataset[dataset_idx] - qnn_outputs)**2).sum()
        """ for inputs, outputs in zip(inputs_dataset, outputs_dataset):
            qnn_outputs = self.eval_QNN(inputs)
            mse += ((outputs - qnn_outputs)**2).sum() """
        return mse/len(inputs_dataset)
    
    def print_qnn(self):
        for layer in range(self.layers):
            print(f"Layer {layer+1}:\nQ1 = {self.Q1_gauss[layer]}\nZ = {self.Z_gauss[layer]}\nQ2 = {self.Q2_gauss[layer]}")
        
    def save_model(self, filename):
        f = open("models/"+filename, 'w')
        f.write(jsonpickle.encode(self))
        f.close()
    
def load_model(filename):
    with open(filename, 'r') as f:
        qnn_str = jsonpickle.decode(f.read())
    return qnn_str
    
def test_model(qnn, testing_dataset):
    '''
    Makes predictions of the given QNN using the input testing dataset.
    
    :param qnn: QNN to be tested
    :param testing_dataset: List of inputs and outputs to be tested
    :return: QNN predictions of the testing set
    '''
    test_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, testing_dataset[0])
    test_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, testing_dataset[1])
    
    error = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
    qnn_outputs = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
    
    # Evaluate all testing set
    for k in range(len(test_inputs)):
        qnn_outputs[k] = np.real_if_close(qnn.eval_QNN(test_inputs[k]))
        error[k] = ((test_outputs[k] - qnn_outputs[k])**2).sum()
    mean_error = error.sum()/(len(error)*len(test_outputs[0]))
    print(f"MSE: {mean_error}")
    """ for (i,j) in zip(test_outputs, qnn_outputs):
        print(f"Expected: {i} Obtained: {j}") """
    
    return reduce(lambda x, func: func(x), qnn.postprocessors, qnn_outputs)
    
def build_and_train_model(name, N, layers, n_inputs, n_outputs, photon_additions, observable, is_input_reupload, 
                          dataset, in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    '''
    Creates and trains a QNN model with the given hyperparameters and dataset by optimizing the 
    tunable parameters of the QNN.
    
    :param name: Name of the model
    :param N: Number of neurons per layer (modes of the quantum system)
    :param layers: Number of layers
    :param observable_modes: Observable operator modes
    :param observable_types: Observable operator ladder types
    :param is_input_reupload: Boolean determining whether the model has input reuploading
    :param dataset: List of inputs and outputs to be learned
    :param init_pars: Initialization parameters for the QNN
    :param save: Boolean determining whether to save the model (default=True)
    :return: Trained QNN model
    '''
    if type(init_pars) == type(None):
        #init_pars = np.random.rand(layers*(2*N**2)) if is_input_reupload else np.random.rand(layers*(2*N**2 + N))
        init_pars = np.random.rand(layers*(2*((N**2 + N) // 2) + N)) if is_input_reupload else np.random.rand(layers*(2*((N**2 + N) // 2) + N + 2*N))
    else:
        if is_input_reupload:
            #assert len(init_pars) == layers*(2*N**2)
            assert len(init_pars) == layers*(2*((N**2 + N) // 2) + N)
        else:
            #assert len(init_pars) == layers*(2*N**2 + N)
            assert len(init_pars) == layers*(2*((N**2 + N) // 2) + N + 2*N)
    
    def callback(xk):
        '''
        Callback function that prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QNN tunable parameters
        '''
        e = training_QNN(xk)
        print(e)
        loss_values.append(e)
        
    def callback_hopping(x,f,accept):
        global best_loss_values
        global loss_values
        print(f"Error of basinhopping iteration: {f}")
        print(f"Previous iteration error: {best_loss_values[-1]}\n")
        if best_loss_values[-1] > f:
            best_loss_values = loss_values.copy()
        loss_values = []
    
    qnn = QNN("model_N" + str(N) + "_L" + str(layers) + "_" + name, N, layers, n_inputs, n_outputs,
              photon_additions, observable, is_input_reupload, in_preprocs, out_prepocs, postprocs)
    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, dataset[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, dataset[1])
    
    global best_loss_values
    best_loss_values = [10]
    global loss_values
    loss_values = []
    
    training_QNN = partial(qnn.train_QNN, inputs_dataset=train_inputs, outputs_dataset=train_outputs)
    training_start = time.time()
    result = opt.minimize(training_QNN, init_pars, method='L-BFGS-B', callback=callback)
    #minimizer_kwargs = {"method": "L-BFGS-B", "callback": callback}
    #result = opt.basinhopping(training_QNN, init_pars, niter=0, minimizer_kwargs=minimizer_kwargs, callback=callback_hopping)
    print(f'Total training time: {time.time() - training_start} seconds')
    
    print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}')
    print(result.fun)
    
    qnn.build_QNN(result.x)
    
    qnn_outputs = test_model(qnn, dataset)
    #plot_qnn_train_results(qnn, dataset[1], qnn_outputs, loss_values)
    #show_times(qnn)
    qnn.print_qnn()
    
    if save:
        qnn.qnn_profiling.clear_times()
        qnn.save_model(qnn.model_name + ".txt")
    
    return qnn, loss_values

