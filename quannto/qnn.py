import numpy as np
import time
import jsonpickle
from numba import njit, prange
import time
import scipy.optimize as opt
from functools import partial

from utils import *
from expectation_value import *
from results_utils import *

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
        self.ladder_superpos_times = []
        self.nongauss_times = []

    def avg_times(self):
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
    def __init__(self, model_name, N, layers, n_in, n_out, observable_modes, observable_types, is_input_reupload):
        self.model_name = model_name
        self.N = N
        self.layers = layers
        self.n_in = n_in
        self.n_out = n_out
        self.is_input_reupload = is_input_reupload
        
        # Normalization expression related to a single photon addition on first mode for each QNN layer
        self.ladder_modes_norm, self.ladder_types_norm = multilayer_ladder_trace_expression(N, layers)
        self.perf_matchings_norm = np.array(perfect_matchings(len(self.ladder_modes_norm[0][0])), dtype='int')
        
        # Full trace expression including the photon additions and the observable to be measured
        self.ladder_modes, self.ladder_types = include_observable(self.ladder_modes_norm[0], self.ladder_types_norm[0], observable_modes, observable_types)
        self.perf_matchings = np.array(perfect_matchings(len(self.ladder_modes[0][0])), dtype='int')
        
        self.ladder_modes_norm, self.ladder_types_norm = np.array(self.ladder_modes_norm, dtype='int'), np.array(self.ladder_types_norm, dtype='int')
        self.ladder_modes, self.ladder_types = np.array(self.ladder_modes, dtype='int'), np.array(self.ladder_types, dtype='int')
        
        self.u_bar = CanonicalLadderTransformations(N)
        self.qnn_profiling = ProfilingQNN(N, layers)

    def build_QNN(self, parameters):
        self.tunable_parameters = np.copy(parameters)
        self.set_parameters(parameters)
        if not(self.is_input_reupload):
            self.trace_coefs = self.non_gauss_symplectic_coefs()

    def set_parameters(self, parameters):
        self.Q1_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Q2_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.Z_gauss = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.G_l = np.zeros((self.layers, 2*self.N, 2*self.N))
        self.G = np.eye(2*self.N)
        
        current_par_idx = 0
        for l in range(self.layers):
            # Build passive-optics Q1 and Q2 for the Gaussian transformation
            H = hermitian_matrix(parameters[current_par_idx : current_par_idx + self.N**2].reshape((self.N, self.N)))
            U = unitary_from_hermitian(H)
            self.Q1_gauss[l] = np.real_if_close(self.u_bar.to_canonical_op(U))
            current_par_idx += self.N**2
            
            H = hermitian_matrix(parameters[current_par_idx : current_par_idx + self.N**2].reshape((self.N, self.N)))
            U = unitary_from_hermitian(H)
            self.Q2_gauss[l] = np.real_if_close(self.u_bar.to_canonical_op(U))
            current_par_idx += self.N**2
            
            if not(self.is_input_reupload):
                # Build squeezing diagonal matrix Z
                sqz_parameters = parameters[current_par_idx : current_par_idx + self.N]
                sqz_inv = 1.0/sqz_parameters
                self.Z_gauss[l] = np.diag(np.concatenate((sqz_parameters, sqz_inv)))
                current_par_idx += self.N

                # Build final Gaussian transformation
                self.G_l[l] = self.Q2_gauss[l] @ self.Z_gauss[l] @ self.Q1_gauss[l]
                self.G = self.G_l[l] @ self.G
        
    def squeezing_operator(self, input):
        r = np.ones(self.N)
        r[0:len(input)] = input
        r_inv = 1.0/r
        return np.diag(np.concatenate((r, r_inv)))

    def gaussian_transformation(self):
        self.final_gauss_state = self.G @ self.initial_gauss_state @ self.G.T
        
    def gaussian_transf_is_input_reupload(self, Z_input):
        self.final_gauss_state = self.initial_gauss_state
        for l in range(self.layers):
            self.G_l[l] = self.Q2_gauss[l] @ Z_input @ self.Q1_gauss[l]
            self.final_gauss_state = self.G_l[l] @ self.final_gauss_state @ self.G_l[l].T
        
    def non_gauss_symplectic_coefs(self):
        self.S_commutator = np.zeros((self.layers, 2*self.N, 2*self.N))
        
        self.S_commutator[self.layers - 1] = np.copy(self.G_l[self.layers - 1])
        for l in range(self.layers-2, -1, -1):
            self.S_commutator[l] = self.S_commutator[l + 1] @ self.G_l[l]
            
        return get_symplectic_coefs(self.N, self.S_commutator, self.ladder_modes_norm, self.ladder_types_norm)

    @njit
    def exp_val_non_gaussianity(ladder_modes, ladder_types, perf_matchings,
                                ladder_modes_norm, ladder_types_norm, perf_matchings_norm, 
                                trace_coefs, K_exp_vals):
        # 1. Calculates the expectation value
        unnorm_exp_val = np.zeros((len(ladder_modes)), dtype='complex')
        for i in prange(len(ladder_modes)):
            for j in prange(len(ladder_modes[i])):
                # Always row 0 because the photon addition is on the first mode
                unnorm_exp_val[i] += trace_coefs[0,j] * ladder_exp_val(
                    perf_matchings, ladder_modes[i][j], ladder_types[i][j], K_exp_vals
                )
                
        # 2. Calculates the normalization factor of the expectation value
        if ladder_modes_norm.size == 0:
            exp_val_norm = np.ones((len(ladder_modes)), dtype='complex')
        else:
            exp_val_norm = np.zeros((len(ladder_modes)), dtype='complex')
            for i in prange(len(ladder_modes_norm)):
                for j in prange(len(ladder_modes_norm[i])):
                    exp_val_norm[i] += trace_coefs[0,j] * ladder_exp_val(
                        perf_matchings_norm, ladder_modes_norm[i][j], ladder_types_norm[i][j], K_exp_vals
                    )
            for i in prange(len(ladder_modes)):
                exp_val_norm[i] = exp_val_norm[0]
                
        return unnorm_exp_val/exp_val_norm

    def eval_QNN(self, input):
        # 1. Prepare initial state: build the squeezing operator used for input encoding
        input_prep_start = time.time()
        Z_input = self.squeezing_operator(input)
        self.initial_gauss_state = Z_input
        self.qnn_profiling.input_prep_times.append(time.time() - input_prep_start)

        # 2. Apply the Gaussian transformation -> Weight matrix in ANN
        gauss_start = time.time()
        self.gaussian_transf_is_input_reupload(Z_input) if self.is_input_reupload else self.gaussian_transformation()
        self.qnn_profiling.gauss_times.append(time.time() - gauss_start)
        
        # 3. Compute the expectation values of all possible ladder operators pairs over the final Gaussian state
        K_exp_vals_start = time.time()
        K_exp_vals = compute_K_exp_vals(self.final_gauss_state)
        self.qnn_profiling.K_exp_vals_times.append(time.time() - K_exp_vals_start)

        # 4. When using input reuploading: build the symplectic coefficients for ladder operators' superposition
        ladder_superpos_start = time.time()
        if self.is_input_reupload:
            self.trace_coefs = self.non_gauss_symplectic_coefs()
        self.qnn_profiling.ladder_superpos_times.append(time.time() - ladder_superpos_start)

        # 5. Compute the observables' normalized expectation value of the non-Gaussianity applied to the final Gaussian state
        nongauss_start = time.time()
        norm_exp_val = QNN.exp_val_non_gaussianity(self.ladder_modes, self.ladder_types, self.perf_matchings, 
                                                   self.ladder_modes_norm, self.ladder_types_norm, self.perf_matchings_norm,
                                                   self.trace_coefs, K_exp_vals)
        self.qnn_profiling.nongauss_times.append(time.time() - nongauss_start)
        
        return np.real_if_close(norm_exp_val, tol=1e6)

    def train_QNN(self, parameters, inputs_dataset, outputs_dataset):
        build_start = time.time()
        self.build_QNN(parameters)
        self.qnn_profiling.build_qnn_times.append(time.time() - build_start)
        
        mse = 0
        for inputs, outputs in zip(inputs_dataset, outputs_dataset):
            qnn_outputs = self.eval_QNN(inputs)
            mse += ((outputs - qnn_outputs)**2).sum()
        return mse/len(inputs_dataset)
    
    def print_qnn(self):
        for layer in range(self.layers):
            print(f"Layer {layer}:\nQ1 = {self.Q1_gauss[layer]}\nZ = {self.Z_gauss[layer] if not(self.is_input_reupload) else None}\nQ2 = {self.Q2_gauss[layer]}")
            print(f"\nGaussian operator:\nQ2={self.Q2_gauss[layer]}\nZg={self.Z_gauss if not(self.is_input_reupload) else None}\nQ1={self.Q1_gauss}\n")
        
    def save_model(self, filename):
        f = open(filename, 'w')
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
    error = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
    qnn_outputs = np.zeros((len(testing_dataset[1]), len(testing_dataset[1][0])))
    
    test_inputs, test_outputs = testing_dataset[0], testing_dataset[1]
    
    # Evaluate all testing set
    for k in range(len(test_inputs)):
        qnn_outputs[k] = np.real_if_close(qnn.eval_QNN(test_inputs[k]))
        error[k] = ((test_outputs[k] - qnn_outputs[k])**2).sum()
    mean_error = error.sum()/len(error)
    print(f"MSE: {mean_error}")
    
    return qnn_outputs
    
def build_and_train_model(name, N, layers, n_inputs, n_outputs, observable_modes, observable_types, is_input_reupload, dataset, init_pars=None, save=True):
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
    if init_pars == None:
        init_pars = np.random.rand(layers*(2*N**2)) if is_input_reupload else np.random.rand(layers*(2*N**2 + N))
    else:
        if is_input_reupload:
            assert len(init_pars) == layers*(2*N**2)
        else:
            assert len(init_pars) == layers*(2*N**2 + N)
    
    def callback(xk):
        '''
        Callback function that prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QNN tunable parameters
        '''
        e = training_QNN(xk)
        print(e)
        loss_values.append(e)
    
    qnn = QNN("model_N" + str(N) + "_L" + str(layers) + "_" + name, N, layers, 
              n_inputs, n_outputs, observable_modes, observable_types, is_input_reupload)
    
    train_inputs, train_outputs = dataset[0], dataset[1]
    
    loss_values = []
    training_QNN = partial(qnn.train_QNN, inputs_dataset=train_inputs, outputs_dataset=train_outputs)
    training_start = time.time()
    result = opt.minimize(training_QNN, init_pars, method='L-BFGS-B', callback=callback)
    print(f'Total training time: {time.time() - training_start} seconds')
    
    print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}')
    print(result.fun)
    
    qnn.build_QNN(result.x)
    
    qnn_outputs = test_model(qnn, dataset)
    plot_qnn_train_results(qnn, dataset[1], qnn_outputs, loss_values)
    show_times(qnn)
    qnn.print_qnn()
    
    if save:
        qnn.qnn_profiling.clear_times()
        qnn.save_model(qnn.model_name + ".txt")
    
    return qnn

