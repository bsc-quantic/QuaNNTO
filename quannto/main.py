import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from functools import partial

from qnn import QNN
from synth_datasets import f1_2var_generate_dataset, f1_2var_linear_dataset, bubblesort, print_dataset

def callback(xk):
    '''
    Callback function that prints and stores the MSE error value for each QNN training epoch.
    
    :param xk: QNN tunable parameters
    '''
    e = training_QNN(xk)
    print(e)
    total_error.append(e)
    

# === HYPERPARAMETERS DEFINITION ===
list_N = [2]
list_layers = [2]

cv_qnns = []
final_pars = []
dataset_inputs = []
dataset_outputs = []
list_loss = []
parameters = []
observable_modes = []
observable_types = []

dataset_size = 40
min_sample = 10
max_sample = 5000

for N in list_N:
    for layers in list_layers:
        parameters.append(np.random.rand((layers-1)*(2*N**2) + 2*(N**2) + N))
        print(f'\n\n===== PARAMETERS FOR N={N}, l={layers} =====')
        print(parameters)
        
        train_inputs, train_outputs = f1_2var_generate_dataset(N, dataset_size, min_sample, max_sample)
        sorted_inputs, sorted_outputs = bubblesort(train_inputs, train_outputs)
        print_dataset(sorted_inputs, sorted_outputs)
        
        dataset_inputs.append(sorted_inputs)
        dataset_outputs.append(sorted_outputs)
        
        # One photon addition on mode 0 and two-mode observable = 6 ladder operators
        # OBSERVABLE: Number operator of first and second mode -> (a1*a1) + (a2*a2)
        observable_modes.append([[0,0], [1,1]])#, [2,2], [3,3]])
        observable_types.append([[1,0], [1,0]])#, [1,0], [1,0]])
        
        
# === QNN TRAINING ===
nn_idx = 0
for N in list_N:
    for layers in list_layers:
        print(f'\n\n===== FOR N={N}, l={layers} =====')

        cv_qnns.append(QNN(N, layers, observable_modes[nn_idx], observable_types[nn_idx]))
        
        total_error = []
        training_QNN = partial(cv_qnns[nn_idx].train_QNN, inputs_dataset=dataset_inputs[nn_idx], outputs_dataset=dataset_outputs[nn_idx])
        result = opt.minimize(training_QNN, parameters[nn_idx], method='L-BFGS-B', callback=callback)#, tol=1e-5))#, options=options))
        #minimizer_kwargs = {"method": "L-BFGS-B", "tol": 1e-5, "options": options}
        #result = opt.basinhopping(training_energy, m,minimizer_kwargs=minimizer_kwargs, niter=30)#, niter=1000)
        
        print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}')
        print(result.fun)
        
        final_pars.append(result.x)
        list_loss.append(total_error)
        
        nn_idx += 1
        
        
# === QNN TESTING ===
qnns_outputs = np.zeros((len(list_N)*len(list_layers), dataset_size))
error = np.zeros((len(list_N)*len(list_layers), dataset_size))

for i in range(len(list_N)):
    N = list_N[i]
    modes = N
    for j in range(len(list_layers)):
        layers = list_layers[j]

        # Training set
        #test_inputs = dataset_inputs[i*len(list_layers)+j]
        #test_outputs = dataset_outputs[i*len(list_layers)+j]

        # Testing set
        #test_inputs, test_outputs = f1_2var_generate_dataset(N, dataset_size, min_sample, max_sample)
        test_inputs, test_outputs = f1_2var_linear_dataset(N, dataset_size, min_sample, max_sample)

        # Ascending order of the outputs 
        test_inputs, test_outputs = bubblesort(test_inputs, test_outputs)

        # Build the QNN with the trained parameters
        qnn = cv_qnns[i*len(list_layers)+j]
        
        params = final_pars[i*len(list_layers)+j]
        qnn.build_QNN(params)

        # Test the trained QNN
        for k in range(len(test_inputs)):
            qnn_output = np.real_if_close(qnn.eval_QNN(test_inputs[k]).sum())
            qnns_outputs[i*len(list_layers)+j, k] = qnn_output
            error[i*len(list_layers)+j, k] = (test_outputs[k] - qnn_output)**2
        
        # Plot results    
        plt.plot(test_outputs,'go',label='Expected results')
        plt.plot(qnns_outputs[i*len(list_layers)+j],'r',label='QNN results')
        plt.title(f'Modes = {list_N[i]}, Layers = {list_layers[j]}')
        plt.legend()
        plt.show()
        
        plt.plot(np.log(np.array(list_loss[i*len(list_layers)+j])+1), 'r', label='Loss (logarithmic) function')
        plt.title(f'Modes = {list_N[i]}, Layers = {list_layers[j]}')
        plt.gca().set_ylim(bottom=0)
        plt.legend()
        plt.show()
        
        print('\n--- OPTIMIZATION ERROR ---')
        print(error[i*len(list_layers)+j].sum()/len(test_inputs))

        qnn_times = qnn.qnn_profiling.avg_times()
        print('\nTime usage per stage:')
        total_time = sum(list(qnn_times.values()))
        for part_time in qnn_times:
            print(f'\t {part_time}: {np.round(100 * qnn_times[part_time] / total_time, 3)} %')
        plt.figure(figsize=(14,5))
        plt.bar(list(qnn_times.keys()), list(qnn_times.values()), color ='maroon')
        plt.xlabel("Time category")
        plt.ylabel("Time (s)")
        plt.title("QNN training times")
        plt.show()

        print(f'\nTotal number of training iterations: {len(qnn.qnn_profiling.gauss_times)}')
        print(f'\tNumber of trace expressions: {len(qnn.ladder_modes)*len(qnn.ladder_modes[0])}')
        print(f'\tNumber of perfect matchings per expression: {len(qnn.perf_matchings)}')
        print(f'\t{len(qnn.perf_matchings)*len(qnn.ladder_modes)*len(qnn.ladder_modes[0])} total summations with {qnn.layers + 1} products per summation.')
        #plt.plot(qnn.qnn_profiling.gauss_times, 'o')
        #plt.show()
        
