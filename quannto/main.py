import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize as opt
from functools import partial

from qnn import QNN
from synth_datasets import *

def plot_qnn_testing(qnn, exp_outputs, qnn_outputs):
    plt.plot(exp_outputs, 'go', label='Expected results')
    plt.plot(qnn_outputs, 'r', label='QNN results')
    plt.title(f'TESTING SET\nModel: {qnn.model_name}, Modes = {qnn.N}, Layers = {qnn.layers}')
    plt.legend()
    plt.show()
    
def plot_qnn_train_results(qnn, exp_outputs, qnn_outputs, loss_values):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'TRAINING SET\nModel: {qnn.model_name}, Modes = {qnn.N}, Layers = {qnn.layers}')
    
    # Plot expected vs obtained outputs of the training set
    ax1.plot(exp_outputs,'go',label='Expected results')
    ax1.plot(qnn_outputs,'r',label='QNN results')
    ax1.legend()
    
    # Plot training loss values
    ax2.plot(np.log(np.array(loss_values)+1), 'r', label='Loss (logarithmic) function')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    plt.show()
    
def show_times(qnn):
    qnn_times = qnn.qnn_profiling.avg_times()
    print('\nAverage time usage per stage:')
    total_time = sum(list(qnn_times.values()))
    for part_time in qnn_times:
        print(f'\t {part_time}: {np.round(100 * qnn_times[part_time] / total_time, 3)} %')
    print(f'\nTotal average time per iteration: {total_time}')
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
    
def test_model(qnn, testing_dataset, sort=False):
    error = np.zeros((len(testing_dataset[0])))
    qnn_outputs = np.zeros((len(testing_dataset[0])))
    
    if sort:
        # Ascending order of the outputs 
        test_inputs, test_outputs = bubblesort(testing_dataset[0], testing_dataset[1])
    else:
        test_inputs, test_outputs = testing_dataset[0], testing_dataset[1]
    
    # Evaluate all testing set
    for k in range(len(test_inputs)):
        qnn_outputs[k] = np.real_if_close(qnn.eval_QNN(test_inputs[k]).sum())
        error[k] = (test_outputs[k] - qnn_outputs[k])**2
    mean_error = error.sum()/len(error)
    print(f"MSE: {mean_error}")
    
    return qnn_outputs
    
def build_and_train_model(name, N, layers, observable_modes, observable_types, dataset, init_pars=None):
    if init_pars == None:
        init_pars = np.random.rand((layers-1)*(2*N**2) + 2*(N**2) + N)
    else:
        assert len(init_pars) == (layers-1)*(2*N**2) + 2*(N**2) + N
    
    def callback(xk):
        '''
        Callback function that prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QNN tunable parameters
        '''
        e = training_QNN(xk)
        print(e)
        loss_values.append(e)
    
    qnn = QNN("model_N" + str(N) + "_L" + str(layers) + "_" + name,
              N, layers, observable_modes, observable_types)
    
    train_inputs, train_outputs = dataset[0], dataset[1]
    
    loss_values = []
    training_QNN = partial(qnn.train_QNN, inputs_dataset=train_inputs, outputs_dataset=train_outputs)
    training_start = time.time()
    result = opt.minimize(training_QNN, init_pars, method='L-BFGS-B', callback=callback)
    print(f'Total training time: {time.time() - training_start} seconds')
    
    print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}')
    print(result.fun)
    
    qnn.build_QNN(result.x)
    
    qnn_outputs = test_model(qnn, dataset, sort=True)
    plot_qnn_train_results(qnn, dataset[1], qnn_outputs, loss_values)
    show_times(qnn)
    
    qnn.print_qnn()
    qnn.qnn_profiling.clear_times()
    qnn.save_model(qnn.model_name + ".txt")
    
    return qnn
    

# === HYPERPARAMETERS DEFINITION ===
N = 2
layers = 2
observable_modes = [[0,0], [1,1]]#, [2,2], [3,3]]
observable_types = [[1,0], [1,0]]#, [1,0], [1,0]]

# === SYNTHETIC DATASET PARAMETERS ===
dataset_size = 50
target_function = test_function_1in_1out
num_inputs = 1
# Minimum and maximum values the inputs/outputs can take
input_range = (1, 50)
output_range = get_outputs_range(generate_linear_dataset_of(target_function, num_inputs, dataset_size*20, input_range)[1])
# Minimum and maximum values the inputs/outputs are normalized between
in_norm_range = (2, 10)
out_norm_range = (5, 15)

# Name for the model
model_name = target_function.__name__

# 1. Generate, sort ascending-wise and print a dataset of the target function to be trained
dataset = generate_dataset_of(target_function, num_inputs, dataset_size, input_range)
norm_dataset = normalize_dataset(dataset, input_range, output_range, in_norm_range, out_norm_range)
sorted_inputs, sorted_outputs = bubblesort(norm_dataset[0], norm_dataset[1])
#sorted_inputs, sorted_outputs = dataset[0], dataset[1]
print_dataset(sorted_inputs, sorted_outputs)

# 2. Build the QNN and train it with the generated dataset
qnn = build_and_train_model(model_name, N, layers, observable_modes, observable_types, [sorted_inputs, sorted_outputs])

# 3. Generate a testing linearly-separed dataset of the target function to test the trained QNN
testing_set = generate_linear_dataset_of(target_function, num_inputs, dataset_size, input_range)
norm_test_set = normalize_dataset(testing_set, input_range, output_range, in_norm_range, out_norm_range)
qnn_test_outputs = test_model(qnn, [norm_test_set[0], norm_test_set[1]])
plot_qnn_testing(qnn, testing_set[1], denormalize_outputs(qnn_test_outputs, output_range, out_norm_range))

