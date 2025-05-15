import numpy as np
import matplotlib.pyplot as plt

def plot_qnn_testing(qnn, exp_outputs, qnn_outputs):
    plt.plot(exp_outputs, 'go', label='Expected results')
    plt.plot(qnn_outputs, 'r', label='QNN results')
    plt.title(f'TESTING SET\nModel: {qnn.model_name}, Modes = {qnn.N}, Layers = {qnn.layers}')
    plt.legend()
    plt.show()
    
def plot_qnn_train_results(qnn, inputs, exp_outputs, qnn_outputs, loss_values):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'TRAINING SET\nModel: {qnn.model_name}, Modes = {qnn.N}, Layers = {qnn.layers}')
    
    # Plot expected vs obtained outputs of the training set
    ax1.plot(inputs, exp_outputs,'go',label='Expected results')
    ax1.plot(inputs, qnn_outputs,'r',label='QNN results')
    ax1.set_xlabel('Input (x)')
    ax1.set_ylabel('Output')
    ax1.grid(linestyle='--', linewidth=0.4)
    ax1.legend()
    
    # Plot training loss values
    ax2.plot(np.log(np.array(loss_values)+1), 'r', label='Loss (logarithmic) function')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Logarithmic loss value')
    ax2.grid(linestyle='--', linewidth=0.4)
    ax2.legend()
    plt.show()
    
def compute_times(qnn):
    qnn_times = qnn.qnn_profiling.avg_benchmark()
    print('\nAverage time per stage per iteration:')
    print(qnn_times)
    print('\nAverage time usage per stage:')
    total_time = sum(list(qnn_times.values()))
    for part_time in qnn_times:
        print(f'\t {part_time}: {np.round(100 * qnn_times[part_time] / total_time, 3)} %')
    print(f'\nTotal average time per iteration: {total_time}')
    return qnn_times
    
def show_times(qnn):
    qnn_times = compute_times(qnn)
    plt.figure(figsize=(14,5))
    plt.bar(list(qnn_times.keys()), list(qnn_times.values()), color ='maroon')
    plt.xlabel("Time category")
    plt.ylabel("Time (s)")
    plt.title("QNN training times")
    plt.show()

    print(f'\nTotal number of training iterations: {len(qnn.qnn_profiling.gauss_times)}')
    #print(f'\tNumber of trace expressions: {len(qnn.ladder_modes)*len(qnn.ladder_modes[0])}')
    #print(f'\tNumber of perfect matchings per expression: {len(qnn.perf_matchings)}')
    #print(f'\t{len(qnn.perf_matchings)*len(qnn.ladder_modes)*len(qnn.ladder_modes[0])} total summations with {qnn.layers + 1} products per summation.')
