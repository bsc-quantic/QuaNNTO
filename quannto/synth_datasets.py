import numpy as np

def f1_2var(inputs):
    #return 0.7*inputs[0]**3 + 1.7*inputs[0]**2 + 2.4*inputs[0] + 5
    #return 0.2*inputs[0]**5 + 0.3*inputs[0]**4 + 0.7*inputs[0]**3 + 1.7*inputs[0]**2 + 2.4*inputs[0] + 5
    #return 0.3 * (inputs[0]**np.log(inputs[0] + 1))
    return 0.08*inputs[0]**7 + 0.12*inputs[0]**6 + 0.2*inputs[0]**5 + 1.1*inputs[0]**4 + 0.7*inputs[0]**3 + 1.7*inputs[0]**2 + 2.4*inputs[0] + 5
    #return 2.5*inputs[0]**2 + 1.2*inputs[1]**2 + 0.5*inputs[0]*inputs[1] + 3 #TESTED!
    #return 0.5*inputs[0]**3 + 0.2*inputs[1]**3 + 1.1*inputs[1]**2 + 0.6*inputs[0]**2 + 0.5*inputs[0]*inputs[1] + 3
    
def f1_2var_generate_dataset(N, num_samples, start, end):
    n_in = 1
    inputs = np.zeros((num_samples, n_in))
    outputs = np.zeros((num_samples))

    norm_range = (5, 10)
    for i in range(num_samples):
        rand_inputs = np.random.uniform(low=start, high=end, size=(n_in))
        inputs[i] = norm_range[0] + norm_range[1] * (rand_inputs - start) / (end - start) #NORMALIZED INPUTS
        #inputs[i] = rand_inputs
        #outputs[i] = f1_2var(inputs[i])
        outputs[i] = f1_2var(rand_inputs)
    min_outputs = f1_2var(np.ones((n_in))*start) # f MUST BE MONOTONIC INCREASING
    max_outputs = f1_2var(np.ones((n_in))*end)
    outputs = norm_range[0] + norm_range[1] * (outputs - min_outputs) / (max_outputs - min_outputs)
    
    return inputs, outputs

def f1_2var_linear_dataset(N, num_samples, start, end):
    n_in = 1
    inputs = np.zeros((num_samples, n_in))
    outputs = np.zeros((num_samples))
    lin_space = np.linspace(start,end,num_samples)
    norm_range = (5, 10)
    for i in range(num_samples):
        inputs[i] = norm_range[0] + norm_range[1] * (lin_space[i] - lin_space[0]) / (lin_space[-1] - lin_space[0])
        #inputs[i] = (np.random.uniform(low=start, high=end, size=(N)) - start) / (end - start) #NORMALIZED INPUTS
        outputs[i] = f1_2var([lin_space[i]]*n_in)
    min_outputs = f1_2var(np.ones((n_in))*start)
    max_outputs = f1_2var(np.ones((n_in))*end)
    outputs = norm_range[0] + norm_range[1] * (outputs - min_outputs) / (max_outputs - min_outputs)   
    
    return inputs, outputs

def bubblesort(inputs, outputs):
    array = np.copy(outputs)
    inp = np.copy(inputs)
    for iter_num in range(len(array)-1,0,-1):
        for idx in range(iter_num):
            if array[idx]>array[idx+1]:
                temp = array[idx]
                array[idx] = array[idx+1]
                array[idx+1] = temp
                temp_inp = np.copy(inp[idx])
                inp[idx] = np.copy(inp[idx+1])
                inp[idx+1] = temp_inp
    return inp, array

def print_dataset(inputs, outputs):
    print('\n')
    for i in range(len(outputs)):
        print(f'Sample {i+1}\n INPUT: {inputs[i]} OUTPUT {outputs[i]}')