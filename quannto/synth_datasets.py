import numpy as np

def test_function_1in_1out(inputs):
    return 0.7*inputs[0]**3 + 1.7*inputs[0]**2 + 2.4*inputs[0] + 5
    
def test_function_2in_1out(inputs):
    return 2.5*inputs[0]**2 + 1.2*inputs[1]**2 + 0.5*inputs[0]*inputs[1] + 3

def test_function_3in_1out(inputs):
    return 0.45*inputs[2] + 2.5*inputs[0]**2 + 1.2*inputs[2]*inputs[1]**2 + 0.5*inputs[0]*inputs[1] + 3

def log_function_1in_1out(inputs):
    return 2 + 5*np.log(inputs[0])/inputs[0]

def hyperbola_1in_1out(inputs):
    return 1 + 2/inputs[0]

def exp_1in_1out(inputs):
    return np.e**inputs[0] + 1

def sin_1in_1out(inputs):
    return 1.5 + np.sin(inputs[0])

def sin_cos_function(x):
    return -np.sin(10*x) + 3*np.cos(18*x) - 8*((x-1/2)**2) + 5/4

def generate_dataset_of(target_function, num_inputs, num_samples, input_range):
    inputs = np.zeros((num_samples, num_inputs))
    outputs = np.zeros((num_samples))
    
    for i in range(num_samples):
        inputs[i] =  np.random.uniform(low=input_range[0], high=input_range[1], size=(num_inputs))
        outputs[i] = target_function(inputs[i])
    
    return [inputs, outputs]

def generate_linear_dataset_of(target_function, num_inputs, num_samples, input_range):
    inputs = np.zeros((num_samples, num_inputs))
    outputs = np.zeros((num_samples))
    input_lin_sp = np.linspace(input_range[0], input_range[1], num_samples)
    
    for i in range(num_samples):
        inputs[i] = np.array([input_lin_sp[i]]*num_inputs)
        outputs[i] = target_function(inputs[i])
    
    return [inputs, outputs]

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

def replicate_inputs(inputs_set, n_inputs):
    rep_inps = np.zeros((len(inputs_set), n_inputs))
    for sample in range(len(inputs_set)):
        rep_inps[sample] = np.array(list(inputs_set[sample])*n_inputs)
    return rep_inps
 
        
def print_dataset(inputs, outputs):
    print('\n')
    for i in range(len(outputs)):
        print(f'Sample {i+1}\n INPUTS: {inputs[i]} OUTPUTS: {outputs[i]}')