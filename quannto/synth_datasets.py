import numpy as np

def test_function_1in_1out(inputs):
    return 0.7*inputs[0]**3 + 1.7*inputs[0]**2 + 2.4*inputs[0] + 5
    
def test_function_2in_1out(inputs):
    return 2.5*inputs[0]**2 + 1.2*inputs[1]**2 + 0.5*inputs[0]*inputs[1] + 3

def log_function_1in_1out(inputs):
    return 2 + 5*np.log(inputs[0])/inputs[0]

def hyperbola_1in_1out(inputs):
    return 1 + 2/inputs[0]

def exp_1in_1out(inputs):
    return np.e**inputs[0] + 1

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

def normalize_dataset(dataset, input_range, output_range, in_norm_range, out_norm_range):
    inputs, outputs = dataset[0], dataset[1]
    num_samples = len(inputs)
    
    norm_inputs = np.zeros((num_samples, len(inputs[0])))
    norm_outputs = np.zeros((num_samples))
    
    for i in range(num_samples):
        norm_inputs[i] = in_norm_range[0] + (in_norm_range[1] - in_norm_range[0]) * (inputs[i] - input_range[0]) / (input_range[1] - input_range[0])
        norm_outputs[i] = out_norm_range[0] + (out_norm_range[1] - out_norm_range[0]) * (outputs[i] - output_range[0]) / (output_range[1] - output_range[0])
        
    return [norm_inputs, norm_outputs]

def denormalize_outputs(norm_outputs, output_range, out_norm_range):
    return output_range[0] + (output_range[1] - output_range[0]) * (norm_outputs - out_norm_range[0]) / (out_norm_range[1] - out_norm_range[0])

def get_outputs_range(dataset_outputs):
    return (np.min(dataset_outputs), np.max(dataset_outputs))

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