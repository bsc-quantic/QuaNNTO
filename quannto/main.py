from qnn import test_model, build_and_train_model
from synth_datasets import *
from results_utils import *

# === HYPERPARAMETERS DEFINITION ===
N = 2
layers = 2
observable_modes = [[0,0], [1,1]]
observable_types = [[1,0], [1,0]]

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

model_name = target_function.__name__
testing_set_size = 500
# ===================================

# 1. Generate, sort ascending-wise and print a dataset of the target function to be trained
dataset = generate_dataset_of(target_function, num_inputs, dataset_size, input_range)
norm_dataset = normalize_dataset(dataset, input_range, output_range, in_norm_range, out_norm_range)
sorted_inputs, sorted_outputs = bubblesort(norm_dataset[0], norm_dataset[1])
print_dataset(sorted_inputs, sorted_outputs)

# 2. Build the QNN and train it with the generated dataset
qnn = build_and_train_model(model_name, N, layers, observable_modes, observable_types, [sorted_inputs, sorted_outputs])

# 3. Generate a testing linearly-spaced dataset of the target function to test the trained QNN
testing_set = generate_linear_dataset_of(target_function, num_inputs, testing_set_size, input_range)
norm_test_set = normalize_dataset(testing_set, input_range, output_range, in_norm_range, out_norm_range)
test_inputs, test_outputs = bubblesort(norm_test_set[0], norm_test_set[1])
qnn_test_outputs = test_model(qnn, [test_inputs, test_outputs])
plot_qnn_testing(qnn, testing_set[1], denormalize_outputs(qnn_test_outputs, output_range, out_norm_range))

