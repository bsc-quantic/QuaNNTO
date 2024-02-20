from qnn import test_model, build_and_train_model, QNN
from synth_datasets import *
from results_utils import *
from preprocessing import *

# === HYPERPARAMETERS DEFINITION ===
N = 4
layers = 1
observable_modes = [[0,0]]
observable_types = [[1,0]]
is_input_reupload = True

# === SYNTHETIC DATASET PARAMETERS ===
target_function = sin_cos_function
num_inputs = 1
dataset_size = 80
# Minimum and maximum values the inputs/outputs can take
input_range = (0.01, 1)
output_range = get_range(generate_linear_dataset_of(target_function, num_inputs, dataset_size*20, input_range)[1])
# Minimum and maximum values the inputs/outputs are normalized between
in_norm_range = (0.5, 3.5)
out_norm_range = (1, 5)

model_name = target_function.__name__
testing_set_size = 100
# ===================================

# 1. Generate, sort ascending-wise and print a dataset of the target function to be learned
inputs, outputs = generate_dataset_of(target_function, num_inputs, dataset_size, input_range)
inputs = trigonometric_feature_expressivity(inputs, N)
norm_inputs = normalize_data(inputs, input_range, in_norm_range)
norm_outputs = normalize_data(outputs, output_range, out_norm_range)
sorted_inputs, sorted_outputs = bubblesort(norm_inputs, norm_outputs)
print_dataset(sorted_inputs, sorted_outputs)

# 2. Build the QNN and train it with the generated dataset
#qnn = build_and_train_model(model_name, N, layers, observable_modes, observable_types, is_input_reupload, [sorted_inputs, sorted_outputs])
qnn = QNN.load_model("model_N4_L1_sin_cos_function.txt")

# 3. Generate a testing linearly-spaced dataset of the target function to test the trained QNN
test_inputs, test_outputs = generate_linear_dataset_of(target_function, num_inputs, testing_set_size, input_range)
test_inputs = trigonometric_feature_expressivity(test_inputs, N)
norm_test_inputs = normalize_data(test_inputs, input_range, in_norm_range)
norm_test_outputs = normalize_data(test_outputs, output_range, out_norm_range)
qnn_test_outputs = test_model(qnn, [norm_test_inputs, norm_test_outputs])
plot_qnn_testing(qnn, test_outputs, denormalize_data(qnn_test_outputs, output_range, out_norm_range))

