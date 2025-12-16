from functools import partial

from quannto.core.qnn import test_model, build_and_train_model
from quannto.dataset_gens.synth_datasets import *
from quannto.utils.results_utils import *
from quannto.core.data_processors import *

# === HYPERPARAMETERS DEFINITION ===
N = 4
layers = 1
observable_modes = [[0,0]]
observable_types = [[1,0]]
is_input_reupload = True
n_inputs = 1
n_outputs = 1

# === TARGET FUNCTION DEFINITION ===
target_function = sin_cos_function
dataset_size = 80
input_range = (0, 1)
output_range = get_range(generate_linear_dataset_of(target_function, n_inputs, n_outputs, dataset_size*100, input_range)[1])
in_norm_range = (1.25, 5.5)
out_norm_range = (1, 7)

model_name = target_function.__name__
testing_set_size = 200

# === PREPROCESSORS AND POSTPROCESSORS ===
in_preprocessors = []
in_preprocessors.append(partial(trigonometric_feature_expressivity, num_final_features=N))
trig_range = (-N, N)
in_preprocessors.append(partial(rescale_data, data_range=trig_range, scale_data_range=in_norm_range))

out_preprocessors = []
out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

postprocessors = []
postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

# === BUILD, TRAIN AND TEST QNN ===
# Generate a dataset of the target function to be learned
train_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, dataset_size, input_range)
#train_df = pd.read_csv("curve_fitting_datasets/sincos_trainset.csv")
#x_train = train_df.iloc[:, 0:n_inputs].to_numpy()
#y_train = train_df.iloc[:, n_inputs:].to_numpy()
#train_dataset = [x_train, y_train]

# Build the QNN and train it with the generated dataset
qnn, loss = build_and_train_model(model_name, N, layers, n_inputs, n_outputs, observable_modes, observable_types, 
                            is_input_reupload, train_dataset, in_preprocessors, out_preprocessors, postprocessors)
plt.plot(np.log(np.array(loss)+1), label=f'N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
plt.legend()
plt.show()

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testing_set_size, input_range)
#test_df = pd.read_csv("curve_fitting_datasets/sincos_testset_lin.csv")
#x_test = test_df.iloc[:, 0:n_inputs].to_numpy()
#y_test = test_df.iloc[:, n_inputs:].to_numpy()
#test_dataset = [x_test, y_test]
qnn_test_outputs = test_model(qnn, test_dataset)
plot_qnn_testing(qnn, test_dataset[1], qnn_test_outputs)
