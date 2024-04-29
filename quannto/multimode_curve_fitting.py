from functools import partial
import matplotlib.pyplot as plt

from .qnn import test_model, build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *

# === HYPERPARAMETERS DEFINITION ===
modes = [2,3,4,5]
layers = 1
observable_modes = [[0,0]]
observable_types = [[1,0]]
is_input_reupload = True
n_inputs = 1
n_outputs = 1

# === TARGET FUNCTION DEFINITION ===
target_function = sin_cos_function
dataset_size = 70
input_range = (0, 1)
output_range = get_range(generate_linear_dataset_of(target_function, n_inputs, n_outputs, dataset_size*100, input_range)[1])
in_norm_range = (1.1, 4)
out_norm_range = (1, 7)

testing_set_size = 200

# Generate a dataset of the target function to be learned
train_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, dataset_size, input_range)
testing_range = get_range(train_dataset[0])
#input_range = get_range(train_dataset[0])
#train_df = pd.read_csv("curve_fitting_datasets/sincos_trainset_features.csv")
#x_train = train_df.iloc[:, 0:n_inputs].to_numpy()
#y_train = train_df.iloc[:, -n_outputs:].to_numpy()
#train_dataset = [x_train, y_train]

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testing_set_size, testing_range)
#test_df = pd.read_csv("curve_fitting_datasets/sincos_testset_lin_features.csv")
#x_test = test_df.iloc[:, 0:n_inputs].to_numpy()
#y_test = test_df.iloc[:, -n_outputs:].to_numpy()
#test_dataset = [x_test, y_test]
plt.plot(test_dataset[1], 'go', label='Expected results')

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
#colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
colors = ['red', 'fuchsia', 'blue', 'purple']
losses = []
for (N, color) in zip(modes, colors):
    model_name = target_function.__name__
    #in_norm_range = (1/N, N)
    #out_norm_range = (1, (N**2 + N**(-2) - 2) / 2)
    #out_norm_range = (1, N)

    # Initialize the desired data processors for pre/post-processing
    in_preprocessors = []
    in_preprocessors.append(partial(trigonometric_feature_expressivity, num_final_features=N))
    input_range = (-N, N)
    in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))

    out_preprocessors = []
    out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

    postprocessors = []
    postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

    # Build the QNN and train it with the generated dataset
    qnn, loss = build_and_train_model(model_name, N, layers, n_inputs, n_outputs, observable_modes, observable_types, 
                                is_input_reupload, train_dataset, in_preprocessors, out_preprocessors, postprocessors)
    losses.append(loss.copy())
    qnn_test_outputs = test_model(qnn, test_dataset)
    plt.plot(qnn_test_outputs, c=color, linestyle='dashed', label=f'N={N}')
plt.title(f'TESTING SET')
plt.xlabel('Input samples')
plt.ylabel('Output')
plt.legend()
plt.show()

for (loss, color, N) in zip(losses, colors, modes):
    plt.plot(np.log(np.array(loss)+1), c=color, label=f'N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
plt.legend()
plt.show()