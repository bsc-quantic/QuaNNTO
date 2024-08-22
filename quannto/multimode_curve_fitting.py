from functools import partial
import matplotlib.pyplot as plt

from .qnn import test_model, build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *

# === HYPERPARAMETERS DEFINITION ===
modes = [2,3,4]
photon_additions = [0]
layers = 1
is_input_reupload = True
n_inputs = 1
n_outputs = 1
observable = 'number'
in_norm_range = (-2, 2)
out_norm_range = (1, 5)

# === TARGET FUNCTION SETTINGS ===
target_function = sin_cos_function
trainset_size = 100
input_range = (0, 1)
output_range = get_range(generate_linear_dataset_of(target_function, n_inputs, n_outputs, trainset_size*100, input_range)[1])
testset_size = 200

# Generate a training dataset of the target function to be learned
train_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, trainset_size, input_range)
dataset_range = get_range(train_dataset[0])
#input_range = get_range(train_dataset[0])
#train_df = pd.read_csv("curve_fitting_datasets/sincos_trainset_features.csv")
#x_train = train_df.iloc[:, 0:n_inputs].to_numpy()
#y_train = train_df.iloc[:, -n_outputs:].to_numpy()
#train_dataset = [x_train, y_train]

# Generate a validation dataset of the target function
validset_size = 40
valid_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, validset_size, dataset_range)

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, dataset_range)
#test_df = pd.read_csv("curve_fitting_datasets/sincos_testset_lin_features.csv")
#x_test = test_df.iloc[:, 0:n_inputs].to_numpy()
#y_test = test_df.iloc[:, -n_outputs:].to_numpy()
#test_dataset = [x_test, y_test]

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
#colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
colors = ['red', 'fuchsia', 'blue', 'purple']
train_losses = []
valid_losses = []
qnn_outs = []
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
    qnn, train_loss, valid_loss = build_and_train_model(model_name, N, layers, n_inputs, n_outputs, photon_additions, observable, is_input_reupload,
                                                        train_dataset, valid_dataset, in_preprocessors, out_preprocessors, postprocessors)#, init_pars=np.ones((12)))
    train_losses.append(train_loss.copy())
    valid_losses.append(valid_loss.copy())
    qnn_test_outputs = test_model(qnn, test_dataset)
    #print(qnn_test_outputs)
    qnn_outs.append(qnn_test_outputs.copy())
    
plt.plot(test_dataset[1], 'go', label='Expected results')
for (qnn_test_outputs, color, N) in zip(qnn_outs, colors, modes):
    plt.plot(qnn_test_outputs, c=color, linestyle='dashed', label=f'N={N}')
plt.title(f'TESTING SET')
plt.xlabel('Input samples')
plt.xlim()
plt.ylabel('Output')
plt.legend()
plt.show()

for (train_loss, valid_loss, color, N) in zip(train_losses, valid_losses, colors, modes):
    plt.plot(np.log(np.array(train_loss)+1), c=color, label=f'Train loss N={N}')
    plt.plot(np.log(np.array(valid_loss)+1), c=color, linestyle='dashed', label=f'Validation loss N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC TRAINING AND VALIDATION LOSS')
plt.legend()
plt.show()