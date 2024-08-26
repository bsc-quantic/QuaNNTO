from functools import partial
import matplotlib.pyplot as plt

from .qnn import test_model, build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

# === HYPERPARAMETERS DEFINITION ===
modes = [4,5,6]
photon_additions = [0]
layers = 1
is_input_reupload = False
n_inputs = 1
n_outputs = 1
observable = 'number'
in_norm_range = (-2, 2)
out_norm_range = (1, 5)
loss_function = mse

# === TARGET FUNCTION SETTINGS ===
target_function = sin_cos_function
trainset_size = 20
input_range = (0, 1)
output_range = get_range(generate_linear_dataset_of(target_function, n_inputs, n_outputs, trainset_size*100, input_range)[1])
testset_size = 200

# Generate a training dataset of the target function to be learned
train_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, trainset_size, input_range)
input_range = get_range(train_dataset[0])

# Generate a validation dataset of the target function
validset_size = 20
valid_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, validset_size, input_range)

plt.plot(train_dataset[0], train_dataset[1], 'go', label='Training set')
#plt.plot(valid_dataset[0], valid_dataset[1], 'ro', label='Validation set')
plt.title(f'Dataset')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.show()

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, input_range)

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
                                                        train_dataset, valid_dataset, loss_function, in_preprocessors, out_preprocessors, postprocessors)#, init_pars=np.ones((12)))
    train_losses.append(train_loss.copy())
    valid_losses.append(valid_loss.copy())
    qnn_test_outputs = test_model(qnn, test_dataset, loss_function)
    #print(qnn_test_outputs)
    qnn_outs.append(qnn_test_outputs.copy())
    
plt.plot(test_dataset[1], 'go', label='Expected results')
for (qnn_test_outputs, color, N) in zip(qnn_outs, colors, modes):
    plt.plot(qnn_test_outputs, c=color, linestyle='dashed', label=f'N={N}')
plt.title(f'TESTING SET')
plt.xlabel('Input samples')
plt.xlim()
plt.ylabel('Output')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.show()

for (train_loss, valid_loss, color, N) in zip(train_losses, valid_losses, colors, modes):
    plt.plot(np.log(np.array(train_loss)+1), c=color, label=f'Train loss N={N}')
    plt.plot(np.log(np.array(valid_loss)+1), c=color, linestyle='dashed', label=f'Validation loss N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.show()