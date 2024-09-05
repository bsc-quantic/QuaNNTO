from functools import partial
import matplotlib.pyplot as plt

from .qnn import test_model, build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

# === HYPERPARAMETERS DEFINITION ===
modes = [3]
photon_additions = [[], [0], [0,1]]
layers = 1
is_input_reupload = False
n_inputs = 1
n_outputs = 1
observable = 'number'
in_norm_range = (0, 1)
out_norm_range = (0.5, 5)
loss_function = mse
noise = 0.3

# === TARGET FUNCTION SETTINGS ===
target_function = sin_cos_function
trainset_size = 60
testset_size = 200
validset_size = 20
input_range = (0, 1)
real_function = generate_linear_dataset_of(target_function, n_inputs, n_outputs, trainset_size*100, input_range)
output_range = get_range(real_function[1])

# Generate a training dataset of the target function to be learned
#train_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, trainset_size, input_range)
train_dataset = generate_noisy_samples(trainset_size, target_function, input_range[0], input_range[1], noise)

train_dataset = bubblesort(np.reshape(train_dataset[1], (trainset_size)), np.reshape(train_dataset[0], (trainset_size)))
train_dataset = (train_dataset[1].reshape((trainset_size, 1)), train_dataset[0].reshape((trainset_size, 1)))

input_range = get_range(train_dataset[0])

# Generate a validation dataset of the target function
valid_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, validset_size, input_range)

plt.plot(train_dataset[0], train_dataset[1], 'go', label='Noisy training set')
plt.plot(real_function[0], real_function[1], 'b', label='Real function')
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
for N in modes:
    for ph_add in photon_additions:
        model_name = target_function.__name__
        #in_norm_range = (1/N, N)
        #out_norm_range = (1, (N**2 + N**(-2) - 2) / 2)
        #out_norm_range = (1, N)

        # Initialize the desired data processors for pre/post-processing
        in_preprocessors = []
        #in_preprocessors.append(partial(trigonometric_one_input))
        #input_range = (0, 1)
        #in_preprocessors.append(partial(trigonometric_feature_expressivity, num_final_features=N))
        #input_range = (-N, N)
        in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))

        out_preprocessors = []
        out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

        postprocessors = []
        postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

        # Build the QNN and train it with the generated dataset
        qnn, train_loss, valid_loss = build_and_train_model(model_name, N, layers, n_inputs, n_outputs, ph_add, observable, is_input_reupload,
                                                            train_dataset, valid_dataset, loss_function, in_preprocessors, out_preprocessors, postprocessors)#, init_pars=np.ones((12)))
        train_losses.append(train_loss[1:].copy())
        valid_losses.append(valid_loss[1:].copy())
        qnn_test_outputs = test_model(qnn, test_dataset, loss_function)
        #print(qnn_test_outputs)
        qnn_outs.append(qnn_test_outputs.copy())
    
plt.plot(test_dataset[0], test_dataset[1], 'go', label='Expected results')
for (qnn_test_outputs, ph_add, color) in zip(qnn_outs, photon_additions, colors):
    plt.plot(test_dataset[0], qnn_test_outputs, c=color, linestyle='dashed', label=f'N={N}, {len(ph_add)} photons')
plt.title(f'TESTING SET')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.show()

for (train_loss, valid_loss, ph_add, color) in zip(train_losses, valid_losses, photon_additions, colors):
    plt.plot(np.log(np.array(train_loss)+1), c=color, label=f'Train loss N={N}, {len(ph_add)} photons')
    plt.plot(np.log(np.array(valid_loss)+1), c=color, linestyle='dashed', label=f'Validation loss N={N}, {len(ph_add)} photons')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.show()