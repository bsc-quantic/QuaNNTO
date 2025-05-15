from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os.path

from .qnn_trainers import build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

#np.random.seed(27)

# === HYPERPARAMETERS DEFINITION ===
modes = [2]
photon_additions = [[0]]
layers = [1]
is_input_reupload = False
n_inputs = 1
n_outputs = 1
observable = 'position'
in_norm_range = (-2, 2)
out_norm_range = (-2, 2)
loss_function = mse
basinhopping_iters = 1
noise = 0.1

# === TARGET FUNCTION SETTINGS ===
target_function = sin_1in_1out
trainset_size = 100
testset_size = 200
validset_size = 50
input_range = (0, 6.3)
real_function = generate_linear_dataset_of(target_function, n_inputs, n_outputs, trainset_size*100, input_range)
output_range = get_range(real_function[1])

# Generate a training dataset of the target function to be learned
if os.path.isfile(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{noise}_rng{input_range[0]}to{input_range[1]}_inputs.npy"):
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{noise}_rng{input_range[0]}to{input_range[1]}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{noise}_rng{input_range[0]}to{input_range[1]}_outputs.npy", "rb") as f:
        outputs = np.load(f)
    train_dataset = [inputs, outputs]
else:
    train_dataset = generate_noisy_samples(trainset_size, target_function, input_range[0], input_range[1], noise)
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{noise}_rng{input_range[0]}to{input_range[1]}_inputs.npy", "wb") as f:
        np.save(f, train_dataset[0])
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{noise}_rng{input_range[0]}to{input_range[1]}_outputs.npy", "wb") as f:
        np.save(f, train_dataset[1])

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
plt.savefig("figures/trainset_"+target_function.__name__+"_size"+str(trainset_size)+".png")
plt.show()

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, input_range)

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
#colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
colors = colormaps['tab10']
train_losses = []
valid_losses = []
qnn_loss = []
qnn_outs = []
qnns = []
for N in modes:
    for l in layers:
        for ph_add in photon_additions:
            # Initialize the desired data processors for pre/post-processing
            in_preprocessors = []
            #in_preprocessors.append(partial(trigonometric_feature_expressivity, num_final_features=N))
            #input_range = (-N, N)
            in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))

            out_preprocessors = []
            out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

            postprocessors = []
            postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

            model_name = target_function.__name__
            # Build the QNN and train it with the generated dataset
            qnn, train_loss, valid_loss = build_and_train_model(model_name, N, l, n_inputs, n_outputs, ph_add, observable, is_input_reupload,
                                                                train_dataset, valid_dataset, loss_function, basinhopping_iters,
                                                                in_preprocessors, out_preprocessors, postprocessors)
            qnns.append(qnn)
            train_losses.append(train_loss.copy())
            valid_losses.append(valid_loss.copy())
            qnn_loss.append(train_loss[-1])
            qnn_test_outputs = qnn.test_model(test_dataset, loss_function)
            #print(qnn_test_outputs)
            qnn_outs.append(qnn_test_outputs.copy())
    
plt.plot(test_dataset[0], test_dataset[1], 'o', markerfacecolor='g', markeredgecolor='none', alpha=0.25, label='Expected results')
c = 0
for (qnn_test_outputs, qnn) in zip(qnn_outs, qnns):
    plt.plot(test_dataset[0], qnn_test_outputs, c=colors(c+3%10), linestyle='dashed', label=f'N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer')
    c+=1
plt.title(f'TESTING SET')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/test_res_"+model_name+"_N"+str(modes[0])+".png")
plt.show()

c=0
for (train_loss, valid_loss, qnn) in zip(train_losses, valid_losses, qnns):
    x_log = np.log(np.array(range(1,len(train_loss)+1)))
    plt.plot(np.log(np.array(train_loss)+1), c=colors(c+3%10), label=f'Train loss N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer')
    plt.plot(np.log(np.array(valid_loss)+1), c=colors(c+3%10), linestyle='dashed', label=f'Validation loss N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer')
    c+=1

plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'LOGARITHMIC TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/log_loss_"+model_name+"_N"+str(modes[0])+".png")
plt.show()

c=0
for (train_loss, valid_loss, qnn) in zip(train_losses, valid_losses, qnns):
    plt.plot(np.array(train_loss), c=colors(c%10), label=f'Train loss N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer')
    plt.plot(np.array(valid_loss), c=colors(c%10), linestyle='dashed', label=f'Validation loss N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer')
    c+=1
    
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/loss_"+model_name+"_N"+str(modes[0])+".png")
plt.show()

print(f'MINIMAL LOSSES ACHIEVED: {qnn_loss}')