from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os.path

from .qnn_trainers import build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
modes = [2,2,2]#,2,3,4]
qnns_ladder_modes = [[[]], [[0]], [[0,1]]]#, [[0],[0]], [[0,1,2]], [[0,1,2,3]]]
layers = [1,1,1]#,2,1,1]
is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'position'
in_norm_ranges = [(-2, 2)]*len(modes)
out_norm_ranges = [(-2, 2)]*len(modes)
loss_function = mse
basinhopping_iters = 0
params = None

# === TARGET FUNCTION SETTINGS ===
target_function = trig_fun
input_range = (-1, 1.5)
trainset_noise = 0.1
trainset_size = 100
testset_size = 200
validset_size = 50

# Generate a training dataset of the target function to be learned
if os.path.isfile(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{trainset_noise}_rng{input_range[0]}to{input_range[1]}_inputs.npy"):
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{trainset_noise}_rng{input_range[0]}to{input_range[1]}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{trainset_noise}_rng{input_range[0]}to{input_range[1]}_outputs.npy", "rb") as f:
        outputs = np.load(f)
    train_dataset = [inputs, outputs]
else:
    train_dataset = generate_noisy_samples(trainset_size, target_function, input_range[0], input_range[1], trainset_noise)
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{trainset_noise}_rng{input_range[0]}to{input_range[1]}_inputs.npy", "wb") as f:
        np.save(f, train_dataset[0])
    with open(f"datasets/{target_function.__name__}_trainsize{trainset_size}_noise{trainset_noise}_rng{input_range[0]}to{input_range[1]}_outputs.npy", "wb") as f:
        np.save(f, train_dataset[1])

train_dataset = bubblesort(np.reshape(train_dataset[1], (trainset_size)), np.reshape(train_dataset[0], (trainset_size)))
train_dataset = (train_dataset[1].reshape((trainset_size, 1)), train_dataset[0].reshape((trainset_size, 1)))

# Generate a validation dataset of the target function
valid_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, validset_size, input_range)
#valid_dataset = None

# Generate a linear dataset without noise to plot the real function
real_function = generate_linear_dataset_of(target_function, n_inputs, n_outputs, trainset_size*100, input_range)
output_range = get_range(real_function[1])

plt.plot(train_dataset[0], train_dataset[1], 'go', label='Noisy training set')
plt.plot(real_function[0], real_function[1], 'b', label='Real function')
#plt.plot(valid_dataset[0], valid_dataset[1], 'ro', label='Validation set')
plt.title(f'Dataset')
plt.xlabel('x')
plt.xlim()
plt.ylabel('f(x)')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend(loc='upper right')
plt.savefig("figures/trainset_"+target_function.__name__+"_size"+str(trainset_size)+"_range"+str(input_range)+".png")
#plt.show()
plt.clf()

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, input_range)
# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
#colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
#colors = colormaps['tab10']
colors = matplotlib.cm.tab10(range(len(modes)))
train_losses = []
valid_losses = []
qnn_loss = []
qnn_outs = []
qnns = []
for (N, l, ladder_modes, in_norm_range, out_norm_range) in zip(modes, layers, qnns_ladder_modes, in_norm_ranges, out_norm_ranges):
    # Initialize the desired data processors for pre/post-processing
    in_preprocessors = []
    in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))

    out_preprocessors = []
    out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

    postprocessors = []
    postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

    model_name = target_function.__name__ + "_N" + str(N) + "_L" + str(l) + "_ph" + str(ladder_modes) + "_in" + str(in_norm_range) + "_out" + str(out_norm_range)
    # Build the QNN and train it with the generated dataset
    qnn, train_loss, valid_loss = build_and_train_model(model_name, N, l, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                                        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                                        train_dataset, valid_dataset, loss_function, basinhopping_iters,
                                                        in_preprocessors, out_preprocessors, postprocessors, init_pars=params)
    qnns.append(qnn)
    train_losses.append(train_loss.copy())
    with open(f"losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(train_loss))
    valid_losses.append(valid_loss.copy())
    with open(f"valid_losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(valid_loss))
    qnn_loss.append(train_loss[-1])
    
    qnn_test_outputs = qnn.test_model(test_dataset, loss_function)
    with open(f"testing/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_test_outputs))
    #print(qnn_test_outputs)
    qnn_outs.append(qnn_test_outputs.copy())
    
# PLOT RESULTS
if is_addition:
    nongauss_op = "â†"
else:
    nongauss_op = "â"
legend_labels = [f'N={qnn.N}, L={qnn.layers}, {nongauss_op} in modes {np.array(qnn.ladder_modes[0]) + 1}' for qnn in qnns] # α ∈ {in_norm_range}

plt.plot(test_dataset[0], test_dataset[1], 'o', markerfacecolor='g', markeredgecolor='none', alpha=0.35, label='Expected results')
c=0
for (qnn_test_outputs, legend_label) in zip(qnn_outs, legend_labels):
    plt.plot(test_dataset[0], qnn_test_outputs, c=colors[c], linestyle='dashed', label=legend_label)
    c+=1
plt.title(f'TESTING SET')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
#plt.ylim(top=output_range[1] + len(qnns)*0.2 + 0.2)
plt.grid(linestyle='--', linewidth=0.4)
#plt.legend(loc='upper right')
plt.legend()
plt.savefig("figures/test_res_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(qnns_ladder_modes)+"_in"+str(input_range)+".png")
plt.show()
plt.clf()

c=0
for (train_loss, valid_loss, legend_label) in zip(train_losses, valid_losses, legend_labels):
    x_log = np.log(np.array(range(1,len(train_loss)+1)))
    plt.plot(np.log(np.array(train_loss)+1), c=colors[c], label=f'Train loss {legend_label}')#, {input_range} disp')
    plt.plot(np.log(np.array(valid_loss)+1), c=colors[c], linestyle='dashed', label=f'Validation loss {legend_label}')#, {input_range} disp')
    c+=1

plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'LOGARITHMIC TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/log_loss_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(qnns_ladder_modes)+"_in"+str(input_range)+".png")
plt.show()
plt.clf()

c=0
for (train_loss, valid_loss, qnn) in zip(train_losses, valid_losses, qnns):
    plt.plot(np.array(train_loss), c=colors[c], label=f'Train loss {legend_label}')
    plt.plot(np.array(valid_loss), c=colors[c], linestyle='dashed', label=f'Validation loss {legend_label}')
    c+=1
    
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/loss_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(qnns_ladder_modes)+"_in"+str(input_range)+".png")
#plt.show()
plt.clf()

print(f'MINIMAL LOSSES ACHIEVED: {qnn_loss}')