from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import os.path

from quannto.core.qnn_trainers import *
from quannto.dataset_gens.synth_datasets import *
from quannto.utils.results_utils import *
from quannto.core.data_processors import *
from quannto.core.loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
modes = [2]
qnns_ladder_modes = [[[0],[1]]]
layers = [2]
is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'position'
#in_norm_ranges = [(-3, 3)]*len(modes)
#out_norm_ranges = [(1, 3)]*len(modes)
in_norm_ranges = [None]*len(modes)
out_norm_ranges = [None]*len(modes)
loss_function = mse

# === OPTIMIZER SETTINGS ===
optimizer = hybrid_build_and_train_model
basinhopping_iters = 1
params = None

# === TARGET FUNCTION SETTINGS ===
target_function = sin_1in_1out
input_range = (-6.3, 6.3)
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
valid_dataset = None

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
plt.savefig("figures/trainset_"+target_function.__name__+"_size"+str(trainset_size)+"_range"+str(input_range)+".pdf")
plt.show()
plt.clf()

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, input_range)
# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
train_losses = []
valid_losses = []
qnn_loss = []
qnn_outs = []
qnns = []
for (N, l, ladder_modes, in_norm_range, out_norm_range) in zip(modes, layers, qnns_ladder_modes, in_norm_ranges, out_norm_ranges):
    # Initialize the desired data processors for pre/post-processing
    in_preprocessors = []
    if in_norm_range != None:
        in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))

    out_preprocessors = []
    postprocessors = []
    if out_norm_range != None:
        out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))
        postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

    model_name = target_function.__name__ + "_N" + str(N) + "_L" + str(l) + "_ph" + str(ladder_modes) + "_in" + str(in_norm_range) + "_out" + str(out_norm_range)
    # Build the QNN and train it with the generated dataset
    qnn, train_loss, valid_loss = optimizer(model_name, N, l, n_inputs, n_outputs, ladder_modes, is_addition, observable,
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
    
    qnn_test_outputs, loss_value = qnn.test_model(test_dataset[0], test_dataset[1], loss_function)
    print(f'\n==========\nTESTING LOSS FOR N={N}, L={l}, LADDER MODES={ladder_modes}: {loss_value}\n==========')
    with open(f"testing/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_test_outputs))
    #print(qnn_test_outputs)
    qnn_outs.append(qnn_test_outputs.copy())
    
# PLOT RESULTS
if is_addition:
    nongauss_op = "â†"
else:
    nongauss_op = "â"
legend_labels = [f'N={qnn.N}, L={qnn.layers}, {nongauss_op} in modes {np.array(qnn.ladder_modes[0])+1}' for qnn in qnns] # α ∈ {in_norm_range}

colors = matplotlib.cm.tab10(range(len(modes)))
linestyles = [
     #(0, (1, 1)),
     (5, (10, 3)),
     (0, (3, 1, 1, 1)),
     (0, (5, 5)),
     (0, (3, 1, 1, 1, 1, 1)),
     (0, (5, 1)),
     (0, (3, 5, 1, 5))]

plt.plot(test_dataset[0], test_dataset[1], c='black', linewidth=6.0, alpha=0.2, label='Expected results')
c=0
for (qnn_test_outputs, legend_label, linestyle) in zip(qnn_outs, legend_labels, linestyles):
    plt.plot(test_dataset[0], qnn_test_outputs, c=colors[c], linestyle=linestyle, linewidth=1.8, label=legend_label)
    c+=1
plt.title(f'TESTING SET')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
#plt.ylim(top=output_range[1] + len(qnns)*0.2 + 0.2)
plt.grid(linestyle='--', linewidth=0.4)
#plt.legend(loc='upper right')
plt.legend()
plt.savefig("figures/test_res_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(qnns_ladder_modes)+"_in"+str(input_range)+".pdf")
plt.show()
plt.clf()

c=0
for (train_loss, valid_loss, legend_label, linestyle) in zip(train_losses, valid_losses, legend_labels, linestyles):
    x_log = np.log(np.array(range(1,len(train_loss)+1)))
    plt.plot(np.log(np.array(train_loss)+1), c=colors[c], linestyle=linestyle, label=f'Train loss {legend_label}')#, {input_range} disp')
    plt.plot(np.log(np.array(valid_loss)+1), c=colors[c], linestyle='dotted', label=f'Validation loss {legend_label}')#, {input_range} disp')
    c+=1

plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'LOGARITHMIC TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/log_loss_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(qnns_ladder_modes)+"_in"+str(input_range)+".pdf")
plt.show()
plt.clf()

c=0
for (train_loss, valid_loss, qnn, linestyle) in zip(train_losses, valid_losses, qnns, linestyles):
    plt.plot(np.array(train_loss), c=colors[c], linestyle=linestyle, label=f'Train loss {legend_label}')
    plt.plot(np.array(valid_loss), c=colors[c], linestyle='dotted', label=f'Validation loss {legend_label}')
    c+=1
    
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/loss_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(qnns_ladder_modes)+"_in"+str(input_range)+".pdf")
#plt.show()
plt.clf()

print('=== MINIMAL LOSSES ACHIEVED ===')
for i in range(len(legend_labels)):
    print(f'{legend_labels[i]}: {qnn_loss[i]}')