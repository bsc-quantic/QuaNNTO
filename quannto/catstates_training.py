from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .qnn_trainers import build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
modes = [2,2]
photon_additions = [[[0]], [[0,0]]]
layers = [1,1]
is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'catstates'
in_norm_ranges = [(-2, 2)]*len(modes)
out_norm_ranges = [(-2, 2)]*len(modes)
loss_function = mse
basinhopping_iters = 5
params = None

# === DATASET SETTINGS ===
phi = 0.0
dataset_size = 1
input_range = (-1, 1)
alpha_list = np.linspace(input_range[0], input_range[1], dataset_size)
model_name = f'catstate_phi{phi}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}'

# Load training dataset of the cubic phase gate to be learned
with open(f"datasets/catstate_phi{phi}_trainsize{dataset_size}_rng{input_range[0]}to{input_range[-1]}_inputs.npy", "rb") as f:
    inputs = np.load(f)
with open(f"datasets/catstate_phi{phi}_trainsize{dataset_size}_rng{input_range[0]}to{input_range[-1]}_outputs.npy", "rb") as f:
    outputs = np.load(f)

train_dataset = [np.array(inputs), np.array(outputs)]


# === BUILD, TRAIN AND TEST THE DIFFERENT QNN MODELS ===
colors = colormaps['tab10']
train_losses = []
valid_losses = []
qnn_loss = []
qnn_outs = []
qnns = []
for (N, l, ph_add, in_norm_range) in zip(modes, layers, photon_additions, in_norm_ranges):
    # Initialize the desired data processors for pre/post-processing
    in_preprocessors = []
    in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))
    out_preprocessors = []
    postprocessors = []

    model_name = model_name + "_N" + str(N) + "_L" + str(l) + "_ph" + str(ph_add)
    # Build the QONN and train it with the generated dataset
    qnn, train_loss, valid_loss = build_and_train_model(model_name, N, l, n_inputs, n_outputs, ph_add, is_addition, observable,
                                                        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                                        train_dataset, None, loss_function, basinhopping_iters,
                                                        in_preprocessors, out_preprocessors, postprocessors, init_pars=params)
    qnns.append(qnn)
    
    train_losses.append(train_loss.copy())
    with open(f"losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(train_loss))
    qnn_loss.append(train_loss[-1])
    
    qnn_test_outputs, loss_value = qnn.test_model(train_dataset[0], train_dataset[1], loss_function)
    with open(f"testing/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_test_outputs))
    qnn_outs.append(qnn_test_outputs.copy())
    
# PLOT RESULTS
if is_addition:
    nongauss_op = "â†"
else:
    nongauss_op = "â"
legend_labels = [f'N={qnn.N}, L={qnn.layers}, {nongauss_op} in modes {np.array(qnn.ladder_modes) + 1}' for qnn in qnns]

print('=== MINIMAL LOSSES ACHIEVED ===')
for i in range(len(legend_labels)):
    print(f'{legend_labels[i]}: {qnn_loss[i]}')

c=0
for (train_loss, legend_label) in zip(train_losses, legend_labels):
    x_log = np.log(np.array(range(1,len(train_loss)+1)))
    plt.plot(np.log(np.array(train_loss)+1), c=colors(c+3%10), label=f'Train loss {legend_label}')
    c+=1

plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'LOGARITHMIC LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.show()
plt.savefig("figures/logloss_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(photon_additions)+"_in"+str(input_range)+".pdf")
plt.clf()

print(f'MINIMAL LOSSES ACHIEVED: {qnn_loss}')