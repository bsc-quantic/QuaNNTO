import numpy as np

from quannto.utils.results_utils import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
qnns_modes = [2,2]
qnns_ladder_modes = [[[0]], [[0,0]]]
qnns_layers = [1,1]
qnns_is_addition = [False, False]
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'catstates'
in_norm_ranges = [None]*len(qnns_modes) # or ranges (a, b)

# === DATASET SETTINGS === (Statistics of a quantum cat state)
phi = 0.0
dataset_size = 1
input_range = (-1, 1)
alpha_list = np.linspace(input_range[0], input_range[1], dataset_size)
task_name = f'catstate_phi{phi}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}'

# Training dataset containing the statistical moments of the target cat state
with open(f"datasets/catstate_phi{phi}_trainsize{dataset_size}_rng{input_range[0]}to{input_range[-1]}_inputs.npy", "rb") as f:
    inputs = np.load(f)
with open(f"datasets/catstate_phi{phi}_trainsize{dataset_size}_rng{input_range[0]}to{input_range[-1]}_outputs.npy", "rb") as f:
    outputs = np.load(f)
train_dataset = [np.array(inputs), np.array(outputs)]

# === PLOT RESULTS FROM QONNs ===
train_losses = []
qnn_outs = []
legend_labels = []
for (N, l, ladder_modes, is_addition, in_norm_range) in zip(qnns_modes, qnns_layers, qnns_ladder_modes, qnns_is_addition, in_norm_ranges):
    # === NAME AND LEGEND OF THE QONN MODEL ===
    model_name = task_name + "_N" + str(N) + "_L" + str(l) + ("_add" if is_addition else "_sub") + str(ladder_modes) + "_in" + str(in_norm_range)
    nongauss_op = "â†" if is_addition else "â"
    legend_labels.append(f'N={N}, L={l}, {nongauss_op} in modes {np.array(ladder_modes[0])+1}')
    
    # === LOAD QONN MODEL RESULTS ===
    with open(f"quannto/tasks/train_losses/{model_name}.npy", "rb") as f:
        train_loss = np.load(f)
    #with open(f"quannto/tasks/testing_results/{model_name}.npy", "rb") as f:
    #    qnn_test_outputs = np.load(f)

    train_losses.append(train_loss.copy())
    #qnn_outs.append(qnn_test_outputs.copy())
    
# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
plot_qnns_loglosses(train_losses, None, legend_labels, filename)