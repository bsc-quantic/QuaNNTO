import os
import numpy as np

from quannto.core.loss_functions import mse
from quannto.core.qnn import QNN
from quannto.dataset_gens.catstates_stats import build_catstates_dataset
from quannto.utils.results_utils import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
qnns_modes = [2,2,2,3,2]
qnns_ladder_modes = [[[0,0]],[[1,1]],[[0,1]],[[1,2]],[[1],[1]]]
#qnns_ladder_modes = [[[1]], [[1],[1]]]
qnns_layers = [1,1,1,1,2]
qnns_is_addition = [False,False,False,False,False]
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'catstates'
in_norm_ranges = [None]*len(qnns_modes) # or ranges (a, b)

# === DATASET SETTINGS === (Statistics of a quantum cat state)
phi = 0.0
cutoff = 20
dataset_size = 1
input_range = (-1, 1)
num_moments = 15
train_num_moments = 15
alpha_list = np.linspace(input_range[0], input_range[1], dataset_size)
dataset_name = f'catstate_phi{phi}_trainsize{dataset_size}_stats{num_moments}_cut{cutoff}_rng{alpha_list[0]}to{alpha_list[-1]}'
task_name = f'catstate_phi{phi}_trainsize{dataset_size}_stats{train_num_moments}_cut{cutoff}_rng{alpha_list[0]}to{alpha_list[-1]}'

# Training dataset containing the statistical moments of the target cat state
if os.path.isfile(f"datasets/{dataset_name}_inputs.npy"):
    with open(f"datasets/{dataset_name}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{dataset_name}_outputs.npy", "rb") as f:
        outputs = np.load(f)
else:
    raise FileNotFoundError("The requested dataset does not exist, generate it from quannto.dataset_gens.catstates_stats")
train_dataset = [np.array(inputs), np.array([outputs[0][:train_num_moments]])]

# === PLOT RESULTS FROM QONNs ===
train_losses = []
qnn_outs = []
legend_labels = []
for (N, l, ladder_modes, is_addition, in_norm_range) in zip(qnns_modes, qnns_layers, qnns_ladder_modes, qnns_is_addition, in_norm_ranges):
    # === NAME AND LEGEND OF THE QONN MODEL ===
    model_name = task_name + "_N" + str(N) + "_L" + str(l) + ("_add" if is_addition else "_sub") + str(ladder_modes) + "_in" + str(in_norm_range)
    nongauss_op = "â†" if is_addition else "â"
    legend_labels.append(f'N={N}, L={l}, {nongauss_op} in modes {np.array(ladder_modes[0])+1}')
    
    qnn = QNN.load_model(f"quannto/tasks/models/pickle_json/{model_name}.txt")
    qnn.print_qnn()
    qnn.save_operator_matrices(f"quannto/tasks/models/trained_operators")
    print('DATASET', train_dataset)
    res, norm, loss = qnn.test_model(train_dataset[0], train_dataset[1], mse)
    print('NORM: ', norm)
    total_loss = 0
    for moment_idx in range(train_num_moments):
        print(f"Moment {moment_idx+1}: expected={train_dataset[1][0][moment_idx]:.6f}, QONN={res[0][moment_idx]:.6f}, diff={(train_dataset[1][0][moment_idx]-res[0][moment_idx])**2:.6e}")
        total_loss += (train_dataset[1][0][moment_idx]-res[0][moment_idx])**2
    print('Total moment loss:', total_loss)
    # === LOAD QONN MODEL RESULTS ===
    with open(f"quannto/tasks/models/train_losses/{model_name}.npy", "rb") as f:
        train_loss = np.load(f)
    #with open(f"quannto/tasks/models/testing_results/{model_name}.npy", "rb") as f:
    #    qnn_test_outputs = np.load(f)

    train_losses.append(train_loss.copy())
    #qnn_outs.append(qnn_test_outputs.copy())

# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
plot_qnns_loglosses(train_losses, None, legend_labels, filename)