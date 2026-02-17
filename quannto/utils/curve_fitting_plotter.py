import numpy as np

from quannto.dataset_gens.synthetic_datasets import *
from quannto.utils.results_utils import *

# === QONNs HYPERPARAMETERS ===
qnns_modes = [2,2,2]
qnns_ladder_modes = [[[0]], [[1]], [[0,1]]]
qnns_layers = [1,1,1]
qnns_is_addition = [False, False, False]
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'position'
in_norm_ranges = [(-3, 3)]*len(qnns_modes) # or [None, ...]
out_norm_ranges = [(1, 3)]*len(qnns_modes) # or [None, ...]

# === DATASET (TARGET FUNCTION) SETTINGS ===
target_function = cosh_1in_1out
input_range = (-5, 5)
trainset_noise = 5
trainset_size = 100
testset_size = 200
task_name = f"curvefitting_{target_function.__name__}_trainsize{trainset_size}_noise{trainset_noise}_rng{input_range[0]}to{input_range[1]}"
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, input_range)

# === PLOT RESULTS FROM QONNs ===
train_losses = []
valid_losses = []
qnns_preds = []
legend_labels = []
for (N, l, ladder_modes, is_addition, in_norm_range, out_norm_range) in zip(qnns_modes, qnns_layers, qnns_ladder_modes, qnns_is_addition, in_norm_ranges, out_norm_ranges):
    # === NAME AND LEGEND OF THE QONN MODEL ===
    model_name = task_name + "_N" + str(N) + "_L" + str(l) + ("_add" if is_addition else "_sub") + str(ladder_modes) + "_in" + str(in_norm_range) + "_out" + str(out_norm_range)
    nongauss_op = "â†" if is_addition else "â"
    legend_labels.append(f'N={N}, L={l}, {nongauss_op} in modes {np.array(ladder_modes[0])+1}')

    # === LOAD QONN MODEL RESULTS ===
    with open(f"quannto/tasks/models/train_losses/{model_name}.npy", "rb") as f:
        train_loss = np.load(f)
    with open(f"quannto/tasks/models/valid_losses/{model_name}.npy", "rb") as f:
        valid_loss = np.load(f)
    with open(f"quannto/tasks/models/testing_results/{model_name}.npy", "rb") as f:
        qnn_pred = np.load(f)
    train_losses.append(train_loss.copy())
    valid_losses.append(valid_loss.copy())
    qnns_preds.append(qnn_pred.copy())

# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
plot_qnns_testing(test_dataset[0], test_dataset[1], qnns_preds, legend_labels, filename)
plot_qnns_loglosses(train_losses, valid_losses, legend_labels, filename)