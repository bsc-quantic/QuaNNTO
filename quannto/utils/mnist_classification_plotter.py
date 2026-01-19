import numpy as np

from quannto.utils.results_utils import *

# === HYPERPARAMETERS DEFINITION ===
qnns_modes = [6, 6]
qnns_ladder_modes = [[[0]], [[5]]]
qnns_layers = [1, 1]
qnns_is_addition = [False, False]
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 3
n_outputs = 5
observable = 'position'
in_norm_ranges = [(-3, 3)]*len(qnns_modes) # or ranges (a, b)
out_norm_ranges = [(1, 3)]*len(qnns_modes)

# === DATASET SETTINGS ===
categories = [0, 1, 2, 3, 4]
num_cats = len(categories)
dataset_size = 75*num_cats
validset_size = 20*num_cats
continuize_method = 'pca' # 'pca' or 'encoding' for Autoencoder
task_name = f'mnist_{continuize_method}_{n_inputs}lat_{num_cats}cats'
     
# 1. FULL DATASET: Load the CV-preprocessed MNIST dataset and shuffle
with open(f"datasets/{task_name}_inputs.npy", "rb") as f:
    inputs = np.load(f)
with open(f"datasets/{task_name}_outputs.npy", "rb") as f:
    outputs = np.load(f)
dataset = [inputs, outputs]
input_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
test_dataset = (dataset[0], dataset[1])
test_outputs_cats = dataset[1].reshape((len(dataset[1])))

# === PLOT RESULTS FROM QONNs ===
qnns = []
train_losses = []
valid_losses = []
qnns_preds = []
qnns_accuracies = []
legend_labels = []
for (N, l, ladder_modes, is_addition, in_norm_range, out_norm_range) in zip(qnns_modes, qnns_layers, qnns_ladder_modes, qnns_is_addition, in_norm_ranges, out_norm_ranges):
    # === NAME AND LEGEND OF THE QONN MODEL ===
    model_name = task_name + "_N" + str(N) + "_L" + str(l) + ("_add" if is_addition else "_sub") + str(ladder_modes) + "_in" + str(in_norm_range) + "_out" + str(out_norm_range)
    nongauss_op = "â†" if is_addition else "â"
    legend_labels.append(f'N={N}, L={l}, {nongauss_op} in modes {np.array(ladder_modes[0])+1}')
    
    # === LOAD QONN MODEL RESULTS ===
    with open(f"quannto/tasks/train_losses/{model_name}.npy", "rb") as f:
        train_loss = np.load(f)
    with open(f"quannto/tasks/valid_losses/{model_name}.npy", "rb") as f:
        valid_loss = np.load(f)
    with open(f"quannto/tasks/testing_results/{model_name}.npy", "rb") as f:
        qnn_preds = np.load(f)
    
    qnn_hits = np.equal(qnn_preds, test_outputs_cats).sum()
    accuracy = qnn_hits/len(qnn_preds)
    
    train_losses.append(train_loss.copy())
    valid_losses.append(valid_loss.copy())
    qnns_preds.append(qnn_preds.copy())
    qnns_accuracies.append(accuracy)
    plot_confusion_matrix(model_name, test_outputs_cats, qnn_preds)

# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
fig, ax, acc = plot_per_class_accuracy_hist(categories, test_outputs_cats, qnns_preds, legend_labels=legend_labels, filename=filename, title="MNIST — QONNs per-class accuracy")
fig, ax, acc = plot_per_class_accuracy_markers(categories, test_outputs_cats, qnns_preds, qnns_accuracies, legend_labels=legend_labels, filename=filename, title="MNIST — QONNs per-class accuracy")
plot_qnns_loglosses(train_losses, valid_losses, legend_labels, filename)
