from functools import partial
import numpy as np
import os.path

from quannto.core.qnn_trainers import *
from quannto.utils.results_utils import *
from quannto.core.data_processors import *
from quannto.core.loss_functions import *

np.random.seed(42)

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

# === OPTIMIZER SETTINGS ===
optimize = hybrid_build_and_train_model
loss_function = cross_entropy
basinhopping_iters = 5
params = None

# === DATASET SETTINGS ===
categories = [0, 1, 2, 3, 4]
num_cats = len(categories)
dataset_size = 75*num_cats
validset_size = 20*num_cats
continuize_method = 'encoding' # 'pca' or 'encoding' for Autoencoder
task_name = f'mnist_{continuize_method}_{n_inputs}lat_{num_cats}cats'
     
# 1. FULL DATASET: Load or build (and save) a CV-preprocessed MNIST dataset and shuffle
if os.path.isfile(f"datasets/{task_name}_inputs.npy"):
    with open(f"datasets/{task_name}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{task_name}_outputs.npy", "rb") as f:
        outputs = np.load(f)
    dataset = [inputs, outputs]
    input_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
else:
    while True:
        dataset = pca_mnist(n_inputs, categories) if continuize_method == 'pca' else autoencoder_mnist(n_inputs, categories)
        input_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
        if np.all(input_ranges[:,-1] > 0):
            break
    with open(f"datasets/{task_name}_inputs.npy", "wb") as f:
        np.save(f, dataset[0])
    with open(f"datasets/{task_name}_outputs.npy", "wb") as f:
        np.save(f, dataset[1])
shuffling = np.random.permutation(len(dataset[0]))
shuffled_dataset = [dataset[0][shuffling], dataset[1][shuffling]]
# 2. TRAINING DATASET
train_dataset = (shuffled_dataset[0][:dataset_size], shuffled_dataset[1][:dataset_size])
# 3. VALIDATION DATASET (None for no validation)
valid_dataset = (shuffled_dataset[0][dataset_size : dataset_size+validset_size], shuffled_dataset[1][dataset_size : dataset_size+validset_size])
# 4. TESTING DATASET: Use the entire MNIST dataset
test_dataset = (dataset[0], dataset[1])
test_outputs_cats = dataset[1].reshape((len(dataset[1])))

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
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
    
    # === PREPROCESSORS AND POSTPROCESSORS ===
    in_preprocessors = []
    if in_norm_range != None:
        in_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=input_ranges, rescale_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))

    out_preprocessors = []
    out_preprocessors.append(partial(one_hot_encoding, num_cats=num_cats))
    if out_norm_range != None:
        out_preprocessors.append(partial(rescale_data, data_range=(0, 1), scale_data_range=out_norm_range))

    postprocessors = []
    postprocessors.append(partial(softmax_discretization))
    postprocessors.append(partial(greatest_probability))
    postprocessors.append(partial(np.ravel))

    # === BUILD, TRAIN AND TEST QNN ===
    qnn, train_loss, valid_loss = optimize(model_name, N, l, n_inputs, n_outputs, ladder_modes, is_addition, observable, 
                                           include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                           train_dataset, valid_dataset, loss_function, basinhopping_iters,
                                           in_preprocessors, out_preprocessors, postprocessors, init_pars=params)
    qnn_preds, loss_value = qnn.test_model(test_dataset[0], test_dataset[1], loss_function)
    qnn_hits = np.equal(qnn_preds, test_outputs_cats).sum()
    accuracy = qnn_hits/len(qnn_preds)
    
    qnns.append(qnn)
    train_losses.append(train_loss.copy())
    valid_losses.append(valid_loss.copy())
    qnns_preds.append(qnn_preds.copy())
    qnns_accuracies.append(accuracy)
    print(f"==========\nACCURACY FOR N={N}, L={l}, LADDER MODES={ladder_modes}: {qnn_hits}/{len(qnn_preds)} = {accuracy}\n==========\n")

    # === SAVE QNN MODEL RESULTS ===
    with open(f"quannto/tasks/train_losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(train_loss))
    with open(f"quannto/tasks/valid_losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(valid_loss))
    with open(f"quannto/tasks/testing_results/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_preds))
    plot_confusion_matrix(model_name, test_outputs_cats, qnn_preds)

# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
fig, ax, acc = plot_per_class_accuracy_hist(categories, test_outputs_cats, qnns_preds, legend_labels=legend_labels, filename=filename, title="MNIST — QONNs per-class accuracy")
fig, ax, acc = plot_per_class_accuracy_markers(categories, test_outputs_cats, qnns_preds, qnns_accuracies, legend_labels=legend_labels, filename=filename, title="MNIST — QONNs per-class accuracy")
