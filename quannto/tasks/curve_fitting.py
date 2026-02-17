from functools import partial
import numpy as np
import os.path

from quannto.core.qnn_trainers import *
from quannto.dataset_gens.synthetic_datasets import *
from quannto.utils.results_utils import *
from quannto.core.data_processors import *
from quannto.core.loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
qnns_modes = [2,3,4]
qnns_ladder_modes = [[[1]], [[1,2]], [[1,2,3]]]
qnns_layers = [1,1,1]
qnns_is_addition = [False, False, False]
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'position'
#in_norm_ranges = [None]*len(qnns_modes) # or ranges (a, b)
in_norm_ranges = [(-3, 3)]*len(qnns_modes)
#out_norm_ranges = [None]*len(qnns_modes) # or ranges (a, b)
out_norm_ranges = [(1, 3)]*len(qnns_modes)

# === OPTIMIZER SETTINGS ===
optimize = hybrid_build_and_train_model
loss_function = mse
basinhopping_iters = 4
params = None

# === DATASET (TARGET FUNCTION) SETTINGS ===
target_function = trig_fun
input_range = (-1, 2.5)
trainset_noise = 0.1
trainset_size = 100
testset_size = 200
validset_size = 50
task_name = f"curvefitting_{target_function.__name__}_trainsize{trainset_size}_noise{trainset_noise}_rng{input_range[0]}to{input_range[1]}"

# 1. TRAINING DATASET: Load or generate (and save) a randomly-sampled and noisy dataset of the target function
if os.path.isfile(f"datasets/{task_name}_inputs.npy"):
    with open(f"datasets/{task_name}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{task_name}_outputs.npy", "rb") as f:
        outputs = np.load(f)
    train_dataset = [inputs, outputs]
else:
    train_dataset = generate_noisy_samples(trainset_size, target_function, input_range[0], input_range[1], trainset_noise)
    with open(f"datasets/{task_name}_inputs.npy", "wb") as f:
        np.save(f, train_dataset[0])
    with open(f"datasets/{task_name}_outputs.npy", "wb") as f:
        np.save(f, train_dataset[1])
input_range = (np.min(train_dataset[0]), np.max(train_dataset[0]))
# 2. VALIDATION DATASET: Generate a randomly-sampled and noiseless dataset of the target function (None for no validation)
valid_dataset = generate_dataset_of(target_function, n_inputs, n_outputs, validset_size, input_range)
# 3. TESTING DATASET: Generate a linearly-spaced and noiseless dataset of the target function
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, input_range)
plot_noisy_dataset(task_name, train_dataset, test_dataset)

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
qnns = []
train_losses = []
valid_losses = []
qnns_preds = []
qnns_norms = []
legend_labels = []
for (N, l, ladder_modes, is_addition, in_norm_range, out_norm_range) in zip(qnns_modes, qnns_layers, qnns_ladder_modes, qnns_is_addition, in_norm_ranges, out_norm_ranges):
    # === NAME AND LEGEND OF THE QONN MODEL ===
    model_name = task_name + "_N" + str(N) + "_L" + str(l) + ("_add" if is_addition else "_sub") + str(ladder_modes) + "_in" + str(in_norm_range) + "_out" + str(out_norm_range)
    nongauss_op = "â†" if is_addition else "â"
    legend_labels.append(f'N={N}, L={l}, {nongauss_op} in modes {np.array(ladder_modes[0])+1}')
    
    # === PREPROCESSORS AND POSTPROCESSORS ===
    in_preprocessors = []
    if in_norm_range != None:
        in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))

    out_preprocessors = []
    postprocessors = []
    if out_norm_range != None:
        output_range = get_range(test_dataset[1])
        out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))
        postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

    # === BUILD, TRAIN AND TEST QNN ===
    qnn, train_loss, valid_loss = optimize(model_name, N, l, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                           include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                           train_dataset, valid_dataset, loss_function, basinhopping_iters,
                                           in_preprocessors, out_preprocessors, postprocessors, init_pars=params)
    qnn_pred, norm, loss_value = qnn.test_model(test_dataset[0], test_dataset[1], loss_function)
    
    qnns.append(qnn)
    train_losses.append(train_loss.copy())
    valid_losses.append(valid_loss.copy())
    qnns_preds.append(qnn_pred.copy())
    qnns_norms.append(norm)
    print(f'\n==========\nTESTING LOSS FOR N={N}, L={l}, LADDER MODES={ladder_modes}: {loss_value}\n==========')
    
    # === SAVE QNN MODEL RESULTS ===
    with open(f"quannto/tasks/models/train_losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(train_loss))
    with open(f"quannto/tasks/models/valid_losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(valid_loss))
    with open(f"quannto/tasks/models/testing_results/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_pred))

# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
plot_qnns_testing(test_dataset[0], test_dataset[1], qnns_preds, legend_labels, filename)
plot_qnns_loglosses(train_losses, valid_losses, legend_labels, filename)