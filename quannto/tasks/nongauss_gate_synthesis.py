from functools import partial
import numpy as np

from quannto.core.qnn_trainers import *
from quannto.utils.results_utils import *
from quannto.core.data_processors import *
from quannto.core.loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
qnns_modes = [2,2]
qnns_ladder_modes = [[[]],[[1]]]
qnns_layers = [1,1]
qnns_is_addition = [False, False, False]
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'cubicphase'
in_norm_ranges = [None]*len(qnns_modes) # or ranges (a, b)

# === OPTIMIZER SETTINGS ===
optimize = hybrid_build_and_train_model
loss_function = mse
basinhopping_iters = 2
params = None

# === DATASET SETTINGS === (Statistics of specific Non-Gaussian gate action over target quantum states)
gamma = 0.2
dataset_size = 50
input_range = (-2, 2)
alpha_list = np.linspace(input_range[0], input_range[1], dataset_size)
task_name = f'fock_cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}'

# Training dataset of the non-Gaussian gate to be learned
with open(f"datasets/{task_name}_inputs.npy", "rb") as f:
    inputs = np.load(f)
with open(f"datasets/{task_name}_outputs.npy", "rb") as f:
    outputs = np.load(f)
train_dataset = [inputs, outputs]

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
qnns = []
train_losses = []
qnns_outs = []
legend_labels = []
for (N, l, ladder_modes, is_addition, in_norm_range) in zip(qnns_modes, qnns_layers, qnns_ladder_modes, qnns_is_addition, in_norm_ranges):
    # === NAME AND LEGEND OF THE QONN MODEL ===
    model_name = task_name + "_N" + str(N) + "_L" + str(l) + ("_add" if is_addition else "_sub") + str(ladder_modes) + "_in" + str(in_norm_range)
    nongauss_op = "â†" if is_addition else "â"
    legend_labels.append(f'N={N}, L={l}, {nongauss_op} in modes {np.array(ladder_modes[0])+1}')
    
    # === PREPROCESSORS AND POSTPROCESSORS ===
    in_preprocessors = []
    if in_norm_range != None:
        in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))
    out_preprocessors = []
    postprocessors = []

    # === BUILD, TRAIN AND TEST QNN ===
    qnn, train_loss, valid_loss = optimize(model_name, N, l, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                           include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                           train_dataset, None, loss_function, basinhopping_iters,
                                           in_preprocessors, out_preprocessors, postprocessors, init_pars=params)
    qnn_test_outputs, loss_value = qnn.test_model(train_dataset[0], train_dataset[1], loss_function)
    
    qnns.append(qnn)
    train_losses.append(train_loss.copy())
    qnns_outs.append(qnn_test_outputs.copy())
    
    # === SAVE QNN MODEL RESULTS ===
    with open(f"quannto/tasks/train_losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(train_loss))
    with open(f"quannto/tasks/testing_results/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_test_outputs))
        
# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
plot_qnns_loglosses(train_losses, None, legend_labels, filename)