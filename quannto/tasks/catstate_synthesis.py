from functools import partial
import os

from quannto.core.qnn_trainers import *
from quannto.utils.path_utils import datasets_dir, models_testing_results_path, models_train_losses_path
from quannto.utils.results_utils import *
from quannto.core.data_processors import *
from quannto.core.loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
qnns_modes = [3]
qnns_ladder_modes = [[[1,2]]]
qnns_layers = [1]
qnns_is_addition = [False]
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'catstates'
in_norm_ranges = [None]*len(qnns_modes) # or ranges (a, b)
qnn_params = [None]*len(qnns_modes) # or list of arrays with initial parameters for each QNN

# === OPTIMIZER SETTINGS ===
optimize = build_and_train_model
loss_function = mse
basinhopping_iters = 2

# === DATASET SETTINGS === (Statistics of a quantum cat state)
phi = 0.0
cutoff = 20
dataset_size = 1
input_range = (-1, 1)
num_moments = 15
train_num_moments = 15
alpha_list = np.linspace(input_range[0], input_range[1], dataset_size)
dataset_name = f'catstate_phi{phi}_trainsize{dataset_size}_stats{num_moments}_cut{cutoff}_rng{input_range[0]}to{input_range[-1]}'
task_name = f'catstate_phi{phi}_trainsize{dataset_size}_stats{train_num_moments}_cut{cutoff}_rng{input_range[0]}to{input_range[-1]}'
dataset_dir = str(datasets_dir() / dataset_name)

# Training dataset containing the statistical moments of the target cat state
if os.path.isfile(dataset_dir + "_inputs.npy") and os.path.isfile(dataset_dir + "_outputs.npy"):
    with open(dataset_dir + "_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(dataset_dir + "_outputs.npy", "rb") as f:
        outputs = np.load(f)
else:
    raise FileNotFoundError("The requested dataset does not exist, generate it from quannto.dataset_gens.catstates_stats")
train_dataset = [np.array(inputs), np.array([outputs[0][:train_num_moments]])]
print(train_dataset)

# === BUILD, TRAIN AND TEST THE DIFFERENT QNN MODELS ===
qnns = []
train_losses = []
qnn_outs = []
legend_labels = []
for (N, l, ladder_modes, is_addition, in_norm_range, params) in zip(qnns_modes, qnns_layers, qnns_ladder_modes, qnns_is_addition, in_norm_ranges, qnn_params):
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
    qnn, train_loss, valid_loss = optimize(model_name, task_name, N, l, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                           include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                           train_dataset, None, loss_function, basinhopping_iters,
                                           in_preprocessors, out_preprocessors, postprocessors, init_pars=params)
    qnn_test_outputs, norms, loss_value = qnn.test_model(train_dataset[0], train_dataset[1], loss_function)
    print("Expected moments:", train_dataset[1])
    print("QONN output moments:", qnn_test_outputs)
    print("Subtractions probability:", norms)

    qnns.append(qnn)
    train_losses.append(train_loss.copy())
    qnn_outs.append(qnn_test_outputs.copy())
    
    # === SAVE QNN MODEL RESULTS ===
    with open(models_train_losses_path(model_name, "txt"), "wb") as f:
        np.save(f, np.array(train_loss))
    with open(models_testing_results_path(model_name, "txt"), "wb") as f:
        np.save(f, np.array(qnn_test_outputs))
    
# === PLOT AND SAVE JOINT RESULTS ===
nongauss_ops = ['â†' if is_addition else 'â' for is_addition in qnns_is_addition]
filename = task_name+"_N"+str(qnns_modes)+"_L"+str(qnns_layers)+"_ph"+str(nongauss_ops)+str(qnns_ladder_modes)
plot_qnns_loglosses(train_losses, None, legend_labels, filename)