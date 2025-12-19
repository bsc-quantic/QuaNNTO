from functools import partial
import numpy as np
import os.path

from quannto.core.qnn_trainers import *
from quannto.utils.results_utils import *
from quannto.core.data_processors import *
from quannto.core.loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
qnn_modes = 3
qnn_ladder_modes = [[0]]
qnn_layers = 1
qnn_is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 2
n_outputs = 2
observable = 'position'
in_norm_range = (-3, 3) # or None
out_norm_range = (1, 3) # or None

# === OPTIMIZER SETTINGS ===
optimize = hybrid_build_and_train_model
loss_function = cross_entropy
basinhopping_iters = 0
params = None

# === DATASET SETTINGS ===
dataset_name = 'circles' # or 'circles'
trainset_size = 100
validset_size = 50
num_cats = 2
dataset_size_per_cat = trainset_size // 2
validset_size_per_cat = validset_size // 2
model_name = f"{dataset_name}_{qnn_modes}modes_{qnn_layers}layers_ph{qnn_ladder_modes}_in{in_norm_range}_out{out_norm_range}_datasize{trainset_size}"

# 1. FULL DATASET: Load moons (or circles) dataset and shuffle
if os.path.isfile(f"datasets/{dataset_name}_inputs.npy"):
    with open(f"datasets/{dataset_name}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{dataset_name}_outputs.npy", "rb") as f:
        outputs = np.load(f)
    shuffling = np.random.permutation(len(inputs))
    dataset = [inputs[shuffling], outputs[shuffling]]
    input_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
else:
    Exception
    
# 2. BALANCED TRAINING DATASET
train_dataset = (
    np.concatenate((
        dataset[0][:dataset_size_per_cat], dataset[0][-dataset_size_per_cat:]
    )),
    np.concatenate(
        (dataset[1][:dataset_size_per_cat], dataset[1][-dataset_size_per_cat:])
    )
)
# 3. BALANCED VALIDATION DATASET (None for no validation)
valid_dataset = (
    np.concatenate((
        dataset[0][dataset_size_per_cat:dataset_size_per_cat+validset_size_per_cat],
        dataset[0][-(dataset_size_per_cat+validset_size_per_cat):-dataset_size_per_cat]
    )),
    np.concatenate((
        dataset[1][:dataset_size_per_cat], dataset[1][-dataset_size_per_cat:]
    ))
)
# 4. TESTING DATASET: Use the entire moons (or circles) dataset
test_dataset = (dataset[0], dataset[1])
test_outputs_classes = dataset[1].ravel()

# === PREPROCESSORS AND POSTPROCESSORS ===
in_preprocessors = []
if in_norm_range != None:
    in_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=input_ranges, rescale_range=in_norm_range))
in_preprocessors.append(partial(pad_data, length=2*qnn_modes))

out_preprocessors = []
out_preprocessors.append(partial(one_hot_encoding, num_cats=num_cats))
if out_norm_range != None:
    out_preprocessors.append(partial(rescale_data, data_range=(0, 1), scale_data_range=out_norm_range))

postprocessors = []
#postprocessors.append(partial(softmax_discretization))
#postprocessors.append(partial(greatest_probability))
#postprocessors.append(partial(np.ravel))

# === BUILD, TRAIN AND TEST QNN ===
qnn, train_loss, valid_loss = optimize(model_name, qnn_modes, qnn_layers, n_inputs, n_outputs, qnn_ladder_modes, qnn_is_addition, observable,
                                        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                        train_dataset, valid_dataset, loss_function, basinhopping_iters, in_preprocessors, out_preprocessors, postprocessors)
qnn_preds, loss_value = qnn.test_model(test_dataset[0], test_dataset[1], loss_function)
qnn_probs_preds = softmax_discretization(qnn_preds)
qnn_class_preds = np.ravel(greatest_probability(qnn_probs_preds))

qnn_hits = np.equal(qnn_class_preds, test_outputs_classes).sum()
qnn_accuracy = qnn_hits/len(qnn_class_preds)
print(f"Accuracy: {qnn_hits}/{len(qnn_class_preds)} = {qnn_accuracy}")

# === SAVE AND PLOT QNN MODEL RESULTS ===
with open(f"quannto/tasks/train_losses/{model_name}.npy", "wb") as f:
    np.save(f, np.array(train_loss))
with open(f"quannto/tasks/valid_losses/{model_name}.npy", "wb") as f:
    np.save(f, np.array(valid_loss))
with open(f"quannto/tasks/testing_results/{model_name}.npy", "wb") as f:
    np.save(f, np.array(qnn_class_preds))

nongauss_op = "â†" if qnn_is_addition else "â"
plot_title = f'QONN of N={qnn.N}, L={qnn.layers}, {nongauss_op} in modes {np.array(qnn.ladder_modes[0]) + 1}'
legend_label = f'N={qnn_modes}, L={qnn_layers}, {nongauss_op} in modes {np.array(qnn.ladder_modes[0]) + 1}'
plot_qnns_loglosses([train_loss], [valid_loss], [legend_label], model_name)
plot_qnn_decision(test_dataset[0], test_outputs_classes, qnn.evaluate_model, model_name, plot_title)
plot_confusion_matrix(model_name, test_outputs_classes, qnn_class_preds)