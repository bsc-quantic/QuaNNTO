from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os.path

from .qnn_trainers import build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
N = 3
ladder_modes = [[0]]
layers = 1
is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 2
n_outputs = 2
observable = 'position'
in_norm_range = (-3, 3)
out_norm_range = (1, 5)
loss_function = cross_entropy
basinhopping_iters = 2

# === DATASET SETTINGS ===
#dataset_name = 'moons'
dataset_name = 'circles'
trainset_size = 100
validset_size = 50
num_cats = 2
dataset_size_per_cat = trainset_size // 2
validset_size_per_cat = validset_size // 2
model_name = f"{dataset_name}_{N}modes_{layers}layers_ph{ladder_modes}_in{in_norm_range}_out{out_norm_range}_datasize{trainset_size}"

if os.path.isfile(f"datasets/{dataset_name}_inputs.npy"):
    with open(f"datasets/{dataset_name}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{dataset_name}_outputs.npy", "rb") as f:
        outputs = np.load(f)
    shuffling = np.random.permutation(len(inputs))
    dataset = [inputs[shuffling], outputs[shuffling]]
    data_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
else:
    Exception

# === PREPROCESSORS AND POSTPROCESSORS ===
in_preprocessors = []
in_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=data_ranges, rescale_range=in_norm_range))
in_preprocessors.append(partial(pad_data, length=2*N))

out_preprocessors = []
output_range = (0, 1)
out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

postprocessors = []

# === BUILD TRAINING, VALIDATION AND TESTING DATASET ===
balanced_trainset = (
    np.concatenate(
        (dataset[0][:dataset_size_per_cat], dataset[0][-dataset_size_per_cat:])
    ),
    np.concatenate(
        (dataset[1][:dataset_size_per_cat], dataset[1][-dataset_size_per_cat:])
    )
)
train_dataset = (balanced_trainset[0], one_hot_encoding(balanced_trainset[1], num_cats))

balanced_validset = (
    np.concatenate(
        (dataset[0][dataset_size_per_cat:dataset_size_per_cat+validset_size_per_cat], dataset[0][-(dataset_size_per_cat+validset_size_per_cat):-dataset_size_per_cat])
    ),
    np.concatenate(
        (dataset[1][:dataset_size_per_cat], dataset[1][-dataset_size_per_cat:])
    )
)
#valid_dataset = (balanced_validset[0], one_hot_encoding(balanced_validset[1], num_cats))
valid_dataset = None

test_dataset = (
    dataset[0],
    one_hot_encoding(dataset[1], num_cats)
)
test_outputs_cats = dataset[1]
test_outputs_cats = test_outputs_cats.reshape((len(test_outputs_cats)))

# === BUILD, TRAIN AND TEST QNN ===
qnn, train_loss, valid_loss = build_and_train_model(model_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                                    include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                                    train_dataset, valid_dataset, loss_function, basinhopping_iters, in_preprocessors, out_preprocessors, postprocessors)

with open(f"losses/{model_name}.npy", "wb") as f:
    np.save(f, np.array(train_loss))
with open(f"valid_losses/{model_name}.npy", "wb") as f:
    np.save(f, np.array(valid_loss))

plt.plot(np.log(np.array(train_loss)+1), c='red', label=f'Train loss N={N}')
plt.plot(np.log(np.array(valid_loss)+1), c='red', linestyle='dashed', label=f'Validation loss N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
plt.legend()
plt.savefig(f"figures/logloss_{model_name}.pdf")
plt.show()
plt.clf()

# Test the trained QONN with the unused samples of the MNIST dataset
qnn_test_outputs = qnn.test_model(test_dataset, loss_function)
with open(f"testing/{model_name}.npy", "wb") as f:
    np.save(f, np.array(qnn_test_outputs))
qnn_test_prob_outs = softmax_discretization(qnn_test_outputs)
qnn_test_cat_outs = greatest_probability(qnn_test_prob_outs)
qnn_test_cat_outs = qnn_test_cat_outs.reshape((len(qnn_test_cat_outs)))
if is_addition:
    nongauss_op = "â†"
else:
    nongauss_op = "â"
plot_title = f'QONN of N={qnn.N}, L={qnn.layers}, {nongauss_op} in modes {np.array(qnn.ladder_modes[0]) + 1}'

def plot_qonn_decision(X, y, predict_proba, title="QONN decision boundary"):
    # 1) Scatter original data
    plt.figure(figsize=(6,6))
    plt.scatter(
        X[:,0], X[:,1],
        c=y, cmap="coolwarm",
        edgecolor="k", s=40, alpha=0.8
    )
    # 2) Create grid
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 3) Predict probabilities on grid
    probs = softmax_discretization(predict_proba(grid))[:,0]      # probability of class=1
    Z = probs.reshape(xx.shape)
    # 4) Plot soft‐background and decision contour
    plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)
    # 5) Final touches
    plt.title(title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(f"figures/{model_name}.pdf")
    
plot_qonn_decision(test_dataset[0], test_outputs_cats, qnn.evaluate_model, plot_title)
plt.show()
plt.clf()

accuracy = np.equal(qnn_test_cat_outs, test_outputs_cats).sum()
print(f"Accuracy: {accuracy}/{len(qnn_test_cat_outs)} = {accuracy/len(qnn_test_cat_outs)}")
plot_qnn_testing(qnn, test_outputs_cats, qnn_test_cat_outs)
plt.show()
plt.clf()

# Generate the confusion matrix
cm = confusion_matrix(test_outputs_cats, qnn_test_cat_outs)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_total = cm.astype('float') / cm.sum()

# Plotting the confusion matrix as a green heatmap with variable opacity
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', vmin=0.0, vmax=1.0, alpha=cm_normalized)
for t in ax.texts:
    val = float(t.get_text())
    t.set_color('white' if val > 0.7 else 'black')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.savefig(f"figures/cm_{model_name}.pdf")
plt.clf()

plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm_total, annot=True, fmt='.2f', cmap='Greens', vmin=0.0, vmax=1.0, alpha=cm_total)
for t in ax.texts:
    val = float(t.get_text())
    t.set_color('white' if val > 0.7 else 'black')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.savefig(f"figures/cmacc_{model_name}.pdf")
plt.clf()
