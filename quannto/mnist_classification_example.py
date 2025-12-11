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
modes = [5,5,5,5]
qnns_ladder_modes = [[[]], [[0]], [[0,1]], [[0],[1]]]
layers = [1, 1]
is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 3
n_outputs = 5
observable = 'position'
in_norm_ranges = [(-3, 3)]*len(modes)
out_norm_ranges = [(1, 50)]*len(modes)
loss_function = cross_entropy
basinhopping_iters = 0

# === DATASET SETTINGS ===
categories = [0, 1, 2, 3, 4]
num_cats = len(categories)
dataset_size = 75*num_cats
validset_size = 20*num_cats
continuize_method = 'encoding' # 'pca' or 'encoding' for Autoencoder
dataset_name = f'mnist_{continuize_method}_{n_inputs}lat_{num_cats}cats'
        
if os.path.isfile(f"datasets/{dataset_name}_inputs.npy"):
    with open(f"datasets/{dataset_name}_inputs.npy", "rb") as f:
        inputs = np.load(f)
    with open(f"datasets/{dataset_name}_outputs.npy", "rb") as f:
        outputs = np.load(f)
    dataset = [inputs, outputs]
    data_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
else:
    while True:
        dataset = pca_mnist(n_inputs, categories) if continuize_method == 'pca' else autoencoder_mnist(n_inputs, categories)
        data_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
        if np.all(data_ranges[:,-1] > 0):
            break
    with open(f"datasets/{dataset_name}_inputs.npy", "wb") as f:
        np.save(f, dataset[0])
    with open(f"datasets/{dataset_name}_outputs.npy", "wb") as f:
        np.save(f, dataset[1])
        
shuffling = np.random.permutation(len(dataset[0]))
dataset = [dataset[0][shuffling], dataset[1][shuffling]]
    
print("ENCODED INPUTS RANGE:")
print(data_ranges)
print(dataset)
output_range = (0, 1)
colors = ['red', 'fuchsia', 'blue', 'purple', 'orange']
for (cat, color) in zip(categories, colors):
    subset = filter_dataset_categories(dataset[0], dataset[1], [cat])
    plt.plot(subset[0][0,:], subset[0][1,:], c=color, linestyle='dotted', label=cat)
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
#plt.show()
plt.savefig(f"figures/trainset_{dataset_name}.pdf")
plt.clf()

qnns = []
qnns_preds = []
qnns_accuracies = []
for (N, l, ladder_modes, in_norm_range, out_norm_range) in zip(modes, layers, qnns_ladder_modes, in_norm_ranges, out_norm_ranges):
    model_name = f"mnist_{N}modes_{l}layers_n{ladder_modes}_{n_inputs}lat_{num_cats}cats_{observable}"
    # === PREPROCESSORS AND POSTPROCESSORS ===
    in_preprocessors = []
    in_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=data_ranges, rescale_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))

    out_preprocessors = []
    out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

    postprocessors = []

    # === BUILD, TRAIN AND TEST QNN ===
    train_dataset = (dataset[0][:dataset_size], one_hot_encoding(dataset[1][:dataset_size], num_cats))
    #valid_dataset = (dataset[0][dataset_size : dataset_size+validset_size], one_hot_encoding(dataset[1][dataset_size : dataset_size+validset_size], num_cats))
    valid_dataset = None
    #test_dataset = (dataset[0][dataset_size+validset_size:], one_hot_encoding(dataset[1][dataset_size+validset_size:], num_cats))
    test_dataset = (dataset[0], one_hot_encoding(dataset[1], num_cats))
    #test_outputs_cats = dataset[1][dataset_size+validset_size:]
    test_outputs_cats = dataset[1].reshape((len(dataset[1])))
    #test_outputs_cats = test_outputs_cats.reshape((len(test_outputs_cats)))
    # Build the QNN and train it with the generated dataset
    qnn, train_loss, valid_loss = build_and_train_model(model_name, N, l, n_inputs, n_outputs, ladder_modes, is_addition, observable, 
                                                        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                                        train_dataset, valid_dataset, loss_function, basinhopping_iters, in_preprocessors, out_preprocessors, postprocessors)
    qnns.append(qnn)

    with open(f"losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(train_loss))
    with open(f"valid_losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(valid_loss))

    """ plt.plot(np.log(np.array(train_loss)+1), c='red', label=f'Train loss N={N}')
    plt.plot(np.log(np.array(valid_loss)+1), c='red', linestyle='dashed', label=f'Validation loss N={N}')
    plt.ylim(bottom=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('Logarithmic loss value')
    plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
    plt.legend()
    plt.savefig(f"figures/logloss_{model_name}.pdf")
    #plt.show()
    plt.clf() """

    # Test the trained QONN with the unused samples of the MNIST dataset
    qnn_test_outputs, loss_value = qnn.test_model(test_dataset[0], test_dataset[1], loss_function)
    with open(f"testing/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_test_outputs))
    qnn_test_prob_outs = softmax_discretization(qnn_test_outputs)
    qnn_test_cat_outs = greatest_probability(qnn_test_prob_outs)
    qnn_test_cat_outs = qnn_test_cat_outs.reshape((len(qnn_test_cat_outs)))
    qnns_preds.append(qnn_test_cat_outs.copy())

    accuracy = np.equal(qnn_test_cat_outs, test_outputs_cats).sum()
    qnns_accuracies.append(accuracy)
    print(f"Accuracy: {accuracy}/{len(qnn_test_cat_outs)} = {accuracy/len(qnn_test_cat_outs)}")
    #plot_qnn_testing(qnn, test_outputs_cats, qnn_test_cat_outs)

    """ # Generate the confusion matrix
    cm = confusion_matrix(test_outputs_cats, qnn_test_cat_outs)
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_total = cm.astype('float') / cm.sum()
    # Plotting the confusion matrix as a green heatmap with variable opacity
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', alpha=cm_normalized)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    #plt.show()
    plt.savefig(f"figures/cm_{model_name}.pdf")
    plt.clf()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_total, annot=True, fmt='.2f', cmap='Greens', alpha=cm_normalized)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    #plt.show()
    plt.savefig(f"figures/cmacc_{model_name}.pdf")
    plt.clf() """

# PLOT RESULTS
if is_addition:
    nongauss_op = "â†"
else:
    nongauss_op = "â"
legend_labels = [f'N={qnn.N}, L={qnn.layers}, {nongauss_op} in modes {np.array(qnn.ladder_modes[0]) + 1}' for qnn in qnns] # α ∈ {in_norm_range}
avg_accuracies = np.array(qnns_accuracies)/len(qnn_test_cat_outs)

fig, ax, acc = plot_per_class_accuracy_hist(categories, test_outputs_cats, qnns_preds, model_names=legend_labels, title="MNIST — QONNs per-class accuracy")
plt.savefig(f"figures/hist_mnist_{modes}_{layers}_{qnns_ladder_modes}.pdf")
#plt.show()
plt.clf()
fig, ax, acc = plot_per_class_accuracy_markers(categories, test_outputs_cats, qnns_preds, avg_accuracies, model_names=legend_labels, title="MNIST — QONNs per-class accuracy")
plt.savefig(f"figures/acc_mnist_{modes}_{layers}_{qnns_ladder_modes}.pdf")
plt.show()
plt.clf()

print(avg_accuracies)

