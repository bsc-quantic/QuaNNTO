from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .qnn import test_model, build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

# === HYPERPARAMETERS DEFINITION ===
N = 4
photon_additions = [0]
layers = 1
is_input_reupload = False
n_inputs = 1
n_outputs = 4
observable = 'position'
in_norm_range = (-1, 1)
out_norm_range = (0, 1)
loss_function = cross_entropy

# === DATASET SETTINGS ===
categories = [0, 1, 2, 3]
num_cats = len(categories)
dataset_size = 200
validset_size = 80
testset_size = 80
model_name = "mnist_encoded"
while True:
    dataset = autoencoder_mnist(n_inputs, categories)
    data_ranges = np.array([(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))])
    if np.all(data_ranges[:,-1] > 0):
        break
print("ENCODED INPUTS RANGE:")
print(data_ranges)
print(dataset)
output_range = (0, 1)
#output_range = get_range(dataset[1])
#print("LABELS RANGE:")
#print(output_range)

# === PREPROCESSORS AND POSTPROCESSORS ===
in_preprocessors = []
in_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=data_ranges, rescale_range=in_norm_range))

out_preprocessors = []
out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

postprocessors = []

# === BUILD, TRAIN AND TEST QNN ===
train_dataset = (dataset[0][:dataset_size], one_hot_encoding(dataset[1][:dataset_size], num_cats))
valid_dataset = (dataset[0][dataset_size : dataset_size+validset_size], one_hot_encoding(dataset[1][dataset_size : dataset_size+validset_size], num_cats))
test_dataset = (dataset[0][dataset_size+validset_size:], one_hot_encoding(dataset[1][dataset_size+validset_size:], num_cats))
test_outputs_cats = dataset[1][dataset_size+validset_size:]
test_outputs_cats = test_outputs_cats.reshape((len(test_outputs_cats)))
# Build the QNN and train it with the generated dataset
qnn, train_loss, valid_loss = build_and_train_model(model_name, N, layers, n_inputs, n_outputs, photon_additions, observable, is_input_reupload, 
                                                    train_dataset, valid_dataset, loss_function, in_preprocessors, out_preprocessors, postprocessors)

plt.plot(np.log(np.array(train_loss)+1), c='red', label=f'Train loss N={N}')
plt.plot(np.log(np.array(valid_loss)+1), c='red', linestyle='dashed', label=f'Validation loss N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
plt.legend()
plt.show()

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
qnn_test_outputs = test_model(qnn, test_dataset, loss_function)
qnn_test_prob_outs = softmax_discretization(qnn_test_outputs)
qnn_test_cat_outs = greatest_probability(qnn_test_prob_outs)
qnn_test_cat_outs = qnn_test_cat_outs.reshape((len(qnn_test_cat_outs)))

accuracy = np.equal(qnn_test_cat_outs, test_outputs_cats).sum()
print(f"Accuracy: {accuracy}/{len(qnn_test_cat_outs)} = {accuracy/len(qnn_test_cat_outs)}")
plot_qnn_testing(qnn, test_outputs_cats, qnn_test_cat_outs)

# Generate the confusion matrix
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
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm_total, annot=True, fmt='.2f', cmap='Greens', alpha=cm_normalized)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
