from functools import partial

from .qnn import test_model, build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *

# === HYPERPARAMETERS DEFINITION ===
N = 6
layers = 1
is_input_reupload = False
n_inputs = 6
n_outputs = 1
observable = 'position'
in_norm_range = (0.15, 3)
out_norm_range = (1, 10)

# === DATASET SETTINGS ===
output_range = (0, 9)
num_categories = 10
dataset_size = 200
testing_set_size = 100
model_name = "mnist_encoded"
dataset = autoencoder_mnist(n_inputs)

# === PREPROCESSORS AND POSTPROCESSORS ===
in_preprocessors = []
data_ranges = [(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))]
print(data_ranges)
in_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=data_ranges, rescale_range=in_norm_range))

out_preprocessors = []
out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

postprocessors = []
#postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))
#postprocessors.append(partial(binning, data_range=out_norm_range, num_categories=num_categories))
postprocessors.append(partial(np.round))
postprocessors.append(partial(lambda x: x-1))
#postprocessors.append(partial(np.floor))

# === BUILD, TRAIN AND TEST QNN ===
train_dataset = (dataset[0][:dataset_size], dataset[1][:dataset_size])

# Build the QNN and train it with the generated dataset
qnn, loss = build_and_train_model(model_name, N, layers, n_inputs, n_outputs, observable, is_input_reupload, 
                                  train_dataset, in_preprocessors, out_preprocessors, postprocessors)

plt.plot(np.log(np.array(loss)+1), label=f'N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
plt.legend()
plt.show()

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = (dataset[0][dataset_size : dataset_size+testing_set_size], dataset[1][dataset_size : dataset_size+testing_set_size])
qnn_test_outputs = test_model(qnn, test_dataset)
for (i,j) in zip(test_dataset[1], qnn_test_outputs):
    print(f"Expected: {i} Obtained: {j}")
accuracy = ((qnn_test_outputs - 1) == test_dataset[1]).sum()
print(f"Accuracy: {accuracy}/{len(qnn_test_outputs)} = {accuracy/len(qnn_test_outputs)}")
plot_qnn_testing(qnn, test_dataset[1], qnn_test_outputs)
