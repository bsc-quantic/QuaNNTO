import argparse
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

from quannto.core.qnn import build_and_train_model, test_model
from quannto.core.data_processors import *
from quannto.utils.results_utils import plot_qnn_testing
from quannto.dataset_gens.synth_datasets import print_dataset, bubblesort
from quannto.core.loss_functions import *

parser = argparse.ArgumentParser(description="Build and train a QNN model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("modes", help="Modes (neurons per layer) of the QNN")
parser.add_argument("layers", help="Number of layers of the QNN")
parser.add_argument("inputs", help="Number of inputs of the dataset")
parser.add_argument("outputs", help="Number of inputs of the dataset")
parser.add_argument("input_reuploading", help="0 for no input reuploading, 1 otherwise")
parser.add_argument("obs", help="Observable operator for the outputs: 'position', 'momentum' or 'number'.")
parser.add_argument("loss", help="Loss function: 'mse' or 'nll'.")
parser.add_argument("dataset", help="Dataset to be evaluated")
args = parser.parse_args()

# Parse arguments for QNN construction
N = int(args.modes)
layers = int(args.layers)
n_in = int(args.inputs)
n_out = int(args.outputs)
is_input_reupload = True if int(args.input_reuploading)==1 else False
observable = str(args.obs)
assert (observable == 'position' or observable == 'momentum' or observable == 'number')
loss = str(args.loss)
assert (loss == 'mse' or loss == 'nll')
model_name = (args.dataset)[(args.dataset).index("/")+1 : (args.dataset).index(".")]

# Non-Gaussianity: by default, photon addition on mode 0
photon_additions = [0]

# Dataset setup
dataset_df = pd.read_csv(args.dataset)
inputs_set = dataset_df.iloc[:, 0:n_in].to_numpy()
outputs_set = dataset_df.iloc[:, n_in:].to_numpy()
dataset = [inputs_set, outputs_set]
in_data_ranges = [(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))]
out_data_ranges = [(np.min(dataset[1][:,col]), np.max(dataset[1][:,col])) for col in range(len(dataset[1][0]))]
loss_function = retrieve_loss_function(loss)

trainset_size = 200
train_dataset = [dataset[0][:trainset_size], dataset[1][:trainset_size]]
print_dataset(train_dataset[0], train_dataset[1])
validset_size = 40
valid_dataset = [dataset[0][trainset_size : trainset_size+validset_size], dataset[1][trainset_size : trainset_size+validset_size]]
print_dataset(valid_dataset[0], valid_dataset[1])

# Data preprocessing and postprocessing
in_preprocessors = []
in_rescaling = (0, 1)
print(in_data_ranges)
in_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=in_data_ranges, rescale_range=in_rescaling))

out_preprocessors = []
out_rescaling = (1, 5)
print(out_data_ranges)
out_preprocessors.append(partial(rescale_data, data_range=out_data_ranges[0], scale_data_range=out_rescaling))
#out_preprocessors.append(partial(rescale_set_with_ranges, data_ranges=out_data_ranges, rescale_range=in_rescaling))

postprocessors = []
postprocessors.append(partial(rescale_data, data_range=out_rescaling, scale_data_range=out_data_ranges[0]))

# Create and train a QNN model
qnn, train_loss, valid_loss = build_and_train_model(model_name, N, layers, n_in, n_out, photon_additions, observable, is_input_reupload,
                                                    train_dataset, valid_dataset, loss_function, in_preprocessors, out_preprocessors, postprocessors)

# Plot the training and validation loss values
plt.plot(np.log(np.array(train_loss)+1), c='red', label=f'Train loss N={N}')
plt.plot(np.log(np.array(valid_loss)+1), c='red', linestyle='dashed', label=f'Validation loss N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
plt.legend()
plt.show()

# Test the model
testing_set_size = 100
testing_set = bubblesort(dataset[0][trainset_size+validset_size : trainset_size+validset_size+testing_set_size], dataset[1][trainset_size+validset_size : trainset_size+validset_size+testing_set_size])
qnn_test_outputs = test_model(qnn, testing_set, loss_function)
plot_qnn_testing(qnn, testing_set[1], qnn_test_outputs)