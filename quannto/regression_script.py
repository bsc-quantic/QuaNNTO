import argparse
import pandas as pd
from functools import partial

from .qnn import build_and_train_model, test_model
from .data_processors import *
from .results_utils import plot_qnn_testing
from .synth_datasets import print_dataset

parser = argparse.ArgumentParser(description="Build and train a QNN model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("modes", help="Modes (neurons per layer) of the QNN")
parser.add_argument("layers", help="Number of layers of the QNN")
parser.add_argument("inputs", help="Number of inputs of the dataset")
parser.add_argument("outputs", help="Number of inputs of the dataset")
parser.add_argument("input_reuploading", help="0 for no input reuploading, 1 otherwise")
parser.add_argument("dataset", help="Dataset to be evaluated")
args = parser.parse_args()

# Parse arguments for QNN construction
N = int(args.modes)
layers = int(args.layers)
n_in = int(args.inputs)
n_out = int(args.outputs)
is_input_reupload = True if int(args.input_reuploading)==1 else False
observable_modes = [[mode, mode] for mode in range(n_out)]
observable_types = [[1,0] for _ in range(n_out)]
model_name = (args.dataset)[(args.dataset).index("/")+1 : (args.dataset).index(".")]

# Dataset setup
dataset_df = pd.read_csv(args.dataset)
inputs_set = dataset_df.iloc[:, 0:n_in].to_numpy()
outputs_set = dataset_df.iloc[:, n_in:].to_numpy()
dataset = [inputs_set, outputs_set]
in_data_ranges = [(np.min(dataset[0][:,col]), np.max(dataset[0][:,col])) for col in range(len(dataset[0][0]))]
out_data_ranges = [(np.min(dataset[1][:,col]), np.max(dataset[1][:,col])) for col in range(len(dataset[1][0]))]

training_set_size = 100
training_set = [dataset[0][:training_set_size], dataset[1][:training_set_size]]
print_dataset(training_set[0], training_set[1])

# Data preprocessing and postprocessing
in_preprocessors = []
in_rescaling = (0.5, 3.5)
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
qnn = build_and_train_model(model_name, N, layers, n_in, n_out, observable_modes, observable_types, 
                            is_input_reupload, training_set, in_preprocessors, out_preprocessors, postprocessors)

# Test the model
testing_set_size = 100
testing_set = (dataset[0][training_set_size : training_set_size+testing_set_size], dataset[1][training_set_size : training_set_size+testing_set_size])
qnn_test_outputs = test_model(qnn, testing_set)
plot_qnn_testing(qnn, testing_set[1], qnn_test_outputs)