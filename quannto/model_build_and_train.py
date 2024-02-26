from qnn import test_model, build_and_train_model
from results_utils import plot_qnn_testing
from preprocessing import *
from synth_datasets import bubblesort, print_dataset
import argparse
import pandas as pd
import numpy as np
from functools import partial

parser = argparse.ArgumentParser(description="Build and train a QNN model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("modes", help="Modes (neurons per layer) of the QNN")
parser.add_argument("layers", help="Number of layers of the QNN")
parser.add_argument("inputs", help="Number of inputs of the dataset")
parser.add_argument("outputs", help="Number of inputs of the dataset")
parser.add_argument("input_reuploading", help="0 for no input reuploading, 1 otherwise")
parser.add_argument("dataset", help="Dataset to be evaluated")
args = parser.parse_args()

N = int(args.modes)
layers = int(args.layers)
n_in = int(args.inputs)
n_out = int(args.outputs)
is_input_reupload = True if int(args.input_reuploading)==1 else False
observable_modes = [[mode, mode] for mode in range(n_out)]
observable_types = [[1,0] for _ in range(n_out)]

model_name = (args.dataset)[:(args.dataset).index(".")]

dataset_df = pd.read_csv(args.dataset)[:150]
inputs_set = dataset_df.iloc[:, 0:n_in].to_numpy()
outputs_set = dataset_df.iloc[:, n_in:].to_numpy()
print_dataset(inputs_set, outputs_set)

in_rescaling = (0.5, 3.5)
out_rescaling = (1, 5)
rescaled_inputs = rescale_set(inputs_set, in_rescaling)
rescaled_outputs = rescale_set(outputs_set, out_rescaling)
print_dataset(rescaled_inputs, rescaled_outputs)

qnn = build_and_train_model(model_name, N, layers, n_in, n_out, observable_modes, observable_types, 
                            is_input_reupload, [rescaled_inputs, rescaled_outputs])

