import argparse
import pandas as pd

from .core.qnn import test_model, load_model
from .utils.results_utils import plot_qnns_testing
from .dataset_gens.synthetic_datasets import print_dataset

parser = argparse.ArgumentParser(description="Load and evaluate a QNN model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("file", help="QNN model file")
parser.add_argument("dataset", help="Dataset to be evaluated")
parser.add_argument("inputs", help="Number of inputs of the dataset")

args = parser.parse_args()

qnn_model = load_model(args.file)
eval_df = pd.read_csv(args.dataset)
n_in = int(args.inputs)

inputs_set = eval_df.iloc[:, 0:n_in].to_numpy()
outputs_set = eval_df.iloc[:, n_in:].to_numpy()
print_dataset(inputs_set, outputs_set)

qnn_test_outputs = test_model(qnn_model, [inputs_set, outputs_set])
plot_qnns_testing(qnn_model, outputs_set, qnn_test_outputs)
