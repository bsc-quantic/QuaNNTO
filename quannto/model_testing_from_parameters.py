from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .core.qnn import QNN
from .dataset_gens.synthetic_datasets import *
from .utils.results_utils import *
from .core.data_processors import *
from .core.loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
modes = 2
ladder_modes = [[0,0]]
layers = 1
is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
is_passive_gaussian = False
n_inputs = 1
n_outputs = 1
observable = 'position'
in_norm_range = (-5, 5)
out_norm_range = (-2, 2)
loss_function = mse
# Random parameters
n_pars = layers*(2*modes**2 + 3*modes)
if include_initial_squeezing:
    n_pars += modes
if include_initial_mixing:
    n_pars += modes**2
params = np.random.rand(n_pars)

# === TARGET FUNCTION SETTINGS ===
target_function = sin_1in_1out
testset_size = 200
input_range = (0, 6.3)
real_function = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size*10, input_range)
output_range = get_range(real_function[1])

# Generate a linearly-spaced testing dataset of the target function and test the trained QNN
test_dataset = generate_linear_dataset_of(target_function, n_inputs, n_outputs, testset_size, input_range)

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
#colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
colors = colormaps['tab10']

# Initialize the desired data processors for pre/post-processing
in_preprocessors = []
in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
in_preprocessors.append(partial(pad_data, length=2*modes))

out_preprocessors = []
out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

postprocessors = []
postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

model_name = "model_N" + str(modes) + "_L" + str(layers) + "_" + target_function.__name__

# Build the QNN with the trained parameters
qnn = QNN(model_name, modes, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
          include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
          in_preprocessors, out_preprocessors, postprocessors)
qnn.tunable_parameters = jnp.array(params)

# Evaluate the model with the testing set
qnn_test_outputs = qnn.test_model(test_dataset, loss_function)
print(">>> TEST DONE!")
qnn.build_QNN(params)
print("=== QNN SUMMARY ===")
print(np.round(qnn.D_l,4))
print(np.round(qnn.D_concat, 4))
print(qnn.S_l)
print(qnn.S_concat)
qnn.print_qnn()
    
c = 0
plt.plot(test_dataset[0], test_dataset[1], 'o', markerfacecolor='g', markeredgecolor='none', alpha=0.25, label='Expected results')
plt.plot(test_dataset[0], qnn_test_outputs, c=colors(c+3%10), linestyle='dashed', label=f'N={qnn.N}, L={qnn.layers}, {len(qnn.ladder_modes)} photons/layer')
plt.title(f'TESTING SET')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/testing_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(ladder_modes)+"_in"+str(input_range)+".pdf")
plt.show()
plt.clf()