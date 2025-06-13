from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .qnn import QNN
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

# === HYPERPARAMETERS DEFINITION ===
modes = 2
photon_additions = [0]
layers = 1
is_input_reupload = False
n_inputs = 1
n_outputs = 1
observable = 'position'
in_norm_range = (-2, 2)
out_norm_range = (-2, 2)
loss_function = mse
params = None
params = np.array([ 1.48748980e+00,  1.18093233e+00,  1.06783413e+00,  1.71574815e+00,
 -2.69817525e-02,  1.21974349e+00,  1.72166995e-04, -2.65560111e+00,
 -2.30258509e+00,  2.77262870e-01, -1.16955051e-02,  4.12015470e-02,
  2.54765364e-01,  2.43054054e-01])

# === TARGET FUNCTION SETTINGS ===
target_function = sin_1in_1out
testset_size = 400
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

model_name = target_function.__name__

# Build the QNN with the trained parameters
qnn = QNN("model_N" + str(modes) + "_L" + str(layers) + "_" + model_name, modes, layers, n_inputs, n_outputs, 
          photon_additions, observable, is_input_reupload, in_preprocessors, out_preprocessors, postprocessors)
qnn.tunable_parameters = jnp.array(params)

# Evaluate the model with the testing set
qnn_test_outputs = qnn.test_model(test_dataset, loss_function)
    
c = 0
plt.plot(test_dataset[0], test_dataset[1], 'o', markerfacecolor='g', markeredgecolor='none', alpha=0.25, label='Expected results')
plt.plot(test_dataset[0], qnn_test_outputs, c=colors(c+3%10), linestyle='dashed', label=f'N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer')
plt.title(f'TESTING SET')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/testing_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(photon_additions)+"_in"+str(input_range)+".png")
#plt.show()
plt.clf()