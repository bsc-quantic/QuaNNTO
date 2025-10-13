from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .qnn import QNN
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
modes = 2
ladder_modes = [0]
layers = 1
is_addition = False
include_initial_squeezing = False
include_initial_mixing = False
n_inputs = 1
n_outputs = 1
observable = 'position'
in_norm_range = (-2, 2)
out_norm_range = (-2, 2)
loss_function = mse
# Random parameters
n_pars = layers*(2*modes**2 + 3*modes)
if include_initial_squeezing:
    n_pars += modes
if include_initial_mixing:
    n_pars += modes**2
params = np.random.rand(n_pars)
""" params = np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452,
 0.05808361, 0.86617615, 0.60111501, 0.70807258, 0.02058449, 0.96990985,
 0.83244264, 0.21233911, 0.18182497, 0.18340451, 0.30424224, 0.52475643,
 0.43194502, 0.29122914, 0.61185289, 0.13949386, 0.29214465, 0.36636184,
 0.45606998, 0.78517596, 0.19967378, 0.51423444]) """
# Squeezing
""" params = np.array([ 0.01025137,  0.47277251,  1.44013569,  1.57896821,  0.03567841,  0.19049764,
  0.2086248,   0.60184093, -2.99487624,  0.07580988,  0.01767892,  0.10691931,
  0.1692862,   0.3223461 ]) """
# No squeezing
""" params = np.array([ 0.16074294,  1.5422531,   0.57696016,  0.70204455,  0.1477138,   1.09022984,
  0.01934681, -0.16175933,  0.,          0.,          0.02865456,  0.11995181,
  1.31325591,  0.19158564]) """

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
          include_initial_squeezing, include_initial_mixing,
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
print(qnn.eval_QNN_nojax(qnn.tunable_parameters, inputs=np.array([2,0,0,0])))
input('alpha 0')
print(qnn.eval_QNN_nojax(qnn.tunable_parameters, inputs=np.array([1,1,0,0])))
input('alpha 1')
print(qnn.eval_QNN_nojax(qnn.tunable_parameters, inputs=np.array([2,2,0,0])))
input('alpha 2')

    
c = 0
plt.plot(test_dataset[0], test_dataset[1], 'o', markerfacecolor='g', markeredgecolor='none', alpha=0.25, label='Expected results')
plt.plot(test_dataset[0], qnn_test_outputs, c=colors(c+3%10), linestyle='dashed', label=f'N={qnn.N}, L={qnn.layers}, {len(qnn.ladder_modes)} photons/layer')
plt.title(f'TESTING SET')
plt.xlabel('Input')
plt.xlim()
plt.ylabel('Output')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/testing_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(ladder_modes)+"_in"+str(input_range)+".png")
plt.show()
plt.clf()