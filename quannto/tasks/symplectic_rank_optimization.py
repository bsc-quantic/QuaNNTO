import matplotlib.pyplot as plt
from matplotlib import colormaps
import os.path

from quannto.core.qnn_trainers import train_symplectic_rank
from quannto.dataset_gens.synth_datasets import *
from quannto.utils.results_utils import *
from quannto.core.data_processors import *
from quannto.core.loss_functions import *

#np.random.seed(27)

# === HYPERPARAMETERS DEFINITION ===
modes = [2]
photon_additions = [[0]]
layers = [2]
is_input_reupload = False
n_inputs = 1
n_outputs = 1
observable = 'witness'
in_norm_range = (-2, 2)
out_norm_range = (1, 5)
loss_function = exp_val
basinhopping_iters = 0
params = np.array([
 0.82730684, 0.43208538, 0.18228398, 0.2870262,  0.197006,   0.81632739,
 0.75076105, 0.2439437,  0.52731704, 0.95077704, 0.18119154, 0.99306807,
 0.99667332, 0.6984572,  0.10448931, 0.82614215, 0.01884125, 0.02328073,
 0.5700898 , 0.87741522, 0.37338494, 0.6235469,  0.10093182, 0.46114707,
 0.68896096, 0.85563143, 0.84310635, 0.93252304, 0.25616017, 0.97075845,
 0.0897363 ])
params = None

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
#colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
colors = colormaps['tab10']
train_losses = []
qnn_loss = []
qnn_outs = []
qnns = []
for N in modes:
    for l in layers:
        for ph_add in photon_additions:
            model_name = "ent_witness"
            # Build the QNN and train it with the generated dataset
            qnn, train_loss = train_symplectic_rank(model_name, N, l, n_inputs, n_outputs, ph_add, observable, init_pars=params)
            qnns.append(qnn)
            train_losses.append(train_loss.copy())
            qnn_loss.append(train_loss[-1])

c=0
for (train_loss, qnn) in zip(train_losses, qnns):
    plt.plot(np.array(train_loss), c=colors(c%10), label=f'N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer')
    c+=1
    
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Expectation value')
plt.title(f'Entanglement witness training')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/ent_wit_loss_"+model_name+"_N"+str(modes[0])+".pdf")
#plt.show()
plt.clf()

print(f'MINIMUM ACHIEVED VALUES: {qnn_loss}')