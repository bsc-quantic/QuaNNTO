import matplotlib.pyplot as plt
from matplotlib import colormaps
import os.path

from .qnn_trainers import train_entanglement_witness
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

#np.random.seed(27)

# === HYPERPARAMETERS DEFINITION ===
modes = [2]
photon_additions = [[0]]
layers = [1]
is_input_reupload = False
n_inputs = 1
n_outputs = 1
observable = 'witness'
in_norm_range = (-2, 2)
out_norm_range = (1, 5)
loss_function = exp_val
basinhopping_iters = 0 

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
            qnn, train_loss = train_entanglement_witness(model_name, N, l, n_inputs, n_outputs, ph_add, observable)
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
plt.savefig("figures/ent_wit_loss_"+model_name+"_N"+str(modes[0])+".png")
plt.show()

print(f'MINIMUM ACHIEVED VALUES: {qnn_loss}')