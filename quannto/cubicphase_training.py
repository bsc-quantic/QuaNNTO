from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .qnn_trainers import build_and_train_model
from .synth_datasets import *
from .results_utils import *
from .data_processors import *
from .loss_functions import *

np.random.seed(42)

# === HYPERPARAMETERS DEFINITION ===
modes = [2]*3
#photon_additions = [[],[0],[0,0],[0,1],[0]]
photon_additions = [[],[0],[0,1]]
layers = [1]*3
is_input_reupload = False
n_inputs = 1
n_outputs = 1
observable = 'third-order'
in_norm_ranges = [(-2, 2)]*len(modes)
out_norm_ranges = [(-2, 2)]*len(modes)
loss_function = mse
basinhopping_iters = 0
params = None

# === DATASET SETTINGS ===
gamma = 0.2
dataset_size = 50
input_range = (-2, 2)
alpha_list = np.linspace(input_range[0], input_range[1], dataset_size)
model_name = f'cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}'

# Load training dataset of the cubic phase gate to be learned
with open(f"datasets/cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}_inputs.npy", "rb") as f:
    inputs = np.load(f)
with open(f"datasets/cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}_outputs.npy", "rb") as f:
    outputs = np.load(f)
train_dataset = [inputs, outputs]

colors = colormaps['tab10']
expvals = ['⟨x⟩', '⟨p⟩', '⟨x²⟩', '⟨p²⟩', '⟨x³⟩' , '⟨p³⟩']
c=0
for exp_val_idx in range(4, outputs.shape[1]):
    plt.plot(inputs, outputs[:,exp_val_idx], c=colors(c), label=expvals[exp_val_idx])
    c += 1
plt.title(f'Dataset')
plt.xlabel('x')
plt.xlim()
plt.ylabel('Statistical moments')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/trainset_"+model_name+".png")
plt.show()
plt.clf()

# === BUILD, TRAIN AND TEST QNN MODELS WITH DIFFERENT MODES ===
#colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
colors = colormaps['tab10']
train_losses = []
valid_losses = []
qnn_loss = []
qnn_outs = []
qnns = []
for (N, l, ph_add, in_norm_range, out_norm_range) in zip(modes, layers, photon_additions, in_norm_ranges, out_norm_ranges):
    # Initialize the desired data processors for pre/post-processing
    in_preprocessors = []
    in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
    in_preprocessors.append(partial(pad_data, length=2*N))

    out_preprocessors = []
    #out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))

    postprocessors = []
    #postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))

    model_name = model_name + "_N" + str(N) + "_L" + str(l) + "_ph" + str(ph_add)# + "_in" + str(in_norm_range) + "_out" + str(out_norm_range)
    # Build the QNN and train it with the generated dataset
    qnn, train_loss, valid_loss = build_and_train_model(model_name, N, l, n_inputs, n_outputs, ph_add, observable, is_input_reupload,
                                                        train_dataset, None, loss_function, basinhopping_iters,
                                                        in_preprocessors, out_preprocessors, postprocessors, init_pars=params)
    qnns.append(qnn)
    
    train_losses.append(train_loss.copy())
    with open(f"losses/{model_name}.npy", "wb") as f:
        np.save(f, np.array(train_loss))
    qnn_loss.append(train_loss[-1])
    
    qnn_test_outputs = qnn.test_model(train_dataset, loss_function)
    with open(f"testing/{model_name}.npy", "wb") as f:
        np.save(f, np.array(qnn_test_outputs))
    qnn_outs.append(qnn_test_outputs.copy())
    
# PLOT RESULTS
#legend_labels = [f'N={qnn.N}, L={qnn.layers}, Photon addition in modes {np.array(qnn.photon_add) + 1}, disp range {in_norm_range}' for (qnn, in_norm_range) in zip(qnns, in_norm_ranges)]
legend_labels = [f'N={qnn.N}, L={qnn.layers}, a† in modes {np.array(qnn.photon_add) + 1}' for qnn in qnns]
#legend_labels = [f'N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer' for qnn in qnns]
#legend_labels = [f'N={qnn.N}, L={qnn.layers}, {len(qnn.photon_add)} photons/layer, α ∈ {in_norm_range}' for (qnn, in_norm_range) in zip(qnns, in_norm_ranges)]

c=0
for exp_val_idx in range(4, outputs.shape[1]):
    plt.plot(inputs, outputs[:,exp_val_idx], c=colors(c), label=expvals[exp_val_idx])
    c += 1
    
colors = colormaps['tab10']
markers = ['*', 's', 'v', '+', 'x', '<', '>', 'd', 'p', 'h', '^']
m = 0
for (qnn_test_outputs, legend_label) in zip(qnn_outs, legend_labels):
    c = 0
    plt.plot([], [],
             marker=markers[m],
             linestyle='None',
             color='black',
             label=legend_label)
    for exp_val_idx in range(4, outputs.shape[1]):
        plt.plot(inputs, qnn_test_outputs[:,exp_val_idx], marker=markers[m], c=colors(c), alpha=0.3, linestyle='dashed')
        c += 1
    m+=1
plt.title(f'QONN EVALUATION')
plt.xlabel('Displacement α')
plt.xlim()
plt.ylabel('Statistical moments')
#plt.ylim(top=output_range[1] + len(qnns)*0.2 + 0.2)
plt.grid(linestyle='--', linewidth=0.4)
#plt.legend(loc='upper right')
plt.legend()
plt.savefig("figures/result_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(photon_additions)+"_in"+str(input_range)+".png")
plt.show()
plt.clf()

c=0
for (train_loss, legend_label) in zip(train_losses, legend_labels):
    x_log = np.log(np.array(range(1,len(train_loss)+1)))
    plt.plot(np.log(np.array(train_loss)+1), c=colors(c+3%10), label=f'Train loss {legend_label}')#, {input_range} disp')
    c+=1

plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title(f'LOGARITHMIC TRAINING AND VALIDATION LOSS')
plt.grid(linestyle='--', linewidth=0.4)
plt.legend()
plt.savefig("figures/logloss_"+model_name+"_N"+str(modes)+"_L"+str(layers)+"_ph"+str(photon_additions)+"_in"+str(input_range)+".png")
plt.show()
plt.clf()

print(f'MINIMAL LOSSES ACHIEVED: {qnn_loss}')