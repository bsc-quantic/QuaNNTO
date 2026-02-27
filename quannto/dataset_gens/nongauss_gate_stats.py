import numpy as np

from quannto.utils.path_utils import datasets_dir
from quannto.utils.strawberryfields.general_tools import cubicphase_moments

# Dataset settings for cubic-phase gate synthesis
gamma      = 0.2
cutoff      = 200
dataset_size = 50
# For real alphas:
alpha_list = np.linspace(-2, 2, dataset_size)
# For complex alphas:
""" alpha_re_list = np.linspace(-2, 2, dataset_size)
alpha_im_list = np.linspace(-2, 2, dataset_size)
alpha_list = alpha_re_list + 1j*alpha_im_list """

dataset = []
inputs_dataset = []
outputs_dataset = []
for α in alpha_list:
    obs = cubicphase_moments(α, gamma, cutoff)
    y = np.array([obs[k] for k in ("a","a2","a3","a†n", "n", "n2")])
    dataset.append(([α], y))
    inputs_dataset.append([α])
    outputs_dataset.append(y)

dataset_dir = str(datasets_dir() / f'fock_cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}')

with open(f"{dataset_dir}_inputs.npy", "wb") as f:
    np.save(f, np.array(inputs_dataset))
with open(f"{dataset_dir}_outputs.npy", "wb") as f:
    np.save(f, np.array(outputs_dataset))
