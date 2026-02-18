import numpy as np
import scipy.integrate as si

from quannto.utils.path_utils import datasets_dir
from quannto.utils.strawberryfields.general_tools import catstate_moments, moments_from_rho, six_order_moments_operators
if not hasattr(si, "simps") and hasattr(si, "simpson"):
    si.simps = si.simpson

def build_catstates_dataset(alpha_list, moments, cutoff=20, phi=0.0):
    """
    For each alpha in alpha_list, prepare the even cat state and compute:
    [<a>, <a^2>, <a^3>, <a^\dagger n>, <n>]
    Returns: list of tuples (alpha, np.array([...])).
    """
    inputs_dataset = []
    outputs_dataset = []
    for alpha in alpha_list:
        moments_expvals = catstate_moments(alpha, moments, cutoff, phi)
        inputs_dataset.append([alpha])
        outputs_dataset.append(moments_expvals)
    return inputs_dataset, outputs_dataset

# Grid of real alphas in some range
dataset_size = 1
alpha_range = (-1, 1)
cutoff = 20
phi = 0.0
alphas = np.linspace(alpha_range[0], alpha_range[1], dataset_size)
stat_moments = six_order_moments_operators(cutoff)
num_moments = len(stat_moments)

#inputs_dataset, outputs_dataset = build_catstates_dataset(alphas, stat_moments, cutoff=cutoff, phi=phi)  # even cat
inputs_dataset = [[alpha] for alpha in alphas]
outputs_dataset = [catstate_moments(alpha, stat_moments, cutoff, phi) for alpha in alphas]
ds = [inputs_dataset, outputs_dataset]
dataset_dir = str(datasets_dir() / f'catstate_phi{phi}_trainsize{dataset_size}_stats{num_moments}_cut{cutoff}_rng{alpha_range[0]}to{alpha_range[-1]}')

with open(f"{dataset_dir}_inputs.npy", "wb") as f:
    np.save(f, np.array(inputs_dataset))

with open(f"{dataset_dir}_outputs.npy", "wb") as f:
    np.save(f, np.array(outputs_dataset))

# Quick sanity checks for real α, even cat (phi=0):
# <a> ≈ 0, <a^3> ≈ 0; <a^2> ≈ α^2 (edge effects if cutoff too small or |α| large)
for alpha, vec in zip(inputs_dataset, outputs_dataset):  # print a few
    print("Alpha: ", alpha)
    print("Moments: ", vec)