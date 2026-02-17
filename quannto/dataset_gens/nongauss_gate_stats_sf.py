import scipy.integrate as si
if not hasattr(si, "simps") and hasattr(si, "simpson"):
    si.simps = si.simpson
import strawberryfields as sf
from strawberryfields.ops import Coherent, Vgate
import numpy as np
    
def compute_cubic_expectations_fockspace(alpha, gamma, n_max):
    """
    Returns a dict of expectation values for {a, a†, , p^2, x^3, p^3}
    after applying Vgate(gamma) to |alpha>, using cutoff n_max.
    """
    # 1) Run the SF program
    eng = sf.Engine("fock", backend_options={"cutoff_dim": n_max})
    prog = sf.Program(1)
    
    # 2) Generate the coherent state and apply cubic phase gate V
    with prog.context as q:
        Coherent(alpha)    | q[0]
        Vgate(gamma)  | q[0]
    state = eng.run(prog).state

    rho = state.reduced_dm(0)  # shape = (n_max, n_max)

    # 3) Compute statistical moments of the state
    a = np.zeros((n_max, n_max), dtype=complex)
    for n in range(n_max-1):
        a[n, n+1] = np.sqrt(n+1)
    adag = a.conj().T
    
    a_mean = np.trace(rho @ a)
    
    a2 = np.trace(rho @ (a @ a))
    n = np.trace(rho @ (adag @ a))

    a3 = np.trace(rho @ (a @ a @ a))
    adag_n = np.trace(rho @ (adag @ adag @ a))
    
    n2 = np.trace(rho @ (adag @ a @ adag @ a))

    return {
        "a":  a_mean,
        "a2": a2,
        "a3": a3,
        "a†n": adag_n,
        "n": n,
        "n2": n2
    }

# Dataset settings for cubic-phase gate synthesis
gamma      = 0.5
n_max      = 200
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
    obs = compute_cubic_expectations_fockspace(α, gamma, n_max)
    y = np.array([obs[k] for k in ("a","a2","a3","a†n", "n", "n2")])
    dataset.append(([α], y))
    inputs_dataset.append([α])
    outputs_dataset.append(y)
    
with open(f"datasets/fock_cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}_inputs.npy", "wb") as f:
    np.save(f, np.array(inputs_dataset))
with open(f"datasets/fock_cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}_outputs.npy", "wb") as f:
    np.save(f, np.array(outputs_dataset))
