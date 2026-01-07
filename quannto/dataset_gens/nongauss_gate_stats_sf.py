import strawberryfields as sf
from strawberryfields.ops import Coherent, Vgate
import numpy as np

def compute_cubic_expectations_phasespace(alpha, gamma, n_max):
    """
    Returns a dict of expectation values for {x, p, x^2, p^2, x^3, p^3}
    after applying Vgate(gamma) to |alpha>, using cutoff n_max.
    """
    # 1) Run the SF program
    eng = sf.Engine("fock", backend_options={"cutoff_dim": n_max})
    prog = sf.Program(1)
    with prog.context as q:
        Coherent(alpha)    | q[0]
        Vgate(gamma)  | q[0]
    state = eng.run(prog).state

    # 2) First and second moments via quad_expectation
    x_mean, var_x = state.quad_expectation(0, phi=0)        # ⟨x⟩, Var(x)
    p_mean, var_p = state.quad_expectation(0, phi=np.pi/2)  # ⟨p⟩, Var(p)
    x2 = var_x + x_mean**2
    p2 = var_p + p_mean**2

    # 3) Get the reduced density matrix ρ for mode 0
    rho = state.reduced_dm(0)  # shape = (n_max, n_max)

    # 4) Build the x and p operators in the Fock basis
    #    a_{n,n+1} = sqrt(n+1), so
    a = np.zeros((n_max, n_max), dtype=complex)
    for n in range(n_max-1):
        a[n, n+1] = np.sqrt(n+1)
    adag = a.conj().T

    x_op = (a + adag) / np.sqrt(2)
    p_op = 1j*(adag - a) / np.sqrt(2)
    
    x_mean = np.trace(rho @ x_op)
    p_mean = np.trace(rho @ p_op)
    
    x2 = np.trace(rho @ (x_op @ x_op))
    p2 = np.trace(rho @ (p_op @ p_op))
    xp = np.trace(rho @ (x_op @ p_op))

    # 5) Compute third moments via Tr[ρ x^3] and Tr[ρ p^3]
    x3 = np.trace(rho @ (x_op @ x_op @ x_op))
    p3 = np.trace(rho @ (p_op @ p_op @ p_op))
    xp2 = np.trace(rho @ (x_op @ p_op @ p_op))
    x2p = np.trace(rho @ (x_op @ x_op @ p_op))

    return {
        "x":  x_mean,
        "p":  p_mean,
        "x2": x2,
        "p2": p2,
        "xp": xp,
        "x3": x3,
        "p3": p3,
        "xp2": xp2,
        "x2p": x2p,
    }
    
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

    # 3) Get the reduced density matrix ρ for mode 0
    rho = state.reduced_dm(0)  # shape = (n_max, n_max)

    # 4) Build the x and p operators in the Fock basis
    #    a_{n,n+1} = sqrt(n+1), so
    a = np.zeros((n_max, n_max), dtype=complex)
    for n in range(n_max-1):
        a[n, n+1] = np.sqrt(n+1)
    adag = a.conj().T
    
    a_mean = np.trace(rho @ a)
    adag_mean = np.trace(rho @ adag)
    
    a2 = np.trace(rho @ (a @ a))
    n = np.trace(rho @ (adag @ a))
    adag2 = np.trace(rho @ (adag @ adag))

    # 5) Compute third moments via Tr[ρ x^3] and Tr[ρ p^3]
    a3 = np.trace(rho @ (a @ a @ a))
    n_a = np.trace(rho @ (adag @ a @ a))
    adag_n = np.trace(rho @ (adag @ adag @ a))
    adag3 = np.trace(rho @ (adag @ adag @ adag))
    
    n2 = np.trace(rho @ (adag @ a @ adag @ a))

    return {
        "a":  a_mean,
        #"a†":  adag_mean,
        "a2": a2,
        #"a†2": adag2,
        "a3": a3,
        #"a†3": adag3,
        #"na": n_a,
        "a†n": adag_n,
        "n": n,
        "n2": n2
    }

# Example usage: build dataset
gamma      = 0.2
n_max      = 200
dataset_size = 50
alpha_list = np.linspace(-2, 2, dataset_size)
#alpha_re_list = np.linspace(-2, 2, dataset_size)
#alpha_im_list = np.linspace(-2, 2, dataset_size)
#alpha_list = alpha_re_list + 1j*alpha_im_list

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
    
""" for α in alpha_list:
    obs = compute_cubic_expectations_phasespace(α, gamma, n_max)
    y = np.array([obs[k] for k in ("x","p","x2","p2","xp","x3","p3","xp2","x2p")])
    dataset.append(([α], y))
    inputs_dataset.append([α])
    outputs_dataset.append(y)

with open(f"datasets/phase_cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}_inputs.npy", "wb") as f:
    np.save(f, np.array(inputs_dataset))
with open(f"datasets/phase_cubicphase_gamma{gamma}_trainsize{dataset_size}_rng{alpha_list[0]}to{alpha_list[-1]}_outputs.npy", "wb") as f:
    np.save(f, np.array(outputs_dataset)) """
# 'dataset' now holds (alpha, [⟨x⟩,⟨p⟩,⟨x²⟩,⟨p²⟩,⟨xp⟩,⟨x³⟩,⟨p³⟩,⟨xp²⟩,⟨x²p⟩]) for the cubic‐phase gate.
