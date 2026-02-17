import numpy as np
import scipy.integrate as si

if not hasattr(si, "simps") and hasattr(si, "simpson"):
    si.simps = si.simpson
import strawberryfields as sf
from strawberryfields.ops import Ket, GaussianTransform, Dgate

from quannto.utils.strawberryfields.general_tools import apply_annihilation_to_ket

def apply_subtractions_layer(ket, cutoff, modes_list):
    """
    Apply a on each mode in modes_list sequentially (order irrelevant for different modes).
    Returns (ket_new, probs_list_for_each_click_in_this_layer).
    """
    probs = []
    for m in modes_list:
        ket, p = apply_annihilation_to_ket(ket, cutoff, m)
        probs.append(p)
    return ket, probs

def apply_gaussian_layer(ket_in, N, cutoff, S, d):
    """
    Applies GaussianTransform(S) + displacement d to ket_in using SF.
    - S: 2N×2N symplectic in ordering (x1..xN, p1..pN)
    - d_phys: length-2N displacement in the SAME ordering, in textbook (phys) units
    If phys_convention=True, we rescale displacement by sqrt(2) for SF gates.
    """
    S = np.real_if_close(np.array(S))
    d = np.real_if_close(np.array(d))

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(N)
    with prog.context as q:
        Ket(ket_in) | tuple(q[i] for i in range(N))

        # Symplectic transform
        GaussianTransform(S, tol=1e-6) | tuple(q[i] for i in range(N))

        # Displacement: d = (dx1..dxN, dp1..dpN)
        for m in range(N):
            alpha_m = d[m] + 1j * d[N + m]
            Dgate(np.abs(alpha_m), np.angle(alpha_m)) | q[m]
    return eng.run(prog).state.ket()

def run_qonn(N, cutoff, layers, subtractions, input_alpha=None):
    """
    Builds and runs an N-mode QONN model with cutoff from each layer's
    symplectic matrix, displacements and subtractions for a given input alpha
    :param N: Number of modes of the QONN model
    :param layers: list of dicts [{"S":..., "d":...}, ...] containing each layer's symplectic S and displacement d
    :param subtractions: list of lists, e.g. [[0],[1]] or [[0,1], []] denoting the subtracted modes of each layer
    :param input_alpha: optional complex coherent displacement encoded on mode 0 at input.
    Returns (ket_out, probs_per_layer).
    """
    assert len(layers) == len(subtractions), "Need one subtractions list per layer."
    # start from vacuum ket
    ket = np.zeros(cutoff**N, dtype=complex)
    ket[0] = 1.0

    # optional input encoding D(alpha) on mode 0
    if input_alpha is not None:        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        prog = sf.Program(N)
        
        r = np.abs(input_alpha)
        phi = np.angle(input_alpha)
        with prog.context as q:
            Ket(ket) | tuple(q[i] for i in range(N))
            Dgate(r, phi) | q[0]
        ket = eng.run(prog).state.ket()

    probs_per_layer = []
    for ell, (layer, subs) in enumerate(zip(layers, subtractions)):
        S, d = layer["S"], layer["d"]
        
        ket = apply_gaussian_layer(ket, N, cutoff, S, d)
        ket, probs_clicks = apply_subtractions_layer(ket, cutoff, subs)

        probs_per_layer.append(np.prod(probs_clicks))

    return ket, probs_per_layer
