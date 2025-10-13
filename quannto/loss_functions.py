from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from .data_processors import softmax_discretization

def mse(expected: jnp.ndarray,
                      obtained: jnp.ndarray) -> jnp.ndarray:
    """
    expected, obtained: shape (B, D) or (B,) — real or complex
    Returns a scalar jnp.ndarray (dtype float64/float32).
    """
    # per-sample squared error (sum over features), then mean over batch
    per_sample_sse = jnp.sum(jnp.abs(expected - obtained) ** 2, axis=-1)
    mse = jnp.mean(per_sample_sse)

    # total loss; ensure real scalar even if inputs are complex
    return mse

def mse_energy_penalty(expected: jnp.ndarray,
                      obtained: jnp.ndarray,
                      penalty_lambda: float = 0.2) -> jnp.ndarray:
    """
    expected, obtained: shape (B, D) or (B,) — real or complex
    Returns a scalar jnp.ndarray (dtype float64/float32).
    """
    # per-sample squared error (sum over features), then mean over batch
    per_sample_sse = jnp.sum(jnp.abs(expected - obtained) ** 2, axis=-1)
    mse = jnp.mean(per_sample_sse)

    penalty = penalty_lambda * jnp.abs(jnp.sum(obtained[:, 8]))

    # total loss; ensure real scalar even if inputs are complex
    return jnp.real(mse + penalty)

def exp_val(obtained):
    return obtained

def cross_entropy(true_labels, obtained_outputs):
    obtained_probs = softmax_discretization(obtained_outputs)
    return jnp.array((-jnp.sum(true_labels * jnp.log(obtained_probs))) / len(obtained_outputs), dtype=jnp.float64)

def retrieve_loss_function(loss_name):
    if loss_name == 'mse':
        return mse
    elif loss_name == 'exp_val':
        return exp_val
    elif loss_name == 'cross_entropy':
        return cross_entropy