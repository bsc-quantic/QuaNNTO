"""Memory-bounded exact Wick--Isserlis evaluation.

The legacy QuaNNTO backend enumerates every loop-perfect matching and is very
fast for short operator strings.  Its vectorized intermediate, however, grows
with the number of matchings and can become much larger than the final scalar
moment.  This module evaluates the same sum with a subset dynamic program and
maps terms sequentially, trading some runtime for a substantially lower peak
memory footprint.
"""
from __future__ import annotations

from functools import lru_cache

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


ONE_OVER_SQRT_TWO = 1.0 / np.sqrt(2.0)


@lru_cache(maxsize=None)
def _build_subset_tables(term_length: int):
    """Return the exact recurrence tables for one operator-string length."""
    if term_length < 2:
        raise ValueError("subset Wick evaluation requires term_length >= 2")

    state_count = 1 << term_length
    partner_slots = term_length - 1
    first_index = np.full(state_count, -1, dtype=np.int32)
    loop_submask = np.zeros(state_count, dtype=np.int32)
    partner_count = np.zeros(state_count, dtype=np.int32)
    partner_index = np.zeros((state_count, partner_slots), dtype=np.int32)
    pair_submask = np.zeros((state_count, partner_slots), dtype=np.int32)

    for mask in range(1, state_count):
        first_bit = mask & -mask
        first = first_bit.bit_length() - 1
        remaining = mask ^ first_bit
        first_index[mask] = first
        loop_submask[mask] = remaining

        slot = 0
        for partner in range(first + 1, term_length):
            partner_bit = 1 << partner
            if remaining & partner_bit:
                partner_index[mask, slot] = partner
                pair_submask[mask, slot] = remaining ^ partner_bit
                slot += 1
        partner_count[mask] = slot

    return first_index, loop_submask, partner_count, partner_index, pair_submask


@lru_cache(maxsize=None)
def _make_subset_kernel(term_length: int):
    """Build a JAX-compatible exact loop-hafnian recurrence."""
    tables = _build_subset_tables(term_length)
    state_count = 1 << term_length
    partner_slots = term_length - 1

    def kernel(modes, types, n_modes, quadratic_moments, means_xxpp):
        first_index, loop_submask, partner_count, partner_index, pair_submask = (
            jnp.asarray(table, dtype=jnp.int32) for table in tables
        )
        slots = jnp.arange(partner_slots, dtype=jnp.int32)

        loop_weights = ONE_OVER_SQRT_TWO * (
            means_xxpp[modes]
            + 1j * (-2 * types + 1) * means_xxpp[n_modes + modes]
        )
        pair_kinds = types[:, None] + 2 * types[None, :]
        pair_weights = quadratic_moments[
            pair_kinds,
            modes[:, None],
            modes[None, :],
        ]
        value_dtype = jnp.result_type(loop_weights.dtype, pair_weights.dtype)
        loop_weights = loop_weights.astype(value_dtype)
        pair_weights = pair_weights.astype(value_dtype)

        zero = jnp.zeros((), dtype=value_dtype)
        values = jnp.zeros((state_count,), dtype=value_dtype)
        values = values.at[0].set(jnp.ones((), dtype=value_dtype))

        def update(mask, current):
            first = first_index[mask]
            loop_value = loop_weights[first] * current[loop_submask[mask]]
            valid = slots < partner_count[mask]
            partners = jnp.where(valid, partner_index[mask], 0)
            submasks = jnp.where(valid, pair_submask[mask], 0)
            pair_values = jnp.where(
                valid,
                pair_weights[first, partners] * current[submasks],
                zero,
            )
            return current.at[mask].set(loop_value + jnp.sum(pair_values))

        values = lax.fori_loop(1, state_count, update, values)
        return values[-1]

    return kernel


def wick_moment_subset_dp(modes, types, n_modes, quadratic_moments, means_xxpp):
    """Evaluate one ordered Gaussian ladder moment with the exact recurrence."""
    term_length = int(modes.shape[0])
    return _make_subset_kernel(term_length)(
        modes, types, n_modes, quadratic_moments, means_xxpp
    )


@lru_cache(maxsize=None)
def _dispatch_branches(max_term_length: int):
    branches = []
    for term_length in range(2, max_term_length + 1):
        kernel = _make_subset_kernel(term_length)

        def branch(args, term_length=term_length, kernel=kernel):
            trace_index, term_index, n_modes, modes, types, quadratic, means = args
            return kernel(
                modes[trace_index, term_index, :term_length],
                types[trace_index, term_index, :term_length],
                n_modes,
                quadratic,
                means,
            )

        branches.append(branch)
    return tuple(branches)


def _wick_term(
    trace_index,
    term_index,
    n_modes,
    term_lengths,
    modes,
    types,
    quadratic_moments,
    means_xxpp,
):
    term_length = term_lengths[trace_index, term_index]

    def padded(_):
        return 0.0 + 0.0j

    def identity(_):
        return 1.0 + 0.0j

    def first_moment(_):
        mode = modes[trace_index, term_index, 0]
        operator_type = types[trace_index, term_index, 0]
        return ONE_OVER_SQRT_TWO * (
            means_xxpp[mode]
            + 1j * (-2 * operator_type + 1) * means_xxpp[n_modes + mode]
        )

    def higher_moment(_):
        max_term_length = int(modes.shape[-1])
        args = (
            trace_index,
            term_index,
            n_modes,
            modes,
            types,
            quadratic_moments,
            means_xxpp,
        )
        return lax.switch(term_length - 2, _dispatch_branches(max_term_length), args)

    return lax.cond(
        term_length < 0,
        padded,
        lambda _: lax.cond(
            term_length == 0,
            identity,
            lambda _: lax.cond(term_length == 1, first_moment, higher_moment, None),
            None,
        ),
        None,
    )


def _trace_value(
    trace_index,
    term_coefficients,
    n_modes,
    trace_term_indices,
    term_lengths,
    modes,
    types,
    quadratic_moments,
    means_xxpp,
):
    """Accumulate one trace sequentially to avoid a term-by-matching tensor."""
    term_count = int(trace_term_indices.shape[1])

    def term_value(position):
        term_index = trace_term_indices[trace_index, position]
        return term_coefficients[position] * _wick_term(
            trace_index,
            term_index,
            n_modes,
            term_lengths,
            modes,
            types,
            quadratic_moments,
            means_xxpp,
        )

    rematerialized_term = jax.checkpoint(term_value)

    def accumulate(position, total):
        return total + rematerialized_term(position)

    initial = jnp.asarray(0.0 + 0.0j, dtype=quadratic_moments.dtype)
    return lax.fori_loop(0, term_count, accumulate, initial)


def compute_exp_val_loop_subset_dp(
    trace_indices,
    trace_term_coefficients,
    n_modes,
    trace_term_indices,
    term_lengths,
    modes,
    types,
    quadratic_moments,
    means_xxpp,
):
    """Evaluate all compiled traces exactly with bounded intermediate memory."""
    def evaluate_trace(items):
        trace_index, coefficients = items
        return _trace_value(
            trace_index,
            coefficients,
            n_modes,
            trace_term_indices,
            term_lengths,
            modes,
            types,
            quadratic_moments,
            means_xxpp,
        )

    return lax.map(evaluate_trace, (trace_indices, trace_term_coefficients))
