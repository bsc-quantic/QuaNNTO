from __future__ import annotations

import json, hashlib, os
import numpy as np
from pathlib import Path

def arch_signature(*, N, layers, n_in, n_out, ladder_modes, is_addition, observable,
                   include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                   cache_version: int = 1):
    sig = {
        "cache_version": int(cache_version),
        "N": int(N),
        "layers": int(layers),
        "n_in": int(n_in),
        "n_out": int(n_out),
        "ladder_modes": ladder_modes,
        "is_addition": bool(is_addition),
        "observable": str(observable),
        "include_initial_squeezing": bool(include_initial_squeezing),
        "include_initial_mixing": bool(include_initial_mixing),
        "is_passive_gaussian": bool(is_passive_gaussian),
    }
    blob = json.dumps(sig, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(blob).hexdigest()[:16]
    return sig, h

def try_load_compiled(cache_root: Path, arch_hash: str):
    d = cache_root / f"arch_{arch_hash}"
    meta_path = d / "meta.json"
    npz_path  = d / "compiled.npz"
    if not (meta_path.is_file() and npz_path.is_file()):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    arrays = dict(np.load(npz_path, allow_pickle=False))
    return meta, arrays

def save_compiled(cache_root: Path, arch_hash: str, meta: dict, arrays: dict, atomic: bool = True):
    d = cache_root / f"arch_{arch_hash}"
    d.mkdir(parents=True, exist_ok=True)

    meta_path = d / "meta.json"
    npz_path  = d / "compiled.npz"

    if not atomic:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        np.savez_compressed(npz_path, **arrays)
        return

    # --- atomic write: write temp files then rename ---
    meta_tmp = d / "meta.tmp.json"
    npz_tmp  = d / "compiled.tmp.npz"

    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(meta_tmp, meta_path)

    np.savez_compressed(npz_tmp, **arrays)
    os.replace(npz_tmp, npz_path)

def extract_arch_arrays(qnn) -> dict:
    """
    Build a numpy-serializable dict for the architecture-dependent artifacts.
    Keep only deterministic, architecture-dependent arrays.
    """
    import jax

    arrays = {
        # ladder term metadata
        "np_modes":   np.asarray(qnn.np_modes, dtype=np.int16),
        "np_types":   np.asarray(qnn.np_types, dtype=np.int8),
        "lens_modes": np.asarray(qnn.lens_modes, dtype=np.int16),

        # wick matchings
        "jax_lpms":   np.asarray(jax.device_get(qnn.jax_lpms), dtype=np.int16),

        # bookkeeping
        "num_terms_per_trace": np.asarray(jax.device_get(qnn.num_terms_per_trace), dtype=np.int16),
        "num_terms_in_trace":  np.asarray(jax.device_get(qnn.num_terms_in_trace), dtype=np.int16),
        "max_terms":           np.asarray(int(qnn.max_terms), dtype=np.int32),
        "trace_terms_ranges":  np.asarray(jax.device_get(qnn.trace_terms_ranges), dtype=np.int16),
        "exp_vals_inds":       np.asarray(jax.device_get(qnn.exp_vals_inds), dtype=np.int32),
    }

    if hasattr(qnn, "jax_traceterms_coefs_inds"):
        arrays["jax_traceterms_coefs_inds"] = np.asarray(
            jax.device_get(qnn.jax_traceterms_coefs_inds), dtype=np.int32
        )
        arrays["jax_traceterms_coefs"] = np.asarray(
            jax.device_get(qnn.jax_traceterms_coefs), dtype=np.int32
        )
    else:
        arrays["jax_ones_coefs"] = np.asarray(jax.device_get(qnn.jax_ones_coefs), dtype=np.int8)

    return arrays

def apply_arch_arrays(qnn, arrays: dict) -> None:
    """
    Restore a cached architecture bundle onto a QNN instance.
    Does NOT rebuild SymPy expressions.
    """
    import jax
    import jax.numpy as jnp

    qnn.np_modes   = arrays["np_modes"]
    qnn.np_types   = arrays["np_types"]
    qnn.lens_modes = arrays["lens_modes"]

    qnn.jax_modes = jnp.array(qnn.np_modes)
    qnn.jax_types = jnp.array(qnn.np_types)
    qnn.jax_lens  = jnp.array(qnn.lens_modes)

    qnn.jax_lpms = jnp.array(arrays["jax_lpms"])

    qnn.num_terms_per_trace = jnp.array(arrays["num_terms_per_trace"])
    qnn.num_terms_in_trace  = jnp.array(arrays["num_terms_in_trace"])
    qnn.max_terms           = int(np.asarray(arrays["max_terms"]).item())

    qnn.trace_terms_ranges = jnp.array(arrays["trace_terms_ranges"])
    qnn.exp_vals_inds      = jnp.array(arrays["exp_vals_inds"], dtype=jnp.int32)

    if "jax_traceterms_coefs_inds" in arrays:
        qnn.jax_traceterms_coefs_inds = jnp.array(arrays["jax_traceterms_coefs_inds"], dtype=jnp.int32)
        qnn.jax_traceterms_coefs      = jnp.array(arrays["jax_traceterms_coefs"], dtype=jnp.int32)
    else:
        qnn.jax_ones_coefs = jnp.array(arrays["jax_ones_coefs"])

    # Put to device (optional but consistent with your current code)
    qnn.exp_vals_inds      = jax.device_put(qnn.exp_vals_inds)
    qnn.trace_terms_ranges = jax.device_put(qnn.trace_terms_ranges)
    qnn.jax_lens           = jax.device_put(qnn.jax_lens)
    qnn.jax_modes          = jax.device_put(qnn.jax_modes)
    qnn.jax_types          = jax.device_put(qnn.jax_types)
    qnn.jax_lpms           = jax.device_put(qnn.jax_lpms)
    if hasattr(qnn, "jax_traceterms_coefs_inds"):
        qnn.jax_traceterms_coefs_inds = jax.device_put(qnn.jax_traceterms_coefs_inds)
        qnn.jax_traceterms_coefs      = jax.device_put(qnn.jax_traceterms_coefs)
    if hasattr(qnn, "jax_ones_coefs"):
        qnn.jax_ones_coefs = jax.device_put(qnn.jax_ones_coefs)
