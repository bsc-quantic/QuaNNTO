import hashlib
import json
from importlib.metadata import version, PackageNotFoundError

from quannto.utils.path_utils import quannto_results_dir

import numpy as np
from functools import reduce, partial
import time
import scipy.optimize as opt
import optax
import jax
import jax.numpy as jnp

from .loss_functions import mse
from .qnn import QNN


def _nongauss_symbol(is_addition):
    """Human-readable non-Gaussian symbol for printing."""
    return "â†" if is_addition else "â"


def _apply_fns(x, fns):
    """Applies a list of callables sequentially."""
    return reduce(lambda acc, fn: fn(acc), fns, x)


def _compute_num_parameters(N, layers, include_initial_squeezing, include_initial_mixing, is_passive_gaussian):
    """
    Computes the number of trainable parameters for the given architecture flags.
    """
    n_pars = layers * (2 * N**2 + 3 * N)
    if include_initial_squeezing:
        n_pars += N
    if include_initial_mixing:
        n_pars += N**2
    if is_passive_gaussian:
        n_pars -= layers * (N**2 + N)
    return int(n_pars)


def _build_bounds(N, layers, include_initial_squeezing, include_initial_mixing, is_passive_gaussian):
    """
    Builds SciPy-style bounds list in the exact parameter order used by QNN.build_QNN.
    """
    passive_bounds = (None, None)
    init_sqz_bounds = (-np.log(5.5), np.log(5.5))  # Initial squeezing
    sqz_bounds = (-np.log(5.5), np.log(5.5))       # Layer squeezing
    disp_bounds = (None, None)

    bounds = []

    # Initial squeezing
    if include_initial_squeezing:
        bounds += [init_sqz_bounds for _ in range(N)]

    # Initial mixing (passive)
    if include_initial_mixing:
        bounds += [passive_bounds for _ in range(N**2)]

    # Layer-by-layer
    for _ in range(layers):
        if is_passive_gaussian:
            bounds += [passive_bounds for _ in range(N**2)]   # Q1
            bounds += [disp_bounds for _ in range(2 * N)]     # displacement
        else:
            bounds += [passive_bounds for _ in range(2 * N**2)]  # Q1, Q2
            bounds += [sqz_bounds for _ in range(N)]             # squeezing
            bounds += [disp_bounds for _ in range(2 * N)]        # displacement

    return bounds


def _init_parameters(n_pars, init_pars=None, rng=None, dtype=np.float64):
    """
    Creates or validates the initial parameter vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    if init_pars is None:
        return rng.random(n_pars, dtype=dtype)

    init_pars = np.asarray(init_pars, dtype=dtype)
    assert init_pars.shape == (n_pars,), f"init_pars shape {init_pars.shape} does not match ({n_pars},)"
    return init_pars


def _build_projector(bounds, dtype=jnp.float64):
    """
    Builds a simple projection function params -> clip(params, lb, ub) from SciPy-style bounds.
    """
    lb = jnp.array([b[0] if b[0] is not None else -jnp.inf for b in bounds], dtype=dtype)
    ub = jnp.array([b[1] if b[1] is not None else  jnp.inf for b in bounds], dtype=dtype)

    def project(params):
        return jnp.clip(params, lb, ub)

    return project, lb, ub


def _build_qnn_and_preprocess_datasets(
    *,
    model_name, task_name,
    N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable,
    include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    train_set, valid_set,
    init_pars,
    in_preprocs=None, out_prepocs=None, postprocs=None,
):
    """
    Builds the QNN object and applies preprocessors to training/validation datasets.
    """
    if in_preprocs is None:
        in_preprocs = []
    if out_prepocs is None:
        out_prepocs = []
    if postprocs is None:
        postprocs = []

    qnn = QNN(
        model_name, task_name,
        N, layers, n_inputs, n_outputs,
        ladder_modes, is_addition, observable,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
        init_pars,
        in_preprocs, out_prepocs, postprocs,
    )

    # Preprocess datasets (Python side, once)
    train_inputs = _apply_fns(train_set[0], qnn.in_preprocessors)
    train_outputs = _apply_fns(train_set[1], qnn.out_preprocessors)

    if valid_set is not None:
        valid_inputs = _apply_fns(valid_set[0], qnn.in_preprocessors)
        valid_outputs = _apply_fns(valid_set[1], qnn.out_preprocessors)
    else:
        valid_inputs = None
        valid_outputs = None

    return qnn, train_inputs, train_outputs, valid_inputs, valid_outputs




LBFGSB_DEFAULTS = {
    "maxcor": 10, "ftol": 2.220446049250313e-9, "gtol": 1e-5,
    "eps": 1e-8, "maxfun": 15000, "maxiter": 15000, "maxls": 20,
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _array_fingerprint(value):
    a = np.ascontiguousarray(np.asarray(value))
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def _finalize_model(qnn, best_params, *, save=True):
    """Return and persist the same checkpoint in all model representations."""
    best_params = np.asarray(best_params, dtype=np.float64).copy()
    qnn.build_QNN(best_params)
    qnn.print_qnn()
    if save:
        _save_checkpoint_files(qnn, best_params)
    return qnn


def _save_checkpoint_files(qnn, params):
    qnn.build_QNN(np.asarray(params, dtype=np.float64))
    qnn.save_model_parameters(params)
    qnn.save_operator_matrices()
    qnn.save_model()
    if hasattr(qnn, "training_record"):
        folder = quannto_results_dir() / "models" / "training_records"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{qnn.model_name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_json_safe(qnn.training_record),
                                        indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


class _CheckpointTracker:
    """Select over recorded iterates, never over finite-difference probes."""

    def __init__(self, has_validation):
        self.has_validation = has_validation
        self.best_params = None
        self.best_index = None
        self.best_metric = np.inf
        self.records = []
        self.global_it = 0
        self.local_it = 0
        self.bh_it = 0
        self.saved_best_index = None

    def record(self, params, train_loss, validation_loss, *, phase, event,
               basin=None, iteration=None, accepted=None):
        params = np.asarray(params, dtype=np.float64).copy()
        train_loss = float(train_loss)
        validation_loss = (float(validation_loss) if self.has_validation
                           else train_loss)
        eligible = (np.all(np.isfinite(params)) and np.isfinite(train_loss)
                    and np.isfinite(validation_loss))
        metric = validation_loss if self.has_validation else train_loss
        self.records.append({"phase": phase, "event": event, "basin": basin,
                             "iteration": iteration, "accepted": accepted,
                             "train_loss": train_loss,
                             "validation_loss": validation_loss,
                             "eligible": bool(eligible)})
        if eligible and metric < self.best_metric:
            self.best_metric = metric
            self.best_index = len(self.records) - 1
            self.best_params = params

    def require_checkpoint(self):
        if self.best_params is None:
            raise ValueError("No finite checkpoint was obtained; check the inputs, "
                             "initial parameters and conditional-state normalization.")

    @property
    def train_history(self):
        return [r["train_loss"] for r in self.records]

    @property
    def valid_history(self):
        return [r["validation_loss"] for r in self.records]

    def attach(self, qnn):
        self.require_checkpoint()
        selected = dict(self.records[self.best_index])
        selected.update({"history_index": self.best_index,
                         "metric": "validation_loss" if self.has_validation else "train_loss",
                         "value": self.best_metric})
        qnn.training_record["selection"] = selected
        qnn.training_record["history"] = list(self.records)
        qnn.training_record["lbfgsb_iterations"] = self.global_it


class _ObjectiveCache:
    def __init__(self, fn):
        self.fn = fn
        self.last_x = None
        self.last_value = None

    def __call__(self, x):
        self.last_value = float(self.fn(x))
        self.last_x = np.asarray(x).copy()
        return self.last_value

    def get(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.last_value
        return self(x)


def callback_opt_general(xk, *, state, cost_cache, cost_validation=None, print_every=1):
    train_loss = cost_cache.get(xk)
    val_loss = float(cost_validation(xk)) if cost_validation is not None else train_loss
    state.local_it += 1
    state.global_it += 1
    state.record(xk, train_loss, val_loss, phase="lbfgsb", event="iterate",
                 basin=state.bh_it, iteration=state.local_it)
    if print_every and state.local_it % print_every == 0:
        print(f"BH {state.bh_it:3d} | Iteration {state.local_it:4d} | "
              f"Train {train_loss:.6e} | Selection loss {val_loss:.6e}",
              end="\r", flush=True)


def callback_hopping_general(x, f, accept, *, state, qnn, cost_cache,
                             cost_validation=None, save_best=True, print_every=1):
    # Re-evaluate at this exact endpoint, even when no local callback occurred.
    train_loss = cost_cache.get(x)
    val_loss = float(cost_validation(x)) if cost_validation is not None else train_loss
    state.record(x, train_loss, val_loss, phase="lbfgsb", event="basin_end",
                 basin=state.bh_it, iteration=state.local_it, accepted=bool(accept))
    if state.best_params is not None:
        state.attach(qnn)
        if save_best and state.saved_best_index != state.best_index:
            _save_checkpoint_files(qnn, state.best_params)
            state.saved_best_index = state.best_index
    if print_every:
        print(f"\nLocal solve {state.bh_it + 1}: train={train_loss:.6e}; "
              f"best selection loss={state.best_metric:.6e}")
    state.bh_it += 1
    state.local_it = 0


def _run_scipy_basinhopping(*, qnn, init_pars, bounds, hopping_iters,
                           training_QNN, validate_QNN=None, save_best=True,
                           print_every=1, state=None, rng=None,
                           lbfgsb_options=None, bh_temperature=1.0,
                           bh_stepsize=0.5, bh_interval=50):
    if not isinstance(hopping_iters, (int, np.integer)) or hopping_iters < 0:
        raise ValueError("hopping_iters must be a nonnegative integer.")
    state = state if state is not None else _CheckpointTracker(validate_QNN is not None)
    cache = _ObjectiveCache(training_QNN)
    train0 = cache.get(init_pars)
    valid0 = float(validate_QNN(init_pars)) if validate_QNN is not None else train0
    state.record(init_pars, train0, valid0, phase="lbfgsb", event="initial",
                 basin=0, iteration=0)
    state.require_checkpoint()
    options = dict(LBFGSB_DEFAULTS)
    if lbfgsb_options is not None:
        options.update(lbfgsb_options)
    callback = partial(callback_opt_general, state=state, cost_cache=cache,
                       cost_validation=validate_QNN, print_every=print_every)
    hop_callback = partial(callback_hopping_general, state=state, qnn=qnn,
                           cost_cache=cache, cost_validation=validate_QNN,
                           save_best=save_best, print_every=print_every)
    t0 = time.perf_counter()
    result = opt.basinhopping(
        cache, np.asarray(init_pars, dtype=np.float64), niter=int(hopping_iters),
        T=bh_temperature, stepsize=bh_stepsize, interval=bh_interval,
        minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds,
                          "callback": callback, "options": options},
        callback=hop_callback, seed=rng,
        target_accept_rate=0.5, stepwise_factor=0.9,
    )
    elapsed = time.perf_counter() - t0
    lowest = result.lowest_optimization_result
    qnn.training_record["scipy_result"] = {
        "message": list(result.message), "nit": int(result.nit),
        "minimization_failures": int(result.minimization_failures),
        "lowest_training_local_result": {
            key: getattr(lowest, key, None)
            for key in ("success", "status", "message", "fun", "nit", "nfev", "njev")
        },
    }
    # The Metropolis accept/reject rule uses training loss. Selection is independent.
    state.require_checkpoint()
    state.attach(qnn)
    return state.best_params.copy(), state.train_history, state.valid_history, elapsed


def _run_adam_warmup(*, init_pars, training_QNN, validate_QNN=None,
                    project=None, epochs=200, learning_rate=1e-2,
                    weight_decay=0.0, print_every=1, state=None):
    if not isinstance(epochs, (int, np.integer)) or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer.")
    state = state if state is not None else _CheckpointTracker(validate_QNN is not None)
    params = jnp.asarray(init_pars, dtype=jnp.float64)
    if project is not None:
        params = project(params)
    optimizer = (optax.adamw(learning_rate, weight_decay=weight_decay)
                 if weight_decay else optax.adam(learning_rate))
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return training_QNN(project(p) if project is not None else p)

    @jax.jit
    def step(p, s):
        _, gradient = jax.value_and_grad(loss_fn)(p)
        updates, s = optimizer.update(gradient, s, p)
        p = optax.apply_updates(p, updates)
        if project is not None:
            p = project(p)
        return p, s

    def record(p, iteration):
        # Both losses and the saved parameters belong to the post-update point.
        train = float(loss_fn(p))
        valid = float(validate_QNN(p)) if validate_QNN is not None else train
        state.record(p, train, valid, phase="adam", iteration=iteration,
                     event="initial" if iteration == 0 else "iterate")
        return train, valid

    t0 = time.perf_counter()
    record(params, 0)
    state.require_checkpoint()
    for epoch in range(1, epochs + 1):
        params, opt_state = step(params, opt_state)
        train, valid = record(params, epoch)
        if print_every and epoch % print_every == 0:
            print(f"[Adam] {epoch}/{epochs} | Train {train:.6e} | "
                  f"Selection loss {valid:.6e}", end="\r", flush=True)
    if print_every:
        print()
    return (state.best_params.copy(), state.train_history, state.valid_history,
            time.perf_counter() - t0)


def _train_scipy_protocol(
    model_name, task_name, N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable, include_initial_squeezing,
    include_initial_mixing, is_passive_gaussian, train_set, valid_set,
    loss_function, hopping_iters, in_preprocs, out_prepocs, postprocs,
    init_pars, save, print_every, save_best_each_basin, *,
    seed, lbfgsb_options, bh_temperature, bh_stepsize, bh_interval,
    run_metadata, adam_epochs=0, adam_learning_rate=1e-2,
    adam_weight_decay=0.0, hybrid=False,
):
    if layers != len(ladder_modes):
        raise ValueError("Provide one list of ladder modes per physical layer.")
    for name, data in (("train", train_set), ("validation", valid_set)):
        if data is None and name == "validation":
            continue
        if data is None or len(data[0]) == 0 or len(data[0]) != len(data[1]):
            raise ValueError(f"{name} data must contain equally many, nonempty inputs and targets.")
    if print_every < 0:
        raise ValueError("print_every must be nonnegative (0 disables progress output).")
    root_seed = np.random.SeedSequence(seed)
    init_sequence, basin_sequence = root_seed.spawn(2)
    init_rng = np.random.default_rng(init_sequence)
    basin_rng = np.random.default_rng(basin_sequence)
    n_pars = _compute_num_parameters(N, layers, include_initial_squeezing,
                                    include_initial_mixing, is_passive_gaussian)
    bounds = _build_bounds(N, layers, include_initial_squeezing,
                           include_initial_mixing, is_passive_gaussian)
    supplied_initial = init_pars is not None
    init_pars = _init_parameters(n_pars, init_pars=init_pars, rng=init_rng)
    if not np.all(np.isfinite(init_pars)):
        raise ValueError("Initial parameters must be finite.")
    project, _, _ = _build_projector(bounds)
    projected = np.asarray(project(jnp.asarray(init_pars)), dtype=np.float64)
    initial_projected = not np.array_equal(projected, init_pars)
    init_pars = projected
    qnn, x, y, xv, yv = _build_qnn_and_preprocess_datasets(
        model_name=model_name, task_name=task_name, N=N, layers=layers,
        n_inputs=n_inputs, n_outputs=n_outputs, ladder_modes=ladder_modes,
        is_addition=is_addition, observable=observable,
        include_initial_squeezing=include_initial_squeezing,
        include_initial_mixing=include_initial_mixing,
        is_passive_gaussian=is_passive_gaussian, train_set=train_set,
        valid_set=valid_set, init_pars=init_pars,
        in_preprocs=list(in_preprocs or []), out_prepocs=list(out_prepocs or []),
        postprocs=list(postprocs or []),
    )
    train = partial(qnn.train_QNN, inputs_dataset=jnp.asarray(x),
                    outputs_dataset=jnp.asarray(y), loss_function=loss_function)
    valid = (partial(qnn.train_QNN, inputs_dataset=jnp.asarray(xv),
                     outputs_dataset=jnp.asarray(yv), loss_function=loss_function)
             if valid_set is not None else None)
    options = dict(LBFGSB_DEFAULTS)
    options.update(lbfgsb_options or {})
    versions = {}
    for package in ("numpy", "scipy", "jax", "jaxlib", "optax"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unknown"
    qnn.training_record = {
        "schema_version": 1,
        "trainer": "train_hybrid_adamjax_scipy" if hybrid else "train_scipy",
        "model_name": model_name, "task_name": task_name,
        "architecture": {"N": N, "layers": layers, "n_inputs": n_inputs,
                         "n_outputs": n_outputs, "ladder_modes": ladder_modes,
                         "is_addition": is_addition, "observable": observable,
                         "include_initial_squeezing": include_initial_squeezing,
                         "include_initial_mixing": include_initial_mixing,
                         "is_passive_gaussian": is_passive_gaussian},
        "initialization": {"source": "provided" if supplied_initial else "uniform_[0,1)",
                           "seed": seed, "seed_sequence_entropy": root_seed.entropy,
                           "initialization_spawn_key": list(init_sequence.spawn_key),
                           "basinhopping_spawn_key": list(basin_sequence.spawn_key),
                           "bit_generator": type(init_rng.bit_generator).__name__,
                           "projected_to_bounds": initial_projected,
                           "parameters": init_pars.tolist()},
        "bounds": bounds, "n_parameters": n_pars, "dtype": "float64",
        "loss_function": f"{loss_function.__module__}.{loss_function.__name__}",
        "lbfgsb": {"options": options, "jac": None,
                   "derivative_method": "SciPy forward finite differences"},
        "basinhopping": {"niter": hopping_iters, "T": bh_temperature,
                         "stepsize": bh_stepsize, "interval": bh_interval,
                         "target_accept_rate": 0.5, "stepwise_factor": 0.9,
                         "niter_success": None},
        "adam": {"epochs": adam_epochs, "learning_rate": adam_learning_rate,
                 "weight_decay": adam_weight_decay, "b1": 0.9, "b2": 0.999,
                 "eps": 1e-8, "eps_root": 0.0} if hybrid else None,
        "software_versions": versions,
        "data": {"train_inputs": _array_fingerprint(train_set[0]),
                 "train_targets": _array_fingerprint(train_set[1]),
                 "processed_train_inputs": _array_fingerprint(x),
                 "processed_train_targets": _array_fingerprint(y)},
        "run_metadata": dict(run_metadata or {}),
        "history_note": "Chronological initial points, Adam/local iterates and basin endpoints; "
                        "the selected model is at selection.history_index, not necessarily the last entry.",
    }
    if valid_set is not None:
        qnn.training_record["data"].update({
            "validation_inputs": _array_fingerprint(valid_set[0]),
            "validation_targets": _array_fingerprint(valid_set[1]),
            "processed_validation_inputs": _array_fingerprint(xv),
            "processed_validation_targets": _array_fingerprint(yv),
        })
    state = _CheckpointTracker(valid is not None)
    t0 = time.perf_counter()
    if hybrid:
        init_pars, _, _, _ = _run_adam_warmup(
            init_pars=init_pars, training_QNN=train, validate_QNN=valid,
            project=project, epochs=adam_epochs, learning_rate=adam_learning_rate,
            weight_decay=adam_weight_decay, print_every=print_every, state=state,
        )
    best, train_history, valid_history, _ = _run_scipy_basinhopping(
        qnn=qnn, init_pars=init_pars, bounds=bounds, hopping_iters=hopping_iters,
        training_QNN=train, validate_QNN=valid,
        save_best=(save and save_best_each_basin), print_every=print_every,
        state=state, rng=basin_rng, lbfgsb_options=options,
        bh_temperature=bh_temperature, bh_stepsize=bh_stepsize, bh_interval=bh_interval,
    )
    qnn.training_record["elapsed_seconds"] = time.perf_counter() - t0
    state.attach(qnn)
    _finalize_model(qnn, best, save=save)
    if print_every:
        print(f"Selected {qnn.training_record['selection']['metric']}: "
              f"{state.best_metric:.6e}; history index {state.best_index}")
    return qnn, train_history, valid_history


def train_scipy(
    model_name, task_name, N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable, include_initial_squeezing,
    include_initial_mixing, is_passive_gaussian, train_set, valid_set,
    loss_function=mse, hopping_iters=2,
    in_preprocs=None, out_prepocs=None, postprocs=None,
    init_pars=None, save=True, print_every=1, save_best_each_basin=True,
    *, seed=42, lbfgsb_options=None, bh_temperature=1.0,
    bh_stepsize=0.5, bh_interval=50, run_metadata=None,
):
    """L-BFGS-B/basinhopping; retain the best recorded validation checkpoint.

    Without validation select by training loss. ``hopping_iters=0`` still runs
    one local minimization. ``seed`` controls independent initialization and
    basinhopping streams. Histories cover the full run; the selected entry is
    ``qnn.training_record['selection']['history_index']``.
    """
    return _train_scipy_protocol(
        model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes,
        is_addition, observable, include_initial_squeezing, include_initial_mixing,
        is_passive_gaussian, train_set, valid_set, loss_function, hopping_iters,
        in_preprocs, out_prepocs, postprocs, init_pars, save, print_every,
        save_best_each_basin, seed=seed, lbfgsb_options=lbfgsb_options,
        bh_temperature=bh_temperature, bh_stepsize=bh_stepsize,
        bh_interval=bh_interval, run_metadata=run_metadata,
    )


def train_hybrid_adamjax_scipy(
    model_name, task_name, N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable, include_initial_squeezing,
    include_initial_mixing, is_passive_gaussian, train_set, valid_set,
    loss_function=mse, hopping_iters=2,
    in_preprocs=None, out_prepocs=None, postprocs=None,
    init_pars=None, save=True, adam_epochs=200, adam_learning_rate=1e-2,
    adam_weight_decay=0.0, print_every=1, save_best_each_basin=True,
    *, seed=42, lbfgsb_options=None, bh_temperature=1.0,
    bh_stepsize=0.5, bh_interval=50, run_metadata=None,
):
    """Projected Adam followed by SciPy, with one selection across both phases."""
    return _train_scipy_protocol(
        model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes,
        is_addition, observable, include_initial_squeezing, include_initial_mixing,
        is_passive_gaussian, train_set, valid_set, loss_function, hopping_iters,
        in_preprocs, out_prepocs, postprocs, init_pars, save, print_every,
        save_best_each_basin, seed=seed, lbfgsb_options=lbfgsb_options,
        bh_temperature=bh_temperature, bh_stepsize=bh_stepsize,
        bh_interval=bh_interval, run_metadata=run_metadata,
        adam_epochs=adam_epochs, adam_learning_rate=adam_learning_rate,
        adam_weight_decay=adam_weight_decay, hybrid=True,
    )


# Alternative trainers (not used by the supplied task scripts).
def _run_optax_lbfgs(
    *,
    init_pars,
    loss_fn_jit,
    val_loss_fn_jit=None,
    project=None,
    max_iter=200,
    tol=1e-6,
    print_every=1,
):
    """
    Optax L-BFGS loop. Returns (best_params_np, train_hist, valid_hist, train_time).
    """
    params = jnp.asarray(init_pars, dtype=jnp.float64)
    if project is not None:
        params = project(params)

    opt_lbfgs = optax.lbfgs()
    opt_state = opt_lbfgs.init(params)

    value_and_grad = optax.value_and_grad_from_state(loss_fn_jit)

    train_hist = []
    valid_hist = []

    # Initial evaluation
    train0 = float(loss_fn_jit(params))
    if val_loss_fn_jit is not None:
        valid0 = float(val_loss_fn_jit(params))
    else:
        valid0 = train0

    train_hist.append(train0)
    valid_hist.append(valid0)

    best_params = params
    best_valid = valid0

    t0 = time.time()
    for it in range(1, max_iter + 1):
        value, grad = value_and_grad(params, state=opt_state)

        updates, opt_state = opt_lbfgs.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=loss_fn_jit,
        )
        params = optax.apply_updates(params, updates)
        if project is not None:
            params = project(params)

        train_loss = float(value)
        if val_loss_fn_jit is not None:
            val_loss = float(val_loss_fn_jit(params))
        else:
            val_loss = train_loss

        train_hist.append(train_loss)
        valid_hist.append(val_loss)

        if (it % print_every) == 0:
            print(
                f"[L-BFGS] Epoch {it}/{max_iter} | train loss = {train_loss:.6e} | val loss = {val_loss:.6e}",
                end="\r",
                flush=True,
            )

        if val_loss < best_valid:
            best_valid = val_loss
            best_params = params

        grad_norm = float(jnp.linalg.norm(grad))
        if grad_norm < tol:
            print()
            print(f"[L-BFGS] Early stopping at iter {it}, ||grad|| = {grad_norm:.3e}")
            break

    train_time = time.time() - t0
    print()

    return np.asarray(best_params, dtype=np.float64), train_hist, valid_hist, train_time


def train_gdjax(
    model_name, task_name,
    N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable,
    include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    train_set, valid_set,
    loss_function=mse, hopping_iters=200,
    in_preprocs=None, out_prepocs=None, postprocs=None,
    init_pars=None, save=True,
    learning_rate=5e-3,
    print_every=1,
):
    """
    Creates and trains a QNN using pure JAX gradient descent.
    hopping_iters: number of epochs.
    """
    assert layers == len(ladder_modes)

    # ---------------------------------------------------------------
    # 1. Number of parameters and initialization
    # ---------------------------------------------------------------
    n_pars = _compute_num_parameters(
        N, layers,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    )
    init_pars = _init_parameters(n_pars, init_pars=init_pars)

    # ---------------------------------------------------------------
    # 2. Build QNN and preprocess datasets
    # ---------------------------------------------------------------
    qnn, train_inputs, train_outputs, valid_inputs, valid_outputs = _build_qnn_and_preprocess_datasets(
        model_name=model_name,
        task_name=task_name,
        N=N,
        layers=layers,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        ladder_modes=ladder_modes,
        is_addition=is_addition,
        observable=observable,
        include_initial_squeezing=include_initial_squeezing,
        include_initial_mixing=include_initial_mixing,
        is_passive_gaussian=is_passive_gaussian,
        train_set=train_set,
        valid_set=valid_set,
        init_pars=init_pars,
        in_preprocs=in_preprocs,
        out_prepocs=out_prepocs,
        postprocs=postprocs,
    )

    train_inputs = jnp.asarray(train_inputs)
    train_outputs = jnp.asarray(train_outputs)

    if valid_inputs is not None:
        valid_inputs = jnp.asarray(valid_inputs)
        valid_outputs = jnp.asarray(valid_outputs)

    # ---------------------------------------------------------------
    # 3. Define loss function and gradient
    # ---------------------------------------------------------------
    params = jnp.asarray(init_pars, dtype=jnp.float64)

    def loss_fn(p):
        return qnn.train_QNN(
            p,
            inputs_dataset=train_inputs,
            outputs_dataset=train_outputs,
            loss_function=loss_function,
        )

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # ---------------------------------------------------------------
    # 4. Training loop
    # ---------------------------------------------------------------
    train_hist = []
    valid_hist = []

    best_params = params
    best_valid = np.inf

    t0 = time.time()
    for epoch in range(1, hopping_iters + 1):
        loss_val, grads = loss_and_grad(params)
        params = params - learning_rate * grads

        train_loss = float(loss_val)
        if valid_inputs is not None:
            val_loss = float(
                qnn.train_QNN(
                    params,
                    inputs_dataset=valid_inputs,
                    outputs_dataset=valid_outputs,
                    loss_function=loss_function,
                )
            )
        else:
            val_loss = train_loss

        train_hist.append(train_loss)
        valid_hist.append(val_loss)

        if val_loss < best_valid:
            best_valid = val_loss
            best_params = params

        if (epoch % print_every) == 0:
            print(
                f"[GD] Epoch {epoch}/{hopping_iters} | train loss = {train_loss:.6e} | val loss = {val_loss:.6e}",
                end="\r",
                flush=True,
            )

    train_time = time.time() - t0
    print()
    print(f"Total training time: {train_time:.3f} seconds")
    print(f"Time per epoch: {train_time / max(1, hopping_iters):.6f} seconds")
    print(
        f"\nFINAL ERROR FOR N={N}, L={layers}, "
        f"{_nongauss_symbol(is_addition)} modes={np.array(ladder_modes[0]) + 1}\n"
        f"{best_valid}"
    )

    _finalize_model(qnn, np.asarray(best_params), save=save)

    return qnn, train_hist, valid_hist


def train_optax_lbfgs(
    model_name, task_name,
    N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable,
    include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    train_set, valid_set,
    loss_function=mse, hopping_iters=200,
    in_preprocs=None, out_prepocs=None, postprocs=None,
    init_pars=None, save=True,
    tol=1e-6,
    print_every=1,
):
    """
    Creates and trains a QNN using Optax L-BFGS.
    """
    assert layers == len(ladder_modes)

    # ---------------------------------------------------------------
    # 1. Number of parameters and bounds (+ projector)
    # ---------------------------------------------------------------
    n_pars = _compute_num_parameters(
        N, layers,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    )
    bounds = _build_bounds(N, layers, include_initial_squeezing, include_initial_mixing, is_passive_gaussian)
    assert len(bounds) == n_pars, f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."

    init_pars = _init_parameters(n_pars, init_pars=init_pars, dtype=np.float64)

    project, _, _ = _build_projector(bounds, dtype=jnp.float64)

    # ---------------------------------------------------------------
    # 2. Build QNN and preprocess datasets
    # ---------------------------------------------------------------
    qnn, train_inputs, train_outputs, valid_inputs, valid_outputs = _build_qnn_and_preprocess_datasets(
        model_name=model_name,
        task_name=task_name,
        N=N,
        layers=layers,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        ladder_modes=ladder_modes,
        is_addition=is_addition,
        observable=observable,
        include_initial_squeezing=include_initial_squeezing,
        include_initial_mixing=include_initial_mixing,
        is_passive_gaussian=is_passive_gaussian,
        train_set=train_set,
        valid_set=valid_set,
        init_pars=init_pars,
        in_preprocs=in_preprocs,
        out_prepocs=out_prepocs,
        postprocs=postprocs,
    )

    train_inputs = jnp.asarray(train_inputs)
    train_outputs = jnp.asarray(train_outputs)

    if valid_inputs is not None:
        valid_inputs = jnp.asarray(valid_inputs)
        valid_outputs = jnp.asarray(valid_outputs)

    # ---------------------------------------------------------------
    # 3. Define loss functions
    # ---------------------------------------------------------------
    training_QNN = partial(
        qnn.train_QNN,
        inputs_dataset=train_inputs,
        outputs_dataset=train_outputs,
        loss_function=loss_function,
    )

    if valid_inputs is not None:
        validate_QNN = partial(
            qnn.train_QNN,
            inputs_dataset=valid_inputs,
            outputs_dataset=valid_outputs,
            loss_function=loss_function,
        )
    else:
        validate_QNN = None

    def loss_fn(params):
        return training_QNN(project(params))

    loss_fn_jit = jax.jit(loss_fn)

    if validate_QNN is not None:
        def val_loss_fn(params):
            return validate_QNN(project(params))
        val_loss_fn_jit = jax.jit(val_loss_fn)
    else:
        val_loss_fn_jit = None

    # ---------------------------------------------------------------
    # 4. Optax L-BFGS loop
    # ---------------------------------------------------------------
    best_params, train_hist, valid_hist, train_time = _run_optax_lbfgs(
        init_pars=init_pars,
        loss_fn_jit=loss_fn_jit,
        val_loss_fn_jit=val_loss_fn_jit,
        project=project,
        max_iter=hopping_iters,
        tol=tol,
        print_every=print_every,
    )

    print(f"Total training time: {train_time:.3f} seconds")
    print(f"Time per epoch: {train_time / max(1, len(train_hist)):.6f} seconds")
    print(
        f"\nOPTIMIZATION ERROR FOR N={N}, L={layers}, "
        f"{_nongauss_symbol(is_addition)} modes={np.array(ladder_modes[0]) + 1}\n"
        f"{valid_hist[-1]}"
    )

    _finalize_model(qnn, best_params, save=save)

    return qnn, train_hist, valid_hist


def train_adamjax(
    model_name, task_name,
    N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable,
    include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    train_set, valid_set,
    loss_function=mse, hopping_iters=200,
    in_preprocs=None, out_prepocs=None, postprocs=None,
    init_pars=None, save=True,
    learning_rate=1e-2,
    weight_decay=0.0,
    print_every=1,
):
    """
    Creates and trains a QNN using pure-JAX Adam/AdamW.
    """
    assert layers == len(ladder_modes)

    # ---------------------------------------------------------------
    # 1. Number of parameters and bounds (+ projector)
    # ---------------------------------------------------------------
    n_pars = _compute_num_parameters(
        N, layers,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    )
    bounds = _build_bounds(N, layers, include_initial_squeezing, include_initial_mixing, is_passive_gaussian)
    assert len(bounds) == n_pars, f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."

    init_pars = _init_parameters(n_pars, init_pars=init_pars, dtype=np.float64)
    project, _, _ = _build_projector(bounds, dtype=jnp.float64)

    # ---------------------------------------------------------------
    # 2. Build QNN and preprocess datasets
    # ---------------------------------------------------------------
    qnn, train_inputs, train_outputs, valid_inputs, valid_outputs = _build_qnn_and_preprocess_datasets(
        model_name=model_name,
        task_name=task_name,
        N=N,
        layers=layers,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        ladder_modes=ladder_modes,
        is_addition=is_addition,
        observable=observable,
        include_initial_squeezing=include_initial_squeezing,
        include_initial_mixing=include_initial_mixing,
        is_passive_gaussian=is_passive_gaussian,
        train_set=train_set,
        valid_set=valid_set,
        init_pars=init_pars,
        in_preprocs=in_preprocs,
        out_prepocs=out_prepocs,
        postprocs=postprocs,
    )

    train_inputs = jnp.asarray(train_inputs)
    train_outputs = jnp.asarray(train_outputs)

    if valid_inputs is not None:
        valid_inputs = jnp.asarray(valid_inputs)
        valid_outputs = jnp.asarray(valid_outputs)

    # ---------------------------------------------------------------
    # 3. Define training/validation functions (JAX)
    # ---------------------------------------------------------------
    training_QNN = partial(
        qnn.train_QNN,
        inputs_dataset=train_inputs,
        outputs_dataset=train_outputs,
        loss_function=loss_function,
    )

    if valid_inputs is not None:
        validate_QNN = partial(
            qnn.train_QNN,
            inputs_dataset=valid_inputs,
            outputs_dataset=valid_outputs,
            loss_function=loss_function,
        )
    else:
        validate_QNN = None

    # ---------------------------------------------------------------
    # 4. Adam/AdamW training
    # ---------------------------------------------------------------
    best_params, train_hist, valid_hist, train_time = _run_adam_warmup(
        init_pars=init_pars,
        training_QNN=training_QNN,
        validate_QNN=validate_QNN,
        project=project,
        epochs=hopping_iters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        print_every=print_every,
    )

    print(f"Total training time: {train_time:.3f} seconds")
    print(f"Time per epoch: {train_time / max(1, hopping_iters):.6f} seconds")
    print(
        f"\nFINAL ERROR FOR N={N}, L={layers}, "
        f"{_nongauss_symbol(is_addition)} modes={np.array(ladder_modes[0]) + 1}\n"
        f"{valid_hist[-1] if len(valid_hist) else np.nan}"
    )

    _finalize_model(qnn, best_params, save=save)

    return qnn, train_hist, valid_hist

