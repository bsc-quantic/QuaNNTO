import numpy as np
from functools import reduce, partial
import time
import scipy.optimize as opt
import optax
import jax
import jax.numpy as jnp

from .loss_functions import mse
from .qnn import QNN

# ---------------------------------------------------------------
# Trainers utilities
# ---------------------------------------------------------------
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

def _finalize_model(qnn, best_params, *, save=True):
    """
    Builds the final model at the best parameters and optionally saves it.
    """
    best_params = np.asarray(best_params, dtype=np.float64)
    qnn.build_QNN(best_params)
    qnn.print_qnn()
    if save:
        qnn.save_model()
    return qnn

# ---------------------------------------------------------------
# SciPy training helpers (L-BFGS-B + basinhopping)
# ---------------------------------------------------------------
class _ObjectiveCache:
    """
    Wrapper that caches the last x -> f(x) evaluation.
    Useful because SciPy callbacks do not receive f(x), so we would otherwise recompute it.
    """
    def __init__(self, fn):
        self.fn = fn
        self.last_x = None
        self.last_value = None

    def __call__(self, x):
        v = float(self.fn(x))
        self.last_x = np.asarray(x).copy()
        self.last_value = v
        return v

    def get(self, x):
        x = np.asarray(x)
        if self.last_x is not None and x.shape == self.last_x.shape and np.array_equal(x, self.last_x):
            return self.last_value
        return float(self.fn(x))

class _BasinhoppingState:
    """
    Tracks current and best histories across basinhopping restarts.
    """
    def __init__(self, init_pars):
        self.global_it = 0
        self.local_it = 0
        self.bh_it = 1

        self.current_train = []
        self.current_valid = []

        self.best_train = [np.inf]
        self.best_valid = [np.inf]
        self.best_pars = np.asarray(init_pars).copy()

    def _metric(self, has_validation):
        return self.current_valid[-1] if has_validation else self.current_train[-1]

    def _best_metric(self, has_validation):
        return self.best_valid[-1] if has_validation else self.best_train[-1]

def callback_opt_general(xk, *, state, cost_cache, cost_validation=None, print_every=1):
    """
    Prints and stores the loss value for each SciPy optimizer epoch.

    :param xk: QONN tunable parameters
    :param state: _BasinhoppingState
    :param cost_cache: _ObjectiveCache for training objective
    :param cost_validation: validation objective (callable) or None
    :param print_every: print frequency
    """
    train_loss = cost_cache.get(xk)
    state.current_train.append(train_loss)

    if cost_validation is not None:
        val_loss = float(cost_validation(xk))
    else:
        val_loss = train_loss
    state.current_valid.append(val_loss)

    state.local_it += 1
    state.global_it += 1

    if (state.local_it % print_every) == 0:
        msg = (
            f"BH {state.bh_it:3d} | "
            f"Epoch {state.local_it:4d} | "
            f"Total epochs {state.global_it:5d} | "
            f"Train loss {train_loss:.6e} | "
            f"Validation loss {val_loss:.6e}"
        )
        print(msg, end="\r", flush=True)

def callback_hopping_general(x, f, accept, *, state, qnn, has_validation=False, save_best=True):
    """
    Basinhopping callback: compares the last local-minimization metric with the best so far.
    If improved, stores best params and histories and optionally saves the model.

    :param x: Current parameters at the end of local minimization
    :param f: Current training loss f(x) (as reported by basinhopping)
    :param accept: Basinhopping accept flag
    """
    print()

    if len(state.current_train) == 0:
        print(f"Basinhopping iteration {state.bh_it}. Loss: {float(f):.6e}\n==========")
        state.bh_it += 1
        state.local_it = 0
        return

    current_metric = state._metric(has_validation)
    best_metric = state._best_metric(has_validation)

    if state.bh_it > 1:
        print(f"Best loss so far: {best_metric:.6e}")
    print(f"Basinhopping iteration {state.bh_it}. Loss: {float(f):.6e}\n==========")

    if current_metric < best_metric:
        state.best_pars = np.asarray(x).copy()
        state.best_train = state.current_train.copy()
        state.best_valid = state.current_valid.copy()

        if save_best:
            qnn.build_QNN(state.best_pars)
            qnn.save_model_parameters(state.best_pars)
            qnn.save_operator_matrices()
            qnn.save_model()

    state.bh_it += 1
    state.local_it = 0
    state.current_train = []
    state.current_valid = []

def _run_scipy_basinhopping(
    *,
    qnn,
    init_pars,
    bounds,
    hopping_iters,
    training_QNN,
    validate_QNN=None,
    save_best=True,
    print_every=1,
):
    """
    Runs SciPy basinhopping + L-BFGS-B with clean state tracking and non-verbose printing.
    Returns (best_pars, best_train_history, best_valid_history, train_time).
    """
    state = _BasinhoppingState(init_pars)
    cost_cache = _ObjectiveCache(training_QNN)

    callback = partial(
        callback_opt_general,
        state=state,
        cost_cache=cost_cache,
        cost_validation=validate_QNN,
        print_every=print_every,
    )
    callback_hop = partial(
        callback_hopping_general,
        state=state,
        qnn=qnn,
        has_validation=(validate_QNN is not None),
        save_best=save_best,
    )

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "callback": callback,
    }

    training_start = time.time()
    _ = opt.basinhopping(
        cost_cache,
        np.asarray(init_pars, dtype=np.float64),
        niter=hopping_iters,
        minimizer_kwargs=minimizer_kwargs,
        callback=callback_hop,
    )
    train_time = time.time() - training_start
    print()

    # If never improved, fall back to the last local history if available
    if np.isinf(state.best_train[-1]):
        state.best_pars = np.asarray(init_pars, dtype=np.float64)
        if len(state.current_train) > 0:
            state.best_train = state.current_train
            state.best_valid = state.current_valid
        else:
            state.best_train = [np.inf]
            state.best_valid = [np.inf]

    return state.best_pars, state.best_train, state.best_valid, train_time

# ---------------------------------------------------------------
# JAX training helpers (Adam/AdamW, Optax L-BFGS, simple GD)
# ---------------------------------------------------------------
def _run_adam_warmup(
    *,
    init_pars,
    training_QNN,
    validate_QNN=None,
    project=None,
    epochs=200,
    learning_rate=1e-2,
    weight_decay=0.0,
    print_every=1,
):
    """
    Pure-JAX Adam/AdamW warmup. Returns (best_params_np, train_hist, valid_hist, warmup_time).
    """
    if epochs <= 0:
        return np.asarray(init_pars, dtype=np.float64), [], [], 0.0

    params = jnp.asarray(init_pars, dtype=jnp.float64)
    if project is not None:
        params = project(params)

    if weight_decay != 0.0:
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(learning_rate=learning_rate)

    opt_state = optimizer.init(params)

    def loss_fn(p):
        if project is not None:
            p = project(p)
        return training_QNN(p)

    loss_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def step(p, s):
        v, g = loss_and_grad(p)
        upd, s = optimizer.update(g, s, p)
        p = optax.apply_updates(p, upd)
        if project is not None:
            p = project(p)
        return p, s, v

    train_hist = []
    valid_hist = []

    best_params = params
    best_train = float(loss_fn(params))
    if validate_QNN is not None:
        best_valid = float(validate_QNN(best_params))
    else:
        best_valid = best_train

    train_hist.append(best_train)
    valid_hist.append(best_valid)

    print(
        f"[Adam] Epoch {0}/{epochs} | train loss = {best_train:.6e} | val loss = {best_valid:.6e}",
        end="\r",
        flush=True,
    )
    t0 = time.time()

    for ep in range(1, epochs + 1):
        params, opt_state, loss_val = step(params, opt_state)
        train_loss = float(loss_val)

        if validate_QNN is not None:
            val_loss = float(validate_QNN(params))
        else:
            val_loss = train_loss

        train_hist.append(train_loss)
        valid_hist.append(val_loss)

        if (ep % print_every) == 0:
            print(
                f"[Adam] Epoch {ep}/{epochs} | train loss = {train_loss:.6e} | val loss = {val_loss:.6e}",
                end="\r",
                flush=True,
            )

        if val_loss < best_valid:
            best_valid = val_loss
            best_train = train_loss
            best_params = params

    warmup_time = time.time() - t0
    print()

    return np.asarray(best_params, dtype=np.float64), train_hist, valid_hist, warmup_time

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

# ---------------------------------------------------------------
# Trainer 1 (Recommended): SciPy basinhopping (L-BFGS-B) with JAX-wrapped model evaluation
# ---------------------------------------------------------------
def train_scipy(
    model_name, task_name,
    N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable,
    include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    train_set, valid_set,
    loss_function=mse, hopping_iters=2,
    in_preprocs=None, out_prepocs=None, postprocs=None,
    init_pars=None, save=True,
    print_every=1,
    save_best_each_basin=True,
):
    """
    Creates and trains a QNN model using SciPy L-BFGS-B + basinhopping.
    """

    # ---------------------------------------------------------------
    # 1. Number of parameters and bounds
    # ---------------------------------------------------------------
    assert layers == len(ladder_modes)

    n_pars = _compute_num_parameters(
        N, layers,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    )
    bounds = _build_bounds(N, layers, include_initial_squeezing, include_initial_mixing, is_passive_gaussian)
    assert len(bounds) == n_pars, f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."

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

    # ---------------------------------------------------------------
    # 3. Define training/validation objectives
    # ---------------------------------------------------------------
    train_inputs_jax = jnp.asarray(train_inputs)
    train_outputs_jax = jnp.asarray(train_outputs)

    training_QNN = partial(
        qnn.train_QNN,
        inputs_dataset=train_inputs_jax,
        outputs_dataset=train_outputs_jax,
        loss_function=loss_function,
    )

    if valid_inputs is not None:
        valid_inputs_jax = jnp.asarray(valid_inputs)
        valid_outputs_jax = jnp.asarray(valid_outputs)

        validate_QNN = partial(
            qnn.train_QNN,
            inputs_dataset=valid_inputs_jax,
            outputs_dataset=valid_outputs_jax,
            loss_function=loss_function,
        )
    else:
        validate_QNN = None

    # SciPy expects numpy params, but we keep dtype stable by converting inside
    def training_cost(x):
        return float(training_QNN(jnp.asarray(x, dtype=jnp.float64)))

    if validate_QNN is not None:
        def validation_cost(x):
            return float(validate_QNN(jnp.asarray(x, dtype=jnp.float64)))
    else:
        validation_cost = None

    # ---------------------------------------------------------------
    # 4. Basinhopping training
    # ---------------------------------------------------------------
    best_pars, best_train, best_valid, train_time = _run_scipy_basinhopping(
        qnn=qnn,
        init_pars=init_pars,
        bounds=bounds,
        hopping_iters=hopping_iters,
        training_QNN=training_cost,
        validate_QNN=validation_cost,
        save_best=(save and save_best_each_basin),
        print_every=print_every,
    )

    n_epochs = max(1, len(best_train))
    print(f"Total training time: {train_time:.3f} seconds")
    print(f"Time per epoch: {train_time / n_epochs:.6f} seconds")

    print(
        f"\nOPTIMIZATION ERROR FOR N={N}, L={layers}, "
        f"{_nongauss_symbol(is_addition)} modes={np.array(ladder_modes[0]) + 1}\n"
        f"{best_valid[-1] if valid_set is not None else best_train[-1]}"
    )

    # ---------------------------------------------------------------
    # 5. Build final QNN with best parameters
    # ---------------------------------------------------------------
    _finalize_model(qnn, best_pars, save=save)

    return qnn, best_train, best_valid

# ---------------------------------------------------------------
# Trainer 2 (Recommended): Hybrid (Pure JAX Adam warmup + SciPy basinhopping with JAX-wrapped model evaluation)
# ---------------------------------------------------------------
def train_hybrid_adamjax_scipy(
    model_name, task_name,
    N, layers, n_inputs, n_outputs,
    ladder_modes, is_addition, observable,
    include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    train_set, valid_set,
    loss_function=mse, hopping_iters=2,
    in_preprocs=None, out_prepocs=None, postprocs=None,
    init_pars=None, save=True,
    adam_epochs=200, adam_learning_rate=1e-2, adam_weight_decay=0.0,
    print_every=1,
    save_best_each_basin=True,
):
    """
    Hybrid training: Adam/AdamW warmup (pure JAX) then SciPy L-BFGS-B + basinhopping.
    """

    # ---------------------------------------------------------------
    # 1. Number of parameters and bounds
    # ---------------------------------------------------------------
    assert layers == len(ladder_modes)

    n_pars = _compute_num_parameters(
        N, layers,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
    )
    bounds = _build_bounds(N, layers, include_initial_squeezing, include_initial_mixing, is_passive_gaussian)
    assert len(bounds) == n_pars, f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."

    init_pars = _init_parameters(n_pars, init_pars=init_pars)

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

    # ---------------------------------------------------------------
    # 3. Define training/validation objectives (JAX)
    # ---------------------------------------------------------------
    train_inputs_jax = jnp.asarray(train_inputs)
    train_outputs_jax = jnp.asarray(train_outputs)

    training_QNN = partial(
        qnn.train_QNN,
        inputs_dataset=train_inputs_jax,
        outputs_dataset=train_outputs_jax,
        loss_function=loss_function,
    )

    if valid_inputs is not None:
        valid_inputs_jax = jnp.asarray(valid_inputs)
        valid_outputs_jax = jnp.asarray(valid_outputs)

        validate_QNN = partial(
            qnn.train_QNN,
            inputs_dataset=valid_inputs_jax,
            outputs_dataset=valid_outputs_jax,
            loss_function=loss_function,
        )
    else:
        validate_QNN = None

    # ---------------------------------------------------------------
    # 4. Phase 1: Adam/AdamW warmup
    # ---------------------------------------------------------------
    if adam_epochs > 0:
        print("\n=== Phase 1: Adam/AdamW warmup (JAX) ===")
        init_pars, _, _, adam_time = _run_adam_warmup(
            init_pars=init_pars,
            training_QNN=training_QNN,
            validate_QNN=validate_QNN,
            project=project,
            epochs=adam_epochs,
            learning_rate=adam_learning_rate,
            weight_decay=adam_weight_decay,
            print_every=max(1, print_every),
        )
        print(f"Adam warmup total time: {adam_time:.3f} s")
    else:
        print("\n[INFO] Skipping Adam warmup (adam_epochs <= 0)")

    # ---------------------------------------------------------------
    # 5. Phase 2: SciPy basinhopping + L-BFGS-B
    # ---------------------------------------------------------------
    def training_cost(x):
        return float(training_QNN(project(jnp.asarray(x, dtype=jnp.float64))))

    if validate_QNN is not None:
        def validation_cost(x):
            return float(validate_QNN(project(jnp.asarray(x, dtype=jnp.float64))))
    else:
        validation_cost = None

    print("\n=== Phase 2: Basinhopping with L-BFGS-B (SciPy) ===")
    best_pars, best_train, best_valid, train_time = _run_scipy_basinhopping(
        qnn=qnn,
        init_pars=init_pars,
        bounds=bounds,
        hopping_iters=hopping_iters,
        training_QNN=training_cost,
        validate_QNN=validation_cost,
        save_best=(save and save_best_each_basin),
        print_every=print_every,
    )

    n_epochs = max(1, len(best_train))
    print(f"Total training time: {train_time:.3f} seconds")
    print(f"Time per epoch: {train_time / n_epochs:.6f} seconds")

    print(
        f"\nOPTIMIZATION ERROR FOR N={N}, L={layers}, "
        f"{_nongauss_symbol(is_addition)} modes={np.array(ladder_modes[0]) + 1}\n"
        f"{best_valid[-1] if valid_set is not None else best_train[-1]}"
    )

    # ---------------------------------------------------------------
    # 6. Build final QNN with best parameters
    # ---------------------------------------------------------------
    _finalize_model(qnn, best_pars, save=save)

    return qnn, best_train, best_valid

# ---------------------------------------------------------------
# Trainer 3: Pure JAX (gradient descent)
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Trainer 4: Optax L-BFGS (pure JAX) - TODO: Fix excessive slow optimization
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Trainer 5: Adam/AdamW (pure JAX) - TODO: Check loss jumps and final performance, maybe add learning rate decay or early stopping
# ---------------------------------------------------------------
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
