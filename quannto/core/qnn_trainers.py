import numpy as np
from functools import reduce, partial
import time
import scipy.optimize as opt
import optax
import jax
import jax.numpy as jnp

from .loss_functions import mse
from .qnn import QNN

global global_it
global local_it
global bh_it

# ---------------------------------------------------------------
# Callbacks for SciPy optimizer and Basinhopping
# ---------------------------------------------------------------
def callback_opt_general(xk, cost_training, cost_validation):
    '''
    Prints and stores the MSE error value for each QNN training epoch.
    
    :param xk: QONN tunable parameters
    '''
    e = cost_training(xk)
    loss_values.append(e)
    #print(f'Training loss: {e}')
    k = local_it["k"]
    local_it["k"] += 1
    
    g_k = global_it["k"]
    global_it["k"] += 1

    if cost_validation != None:
        val_e = cost_validation(xk)
        #print(f'Validation loss: {val_e}')
        msg = f" Epoch {k:4d} | Total epochs {g_k:4d} | Train loss {loss_values[-1]:.6e} | Validation loss {validation_loss[-1]:.6e}"
        validation_loss.append(val_e)
    else:
        msg = f"Epoch {k:4d} | Total epochs {g_k:4d} | | Train loss {loss_values[-1]:.6e}"
        validation_loss.append(e)
    print(msg, end="\r", flush=True)
    
def callback_hopping_general(x, f, accept, qnn, has_validation=False):
    '''
    Prints the best Basinhopping error achieved so far and the current one. 
    If the current error is less than the best one so far, it substitutes the best loss values.
    
    :param x: QONN tunable parameters
    :param f: Current Basinhopping error achieved
    :param accept: Determines whether the current error achieved is accepted (stopping iterations)
    '''
    global best_loss_values
    global loss_values
    global best_validation_loss
    global validation_loss
    global best_pars
    print()
    if bh_it["k"] > 1:
        print(f"Best loss so far: {best_loss_values[-1]}")
    print(f"Basinhopping iteration {bh_it['k']}. Loss: {f}\n==========")
    bh_it["k"] += 1
    local_it["k"] = 0
    if has_validation:
        if best_validation_loss[-1] > validation_loss[-1]:
            best_pars = x.copy()
            best_loss_values = loss_values.copy()
            best_validation_loss = validation_loss.copy()
            qnn.build_QNN(best_pars)
            qnn.save_model_parameters(best_pars)
            qnn.save_operator_matrices()
            qnn.save_model()
    else:
        if best_loss_values[-1] > loss_values[-1]:
            best_pars = x.copy()
            best_loss_values = loss_values.copy()
            qnn.build_QNN(best_pars)
            qnn.save_model_parameters(best_pars)
            qnn.save_operator_matrices()
            qnn.save_model()
    loss_values = [9999]
    validation_loss = [9999]

def build_and_train_model(model_name, task_name, N, layers, n_inputs, n_outputs,
                          ladder_modes, is_addition, observable, 
                          include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                          train_set, valid_set, loss_function=mse, hopping_iters=2,
                          in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    '''
    Creates and trains a QNN model with the given hyperparameters and dataset by optimizing the 
    tunable parameters of the QNN using L-BFGS-B + Basinhopping from SciPy which uses a 
    JAX-jitted function for the training.
    
    :param name: Name of the model
    :param N: Number of neurons per layer (modes of the quantum system)
    :param layers: Number of layers
    :param n_inputs: Number of QONN inputs
    :param n_outputs: Number of QONN outputs
    :param ladder_modes: Photon additions over system modes' that are desired (per layer)
    :param is_addition: Boolean determining whether the non-Gaussian operation is photon addition (True) or cubic phase gate (False)
    :param observable: Name of the observable to be measured ('position', 'momentum', 'number' or 'witness')
    :param include_initial_squeezing: Boolean determining whether to include initial squeezing operations before the first layer
    :param include_initial_mixing: Boolean determining whether to include initial mixing operation before the first layer
    :param is_passive_gaussian: Boolean determining whether the QNN is passive Gaussian (no squeezing operations in the layers)
    :param train_set: List of inputs and outputs to be learned
    :param valid_set: Validation set to be evaluated every epoch testing the generalization of the QONN
    :param loss_function: Function that computes the loss between the predicted and the expected value (default 'mse')
    :param hopping_iters: Number of Basinhopping iterations
    :param in_preprocs: List of preprocessors for the dataset inputs
    :param out_preprocs: List of preprocessors for the dataset outputs
    :param postprocs: List of postprocessors (postprocessing of QONN yielded data)
    :param init_pars: Initialization parameters for the QNN
    :param save: Boolean determining whether to save the model (default=True)
    :return: Trained QNN model
    '''
    # ---------------------------------------------------------------
    # 1. Number of parameters and bounds
    # ---------------------------------------------------------------
    assert layers == len(ladder_modes)
    
    n_pars = layers*(2*N**2 + 3*N)
    if include_initial_squeezing:
        n_pars += N
    if include_initial_mixing:
        n_pars += N**2
    if is_passive_gaussian:
        n_pars -= layers*(N**2 + N)
    
    if type(init_pars) == type(None):
        init_pars = np.random.rand(n_pars) # TODO: Consider parameterize momentum too
    else:
        assert len(init_pars) == n_pars
    
    passive_bounds = (None, None)
    init_sqz_bounds = (-np.log(5.5), np.log(5.5)) # Initial squeezing
    sqz_bounds = (-np.log(5.5), np.log(5.5))
    disp_bounds = (None, None)
    bounds = []
    if include_initial_squeezing:
        bounds += [init_sqz_bounds for _ in range(N)] # Initial squeezing
    if include_initial_mixing:
        bounds += [passive_bounds for _ in range(N**2)] # Initial mixing
    for _ in range(layers):
        if is_passive_gaussian:
            # Passive optics bounds for Q1
            bounds += [passive_bounds for _ in range(N**2)]
            # Displacement bounds
            bounds += [disp_bounds for _ in range(2*N)]
        else:
            # Passive optics bounds for Q1, Q2
            bounds += [passive_bounds for _ in range(2*N**2)]
            # Squeezing bounds
            bounds += [sqz_bounds for _ in range(N)]
            # Displacement bounds
            bounds += [disp_bounds for _ in range(2*N)]
    
    assert len(bounds) == n_pars, f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."
    
    # ---------------------------------------------------------------
    # 3. Build QNN, preprocess dataset and define training functions
    # ---------------------------------------------------------------
    qnn = QNN(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
              include_initial_squeezing, include_initial_mixing, is_passive_gaussian, init_pars,
              in_preprocs, out_prepocs, postprocs)
    
    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])
    if valid_set != None:
        valid_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, valid_set[0])
        valid_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, valid_set[1])
    
    global best_loss_values
    best_loss_values = [9999]
    
    global loss_values
    loss_values = [9999]
    
    global best_validation_loss
    best_validation_loss = [9999]
    
    global validation_loss
    validation_loss = [9999]
    
    global best_pars
    best_pars = []
    
    global global_it
    global local_it
    global bh_it
    global_it = {"k": 0}
    local_it = {"k": 0}
    bh_it = {"k": 1}
    
    training_QNN = partial(qnn.train_QNN, inputs_dataset=train_inputs, outputs_dataset=train_outputs, loss_function=loss_function)
    if valid_set != None:
        validate_QNN = partial(qnn.train_QNN, inputs_dataset=valid_inputs, outputs_dataset=valid_outputs, loss_function=loss_function)
    else:
        validate_QNN = None
    
    # ---------------------------------------------------------------
    # 4. Training with SciPy L-BFGS-B + Basinhopping
    # ---------------------------------------------------------------
    callback = partial(callback_opt_general, cost_training=training_QNN, cost_validation=validate_QNN)
    callback_hopping = partial(callback_hopping_general, qnn=qnn, has_validation=False if validate_QNN == None else True)
    
    training_start = time.time()
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "callback": callback}
    opt_result = opt.basinhopping(training_QNN, init_pars, niter=hopping_iters, minimizer_kwargs=minimizer_kwargs, callback=callback_hopping)
    train_time = time.time() - training_start
    print(f'Total training time: {train_time} seconds')
    print(f'Time per epoch: {train_time / (len(best_loss_values)-1)} seconds')
    
    if is_addition:
        nongauss_op = "â†"
    else:
        nongauss_op = "â"
    print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}, {nongauss_op} modes={np.array(ladder_modes[0])+1}\n{best_loss_values[-1]}')
    
    # Build final QNN with best Basinhopping parameters
    qnn.build_QNN(best_pars)
    qnn.print_qnn()
    
    if save:
        qnn.save_model()
        
    if best_loss_values[0] == 9999:
        best_loss_values = best_loss_values[1:]
        best_validation_loss = best_validation_loss[1:]
    
    return qnn, best_loss_values, best_validation_loss

def hybrid_build_and_train_model(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                 include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                 train_set, valid_set, loss_function=mse, hopping_iters=2,
                                 in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True,
                                 adam_epochs=200, adam_learning_rate=1e-2, adam_weight_decay=0.0):
    """
    Creates and trains a QNN model with the given hyperparameters and dataset
    by optimizing the tunable parameters of the QNN using a hybrid approach:
    first using Adam/AdamW (pure JAX) for a number of epochs, and then
    refining the solution with SciPy's L-BFGS-B.
    
    :param model_name: Name of the QNN model
    :param N: Number of modes
    :param layers: Number of layers in the QNN
    :param n_inputs: Number of input features
    :param n_outputs: Number of output features
    :param ladder_modes: Ladder modes for non-Gaussian operations
    :param is_addition: Whether to use addition non-Gaussian operation
    :param observable: Observable to measure
    :param include_initial_squeezing: Whether to include initial squeezing layer
    :param include_initial_mixing: Whether to include initial mixing layer
    :param is_passive_gaussian: Whether the Gaussian layers are passive
    :param train_set: Training dataset (inputs, outputs)
    :param valid_set: Validation dataset (inputs, outputs)
    :param loss_function: Loss function to optimize
    :param hopping_iters: Number of Basinhopping iterations (SciPy)
    :param in_preprocs: Input preprocessors
    :param out_prepocs: Output preprocessors
    :param postprocs: Postprocessors
    :param init_pars: Initial parameters for optimization
    :param save: Whether to save the trained model
    :param adam_epochs: Number of Adam epochs for initial training
    :param adam_learning_rate: Learning rate for Adam optimizer
    :param adam_weight_decay: Weight decay for AdamW optimizer
    :return: Trained QNN model, training loss history, validation loss history
    """

    # ---------------------------------------------------------------
    # 1. Number of parameters and bounds
    # ---------------------------------------------------------------
    assert layers == len(ladder_modes)
    n_pars = layers * (2 * N**2 + 3 * N)

    if include_initial_squeezing:
        n_pars += N
    if include_initial_mixing:
        n_pars += N**2
    if is_passive_gaussian:
        n_pars -= layers * (N**2 + N)

    if isinstance(init_pars, type(None)):
        init_pars = np.random.rand(n_pars)
    else:
        init_pars = np.asarray(init_pars)
        assert len(init_pars) == n_pars

    passive_bounds = (None, None)
    init_sqz_bounds = (-np.log(5.5), np.log(5.5))  # Initial squeezing
    sqz_bounds = (-np.log(5.5), np.log(5.5))       # Layer squeezing
    disp_bounds = (None, None)

    bounds = []

    # Initial squeezing
    if include_initial_squeezing:
        bounds += [init_sqz_bounds for _ in range(N)]

    # Initial mixing
    if include_initial_mixing:
        bounds += [passive_bounds for _ in range(N**2)]

    # Layer-by-layer
    for _ in range(layers):
        if is_passive_gaussian:
            # Passive optics bounds for Q1
            bounds += [passive_bounds for _ in range(N**2)]
            # Displacement bounds
            bounds += [disp_bounds for _ in range(2 * N)]
        else:
            # Passive optics bounds for Q1, Q2
            bounds += [passive_bounds for _ in range(2 * N**2)]
            # Squeezing bounds
            bounds += [sqz_bounds for _ in range(N)]
            # Displacement bounds
            bounds += [disp_bounds for _ in range(2 * N)]

    assert len(bounds) == n_pars, (
        f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."
    )
    
    # Set up bounds in JAX arrays for projection
    lb = jnp.array(
        [b[0] if b[0] is not None else -jnp.inf for b in bounds],
        dtype=jnp.float64,
    )
    ub = jnp.array(
        [b[1] if b[1] is not None else jnp.inf for b in bounds],
        dtype=jnp.float64,
    )

    def project(params_jax):
        """
        Projects parameters into the bounds (like L-BFGS-B in SciPy).
        :param params: QNN tunable parameters
        :return: Projected parameters
        """
        return jnp.clip(params_jax, lb, ub)


    # ---------------------------------------------------------------
    # 3. Build QNN, preprocess dataset and define training functions
    # ---------------------------------------------------------------
    qnn = QNN(
        model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian, init_pars,
        in_preprocs, out_prepocs, postprocs,
    )
    
    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])

    if valid_set is not None:
        valid_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, valid_set[0])
        valid_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, valid_set[1])
    else:
        valid_inputs = None
        valid_outputs = None

    global best_loss_values
    best_loss_values = [9999]

    global loss_values
    loss_values = [9999]

    global best_validation_loss
    best_validation_loss = [9999]

    global validation_loss
    validation_loss = [9999]

    global best_pars
    best_pars = []
    
    global global_it
    global local_it
    global bh_it
    global_it = {"k": 0}
    local_it = {"k": 0}
    bh_it = {"k": 1}

    training_QNN = partial(
        qnn.train_QNN,
        inputs_dataset=train_inputs,
        outputs_dataset=train_outputs,
        loss_function=loss_function,
    )

    if valid_set is not None:
        validate_QNN = partial(
            qnn.train_QNN,
            inputs_dataset=valid_inputs,
            outputs_dataset=valid_outputs,
            loss_function=loss_function,
        )
    else:
        validate_QNN = None

    # ---------------------------------------------------------------
    # 5. Phase 1: Adam/AdamW warmup (pure JAX)
    # ---------------------------------------------------------------
    if adam_epochs > 0:
        print("\n=== Phase 1: Adam/AdamW warmup (JAX) ===")

        params = project(jnp.array(init_pars, dtype=jnp.float64))

        # Choose Adam or AdamW
        if adam_weight_decay != 0.0:
            optimizer = optax.adamw(
                learning_rate=adam_learning_rate,
                weight_decay=adam_weight_decay,
            )
        else:
            optimizer = optax.adam(learning_rate=adam_learning_rate)

        opt_state = optimizer.init(params)

        def loss_fn_jax(p):
            p_proj = project(p)
            return training_QNN(p_proj)

        loss_and_grad = jax.value_and_grad(loss_fn_jax)

        @jax.jit
        def adam_step(p, state):
            loss_val, grads = loss_and_grad(p)
            updates, state = optimizer.update(grads, state, p)
            p = optax.apply_updates(p, updates)
            p = project(p)
            return p, state, loss_val
        
        # Initial evaluation
        adam_best_params = params
        adam_best_loss = float(loss_fn_jax(params))

        if validate_QNN is not None:
            adam_best_val_loss = float(validate_QNN(np.array(adam_best_params)))
        else:
            adam_best_val_loss = adam_best_loss

        print(
            f"[Adam] Epoch 0/{adam_epochs} | "
            f"train loss = {adam_best_loss:.6g} | "
            f"val loss = {adam_best_val_loss:.6g}",
            end="\r",
            flush=True,
        )
        adam_start = time.time()

        for epoch in range(1, adam_epochs + 1):
            params, opt_state, loss_val = adam_step(params, opt_state)
            train_loss = float(loss_val)

            if validate_QNN is not None:
                val_loss = float(validate_QNN(np.array(params)))
            else:
                val_loss = train_loss

            print(
                f"[Adam] Epoch {epoch}/{adam_epochs} | "
                f"train loss = {train_loss:.6g} | "
                f"val loss = {val_loss:.6g}",
                end="\r",
                flush=True,
            )

            if val_loss < adam_best_val_loss:
                adam_best_val_loss = val_loss
                adam_best_loss = train_loss
                adam_best_params = params

        adam_time = time.time() - adam_start
        print(f"Adam warmup total time: {adam_time:.3f} s")
        print(f"Adam warmup time per epoch: {adam_time / adam_epochs:.6f} s")
        print(
            f"Best Adam train loss = {adam_best_loss:.6g}, "
            f"best Adam val loss = {adam_best_val_loss:.6g}"
        )

        init_pars = np.asarray(adam_best_params, dtype=np.float64)

    else:
        print("\n[INFO] Skipping Adam warmup (adam_epochs <= 0)")

    # ---------------------------------------------------------------
    # 6. Phase 2: Basinhopping with L-BFGS-B (SciPy)
    # ---------------------------------------------------------------
    callback = partial(callback_opt_general, cost_training=training_QNN, cost_validation=validate_QNN)
    callback_hopping = partial(callback_hopping_general, qnn=qnn, has_validation=False if validate_QNN == None else True)
    
    training_start = time.time()

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "callback": callback,
    }

    print("\n=== Phase 2: Basinhopping with L-BFGS-B (SciPy) ===")
    opt_result = opt.basinhopping(
        training_QNN,
        init_pars,
        niter=hopping_iters,
        minimizer_kwargs=minimizer_kwargs,
        callback=callback_hopping,
    )

    train_time = time.time() - training_start
    print(f"Total training time: {train_time} seconds")
    print(f"Time per epoch: {train_time / (len(best_loss_values) - 1)} seconds")

    if is_addition:
        nongauss_op = "â†"
    else:
        nongauss_op = "â"

    print(
        f"\nOPTIMIZATION ERROR FOR N={N}, L={layers}, "
        f"{nongauss_op} modes={np.array(ladder_modes[0]) + 1}\n"
        f"{best_loss_values[-1]}"
    )

    # Build final QNN with best Basinhopping parameters
    qnn.build_QNN(best_pars)
    qnn.print_qnn()

    if save:
        qnn.save_model()

    if best_loss_values[0] == 9999:
        best_loss_values = best_loss_values[1:]
        best_validation_loss = best_validation_loss[1:]

    return qnn, best_loss_values, best_validation_loss

def jax_gd_build_and_train_model(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                 include_initial_squeezing, include_initial_mixing, is_passive_gaussian, 
                                 train_set, valid_set, loss_function=mse, hopping_iters=200,
                                 in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True, learning_rate=5e-3):
    """
    Creates and trains a QNN model with the given hyperparameters and dataset
    by optimizing the tunable parameters of the QNN using pure JAX
    (gradient descent) instead of SciPy.

    hopping_iters: number of epochs
    learning_rate: step size for gradient descent
    """
    assert layers == len(ladder_modes)

    # =========================
    # 1. Number of tunable parameters
    # =========================
    n_pars = layers * (2 * N**2 + 3 * N)
    if include_initial_squeezing:
        n_pars += N
    if include_initial_mixing:
        n_pars += N**2
    if is_passive_gaussian:
        n_pars -= layers * (N**2 + N)

    if init_pars is None:
        init_pars = np.random.rand(n_pars)
    else:
        assert len(init_pars) == n_pars

    # =========================
    # 2. Create QNN and preprocess dataset
    # =========================
    qnn = QNN(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian, init_pars,
        in_preprocs, out_prepocs, postprocs
    )

    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])

    if valid_set is not None:
        valid_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, valid_set[0])
        valid_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, valid_set[1])
    else:
        valid_inputs = None
        valid_outputs = None

    train_inputs = jax.device_put(jnp.array(train_inputs))
    train_outputs = jax.device_put(jnp.array(train_outputs))

    if valid_inputs is not None:
        valid_inputs = jax.device_put(jnp.array(valid_inputs))
        valid_outputs = jax.device_put(jnp.array(valid_outputs))

    # =========================
    # 3. Initialize JAX parameters
    # =========================
    params = jax.device_put(jnp.array(init_pars))

    # =========================
    # 4. Define loss and gradient functions purely in JAX
    # =========================
    def full_loss_fn(p):
        """
        Full-JAX loss function
        :param p: QNN tunable parameters
        :return: Loss value
        devuelve: escalar jnp (loss sobre todo el train_set)
        """
        return qnn.train_QNN(
            p,
            inputs_dataset=train_inputs,
            outputs_dataset=train_outputs,
            loss_function=loss_function,
        )

    # value_and_grad with respect to parameters
    loss_and_grad = jax.jit(jax.value_and_grad(full_loss_fn))

    # =========================
    # 5. Training (pure JAX + Python loop)
    # =========================
    loss_values = []
    validation_loss = []
    best_loss = jnp.inf
    best_pars = params

    training_start = time.time()

    for epoch in range(hopping_iters):
        # 5.1. Forward + backward
        loss, grads = loss_and_grad(params)

        loss_values.append(float(loss))

        # 5.2. Update parameters (simple GD)
        params = params - learning_rate * grads

        # 5.3. Validation (if any)
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
            val_loss = float(loss)

        validation_loss.append(val_loss)

        # 5.4. Save best parameters
        if val_loss < best_loss:
            best_loss = val_loss
            best_pars = params

        print(
            f"Epoch {epoch+1}/{hopping_iters} "
            f" | train loss = {float(loss):.6g}"
            f" | val loss = {val_loss:.6g}"
        )

    train_time = time.time() - training_start
    print(f"Total training time: {train_time:.3f} seconds")
    print(f"Time per epoch: {train_time / hopping_iters:.3f} seconds")

    # =========================
    # 6. Build final QNN with best parameters
    # =========================
    if is_addition:
        nongauss_op = "â†"
    else:
        nongauss_op = "â"

    print(
        f"\nFINAL ERROR FOR N={N}, L={layers}, {nongauss_op} "
        f"modes={np.array(ladder_modes[0]) + 1}\n{best_loss}"
    )

    # Construye la QNN con los mejores parámetros encontrados
    qnn.build_QNN(best_pars)
    qnn.tunable_parameters = best_pars
    qnn.print_qnn()

    if save:
        qnn.save_model()

    return qnn, loss_values, validation_loss


def optax_build_and_train_model(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                                include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                                train_set, valid_set, loss_function=mse, hopping_iters=200,
                                in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    """
    Creates and trains a QNN model with the given hyperparameters and dataset by
    optimizing the tunable parameters of the QNN using Optax L-BFGS.
    """
    assert layers == len(ladder_modes)

    # ----------------------------------------------------------------------
    # 1. Number of parameters and bounds
    # ----------------------------------------------------------------------
    n_pars = layers * (2 * N**2 + 3 * N)
    if include_initial_squeezing:
        n_pars += N
    if include_initial_mixing:
        n_pars += N**2
    if is_passive_gaussian:
        n_pars -= layers * (N**2 + N)

    if init_pars is None:
        init_pars = np.random.rand(n_pars).astype(np.float64)
    else:
        init_pars = np.asarray(init_pars, dtype=np.float64)
        assert init_pars.shape == (n_pars,)

    # Bounds like L-BFGS-B from SciPy
    passive_bounds = (None, None)
    init_sqz_bounds = (-np.log(5.5), np.log(5.5))  # Initial squeezing
    sqz_bounds = (-np.log(5.5), np.log(5.5))       # Layer squeezing
    disp_bounds = (None, None)

    bounds = []
    # Initial squeezing
    if include_initial_squeezing:
        bounds += [init_sqz_bounds for _ in range(N)]
    # Initial mixing
    if include_initial_mixing:
        bounds += [passive_bounds for _ in range(N**2)]

    # Layer-by-layer
    for _ in range(layers):
        if is_passive_gaussian:
            # Passive optics bounds for Q1
            bounds += [passive_bounds for _ in range(N**2)]
            # Displacement
            bounds += [disp_bounds for _ in range(2 * N)]
        else:
            # Passive optics bounds for Q1, Q2
            bounds += [passive_bounds for _ in range(2 * N**2)]
            # Squeezing
            bounds += [sqz_bounds for _ in range(N)]
            # Displacement
            bounds += [disp_bounds for _ in range(2 * N)]

    assert len(bounds) == n_pars, (
        f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."
    )

    # Set up bounds in JAX arrays for projection
    lb = jnp.array(
        [b[0] if b[0] is not None else -jnp.inf for b in bounds],
        dtype=jnp.float64,
    )
    ub = jnp.array(
        [b[1] if b[1] is not None else jnp.inf for b in bounds],
        dtype=jnp.float64,
    )

    def project(params):
        """
        Projects parameters into the bounds (like L-BFGS-B in SciPy).
        :param params: QNN tunable parameters
        :return: Projected parameters
        """
        return jnp.clip(params, lb, ub)

    # ----------------------------------------------------------------------
    # 2. Create QNN and preprocess dataset
    # ----------------------------------------------------------------------
    qnn = QNN(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian, init_pars,
        in_preprocs, out_prepocs, postprocs,
    )

    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])

    if valid_set is not None:
        valid_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, valid_set[0])
        valid_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, valid_set[1])
    else:
        valid_inputs = None
        valid_outputs = None

    train_inputs = jnp.asarray(train_inputs)
    train_outputs = jnp.asarray(train_outputs)
    if valid_inputs is not None:
        valid_inputs = jnp.asarray(valid_inputs)
        valid_outputs = jnp.asarray(valid_outputs)

    # ----------------------------------------------------------------------
    # 3. Loss functions (JAX) with projected parameters
    # ----------------------------------------------------------------------

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
        """Loss over the training dataset (JAX, differentiable).
        :param params: QNN tunable parameters
        :return: Loss value
        """
        params_proj = project(params)
        return training_QNN(params_proj)

    loss_fn_jit = jax.jit(loss_fn)

    if validate_QNN is not None:
        def val_loss_fn(params):
            params_proj = project(params)
            return validate_QNN(params_proj)

        val_loss_fn_jit = jax.jit(val_loss_fn)
    else:
        val_loss_fn_jit = None

    # ----------------------------------------------------------------------
    # 4. L-BFGS training loop (Optax)
    # ----------------------------------------------------------------------
    params = jnp.asarray(init_pars, dtype=jnp.float64)
    params = project(params)

    opt = optax.lbfgs()
    opt_state = opt.init(params)

    value_and_grad = optax.value_and_grad_from_state(loss_fn_jit)

    max_iter = hopping_iters
    tol = 1e-6

    loss_values = []
    validation_loss = []

    # Initial evaluation
    train_loss_0 = float(loss_fn_jit(params))
    if val_loss_fn_jit is not None:
        valid_loss_0 = float(val_loss_fn_jit(params))
    else:
        valid_loss_0 = train_loss_0

    loss_values.append(train_loss_0)
    validation_loss.append(valid_loss_0)

    best_params = params
    best_train_loss = train_loss_0
    best_valid_loss = valid_loss_0

    training_start = time.time()

    for i in range(max_iter):
        value, grad = value_and_grad(params, state=opt_state)

        updates, opt_state = opt.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=loss_fn_jit,
        )
        params = optax.apply_updates(params, updates)
        params = project(params)

        train_loss_i = float(value)
        loss_values.append(train_loss_i)

        if val_loss_fn_jit is not None:
            valid_loss_i = float(val_loss_fn_jit(params))
        else:
            valid_loss_i = train_loss_i

        validation_loss.append(valid_loss_i)

        print(
            f"Epoch {i+1}/{max_iter} "
            f" | train loss = {float(train_loss_i):.6g}"
            f" | val loss = {valid_loss_i:.6g}"
        )
        if valid_loss_i < best_valid_loss:
            best_valid_loss = valid_loss_i
            best_train_loss = train_loss_i
            best_params = params

        grad_norm = float(jnp.linalg.norm(grad))
        if grad_norm < tol:
            print(f"Early stopping at iter {i+1}, ||grad|| = {grad_norm:.3e}")
            break

    train_time = time.time() - training_start
    n_epochs = max(1, len(loss_values) - 1)
    print(f"Total training time: {train_time:.3f} seconds")
    print(f"Time per epoch: {train_time / n_epochs:.6f} seconds")

    if is_addition:
        nongauss_op = "â†"
    else:
        nongauss_op = "â"
    print(
        f"\nOPTIMIZATION ERROR FOR N={N}, L={layers}, "
        f"{nongauss_op} modes={np.array(ladder_modes[0]) + 1}\n"
        f"{best_train_loss}"
    )

    # ----------------------------------------------------------------------
    # 5. Build final QNN with best parameters
    # ----------------------------------------------------------------------
    qnn.build_QNN(np.asarray(best_params, dtype=np.float64))
    qnn.print_qnn()

    if save:
        qnn.save_model()

    return qnn, loss_values, validation_loss


def adam_build_and_train_model(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
                               include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
                               train_set, valid_set, loss_function=mse, hopping_iters=200, 
                               in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True,
                               learning_rate=1e-2, weight_decay=0.0):
    """
    Creates and trains a QNN model with the given hyperparameters and dataset
    by optimizing the tunable parameters of the QNN using a pure-JAX optimizer
    (Adam/AdamW via Optax).
    """
    assert layers == len(ladder_modes)

    # ------------------------------------------------------------------
    # 1. Number of parameters and bounds
    # ------------------------------------------------------------------
    n_pars = layers * (2 * N**2 + 3 * N)
    if include_initial_squeezing:
        n_pars += N
    if include_initial_mixing:
        n_pars += N**2
    if is_passive_gaussian:
        n_pars -= layers * (N**2 + N)

    if init_pars is None:
        init_pars = np.random.rand(n_pars)
    else:
        init_pars = np.asarray(init_pars)
        assert init_pars.shape == (n_pars,)

    passive_bounds = (None, None)
    init_sqz_bounds = (-np.log(5.5), np.log(5.5))  # Initial squeezing
    sqz_bounds = (-np.log(5.5), np.log(5.5))       # Layer squeezing
    disp_bounds = (None, None)

    bounds = []
    # Initial squeezing
    if include_initial_squeezing:
        bounds += [init_sqz_bounds for _ in range(N)]
    # Initial mixing
    if include_initial_mixing:
        bounds += [passive_bounds for _ in range(N**2)]

    # Layer-by-layer bounds
    for _ in range(layers):
        if is_passive_gaussian:
            # Passive optics bounds for Q1
            bounds += [passive_bounds for _ in range(N**2)]
            # Displacement bounds
            bounds += [disp_bounds for _ in range(2 * N)]
        else:
            # Passive optics bounds for Q1, Q2
            bounds += [passive_bounds for _ in range(2 * N**2)]
            # Squeezing bounds
            bounds += [sqz_bounds for _ in range(N)]
            # Displacement bounds
            bounds += [disp_bounds for _ in range(2 * N)]

    assert len(bounds) == n_pars, (
        f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."
    )

    # Convert bounds to JAX arrays for projection
    lb = jnp.array(
        [b[0] if b[0] is not None else -jnp.inf for b in bounds],
        dtype=jnp.float64,
    )
    ub = jnp.array(
        [b[1] if b[1] is not None else jnp.inf for b in bounds],
        dtype=jnp.float64,
    )

    def project(params):
        """Proyección simple a la caja [lb, ub]."""
        return jnp.clip(params, lb, ub)

    # ------------------------------------------------------------------
    # 2. Build QNN and preprocess dataset
    # ------------------------------------------------------------------
    qnn = QNN(model_name, task_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
        include_initial_squeezing, include_initial_mixing, is_passive_gaussian, init_pars, 
        in_preprocs, out_prepocs, postprocs,
    )

    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])

    if valid_set is not None:
        valid_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, valid_set[0])
        valid_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, valid_set[1])
    else:
        valid_inputs = None
        valid_outputs = None

    train_inputs = jax.device_put(jnp.array(train_inputs))
    train_outputs = jax.device_put(jnp.array(train_outputs))

    if valid_inputs is not None:
        valid_inputs = jax.device_put(jnp.array(valid_inputs))
        valid_outputs = jax.device_put(jnp.array(valid_outputs))

    # ------------------------------------------------------------------
    # 3. Define loss functions (JAX) with projected parameters
    # ------------------------------------------------------------------

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
        params_proj = project(params)
        return training_QNN(params_proj)

    # ------------------------------------------------------------------
    # 4. Configure Adam/AdamW optimizer (Optax)
    # ------------------------------------------------------------------
    params = jnp.array(init_pars, dtype=jnp.float64)
    params = project(params)

    # AdamW with decoupled weight decay. For plain Adam, set weight_decay=0.0
    if weight_decay != 0.0:
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(learning_rate=learning_rate)

    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        """
        Single training step (forward + backward + update).
        :param params: Current QNN tunable parameters
        :param opt_state: Current optimizer state
        :return: Updated parameters, updated optimizer state, loss value
        """
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = project(params)
        return params, opt_state, loss

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    loss_values = []
    validation_loss = []

    best_params = params
    best_train_loss = jnp.inf
    best_valid_loss = jnp.inf

    training_start = time.time()

    # Initial evaluation
    init_train_loss = float(loss_fn(params))
    if validate_QNN is not None:
        init_valid_loss = float(validate_QNN(project(params)))
    else:
        init_valid_loss = init_train_loss

    loss_values.append(init_train_loss)
    validation_loss.append(init_valid_loss)
    best_train_loss = init_train_loss
    best_valid_loss = init_valid_loss
    best_params = params

    epoch = 0
    max_iter = 10000
    target_loss = 1e-3
    val_loss = 9999
    #for epoch in range(hopping_iters):
    while val_loss > target_loss and epoch < max_iter - 1:
        epoch += 1
        params, opt_state, loss = train_step(params, opt_state)
        train_loss = float(loss)
        loss_values.append(train_loss)

        if validate_QNN is not None:
            val_loss = float(validate_QNN(project(params)))
        else:
            val_loss = train_loss

        validation_loss.append(val_loss)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_train_loss = train_loss
            best_params = params

        print(
            f"Epoch {epoch+1}/{hopping_iters} "
            f"| train loss = {train_loss:.6g} "
            f"| val loss = {val_loss:.6g}"
        )

    train_time = time.time() - training_start
    effective_epochs = max(1, len(loss_values) - 1)
    print(f"Total training time: {train_time} seconds")
    print(f"Time per epoch: {train_time / effective_epochs} seconds")

    # ------------------------------------------------------------------
    # 6. Build final QNN with best parameters
    # ------------------------------------------------------------------
    if is_addition:
        nongauss_op = "â†"
    else:
        nongauss_op = "â"

    print(
        f"\nFINAL ERROR FOR N={N}, L={layers}, {nongauss_op} "
        f"modes={np.array(ladder_modes[0]) + 1}\n{best_train_loss}"
    )

    best_pars_np = np.asarray(best_params, dtype=np.float64)
    qnn.build_QNN(best_pars_np)
    qnn.print_qnn()

    if save:
        qnn.save_model()

    return qnn, loss_values, validation_loss
