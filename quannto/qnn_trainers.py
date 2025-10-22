import numpy as np
from functools import reduce, partial
import time
import scipy.optimize as opt
import jax.numpy as jnp

from .loss_functions import mse
from .qnn import QNN


def build_and_train_model(model_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable, include_initial_squeezing, include_initial_mixing,
                          is_passive_gaussian, train_set, valid_set, loss_function=mse, hopping_iters=2, in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    '''
    Creates and trains a QNN model with the given hyperparameters and dataset by optimizing the 
    tunable parameters of the QNN.
    
    :param name: Name of the model
    :param N: Number of neurons per layer (modes of the quantum system)
    :param layers: Number of layers
    :param n_inputs: Number of QONN inputs
    :param n_outputs: Number of QONN outputs
    :param ladder_modes: Photon additions over system modes' that are desired (per layer)
    :param observable: Name of the observable to be measured ('position', 'momentum', 'number' or 'witness')
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
    assert layers == len(ladder_modes)
    
    n_pars = layers*(2*N**2 + 3*N)# + 3 # 2nd mode real input displacement # + (N + 1) # Linear readout weights + bias
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
    #bounds += [(None, None)]*(N+1) # Linear readout weights + bias
    #bounds += [disp_bounds]*3 # Initial displacement on mode 2
    
    print(f'Number of tunable parameters: {n_pars}')
    assert len(bounds) == n_pars, f"Number of bounds {len(bounds)} does not match number of parameters {n_pars}."
    
    def callback(xk):
        '''
        Prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QONN tunable parameters
        '''
        e = training_QNN(xk)
        loss_values.append(e)
        print(f'Training loss: {e}')
        if valid_set != None:
            val_e = validate_QNN(xk)
            print(f'Validation loss: {val_e}')
            validation_loss.append(val_e)
        else:
            validation_loss.append(e)
        
    def callback_hopping(x,f,accept):
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
        print(f"Best basinhopping iteration error so far: {best_loss_values[-1]}")
        print(f"Current basinhopping iteration error: {f}\n==========\n")
        if valid_set != None:
            if best_validation_loss[-1] > validation_loss[-1]:# and best_loss_values[-1] > loss_values[-1]:
                best_pars = x.copy()
                best_loss_values = loss_values.copy()
                best_validation_loss = validation_loss.copy()
        else:
            if best_loss_values[-1] > loss_values[-1]:# and best_loss_values[-1] > loss_values[-1]:
               best_pars = x.copy()
               best_loss_values = loss_values.copy()
        loss_values = [9999]
        validation_loss = [9999]
    qnn = QNN(model_name, N, layers, n_inputs, n_outputs, ladder_modes, is_addition, observable,
              include_initial_squeezing, include_initial_mixing, is_passive_gaussian,
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
    
    training_QNN = partial(qnn.train_QNN, inputs_dataset=train_inputs, outputs_dataset=train_outputs, loss_function=loss_function)
    if valid_set != None:
        validate_QNN = partial(qnn.train_QNN, inputs_dataset=valid_inputs, outputs_dataset=valid_outputs, loss_function=loss_function)
    #train_validation_inputs = np.concatenate((train_inputs, valid_inputs))
    #train_validation_outputs = np.concatenate((train_outputs, valid_outputs))
    
    training_start = time.time()
    # ==========
    """ jax_training_QNN = jax.jit(training_QNN)
    epochs = 100
    for epoch in range(epochs):
        loss, grads = jax.value_and_grad(jax_training_QNN)(init_pars)
        print(f"Epoch {epoch} loss: {loss}") """
    # ==========
    
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "callback": callback}
    opt_result = opt.basinhopping(training_QNN, init_pars, niter=hopping_iters, minimizer_kwargs=minimizer_kwargs, callback=callback_hopping)
    print(f'Total training time: {time.time() - training_start} seconds')
    
    print(f'\nOPTIMIZATION ERROR FOR N={N}, L={layers}')
    print(best_loss_values[-1])
    
    #qnn.build_QNN(opt_result.lowest_optimization_result.x)
    qnn.build_QNN(best_pars)

    qnn.print_qnn()
    
    if save:
        qnn.save_model(qnn.model_name + ".txt")
        
    if best_loss_values[0] == 9999:
        best_loss_values = best_loss_values[1:]
        best_validation_loss = best_validation_loss[1:]
    
    return qnn, best_loss_values, best_validation_loss
