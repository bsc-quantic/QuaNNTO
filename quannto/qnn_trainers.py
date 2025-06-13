import numpy as np
from functools import reduce, partial
import time
import scipy.optimize as opt
import jax.numpy as jnp

from .loss_functions import mse
from .qnn import QNN


def build_and_train_model(name, N, layers, n_inputs, n_outputs, photon_additions, observable, is_input_reupload, 
                          train_set, valid_set, loss_function=mse, hopping_iters=2, in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    '''
    Creates and trains a QNN model with the given hyperparameters and dataset by optimizing the 
    tunable parameters of the QNN.
    
    :param name: Name of the model
    :param N: Number of neurons per layer (modes of the quantum system)
    :param layers: Number of layers
    :param n_inputs: Number of QONN inputs
    :param n_outputs: Number of QONN outputs
    :param photon_additions: Photon additions over system modes' that are desired (per layer)
    :param observable: Name of the observable to be measured ('position', 'momentum', 'number' or 'witness')
    :param is_input_reupload: Boolean variable telling whether the QONN has to have input-reuploading or not
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
    if type(init_pars) == type(None):
        #init_pars = np.random.rand(2*((N**2 + N) //2) + 3*N + layers*(2*((N**2 + N) // 2) + N)) if is_input_reupload else np.random.rand(2*((N**2 + N) //2) + 3*N + layers*(2*((N**2 + N) // 2) + N + 2*N))
        #init_pars = np.random.rand(2*N**2 + 3*N + layers*(2*N**2 + N)) if is_input_reupload else np.random.rand(2*N**2 + 3*N + layers*(2*N**2 + 3*N))
        n_pars = layers*(2*N**2 + N) if is_input_reupload else layers*(2*N**2 + 3*N)
        init_pars = np.random.rand(n_pars) if observable!='witness' else np.random.rand(n_pars + N + 1) # TODO: Consider parameterize momentum too
    else:
        aux_pars = 0 if observable!='witness' else 1
        if is_input_reupload:
            #assert len(init_pars) == 2*((N**2 + N) //2) + N + layers*(2*((N**2 + N) // 2) + N)
            #assert len(init_pars) == 2*N**2 + N + layers*(2*N**2 + N)
            assert len(init_pars) == layers*(2*N**2 + N) + aux_pars
        else:
            #assert len(init_pars) == 2*((N**2 + N) //2) + 3*N + layers*(2*((N**2 + N) // 2) + N + 2*N)
            assert len(init_pars) == layers*(2*N**2 + 3*N) + aux_pars
    
    passive_bounds = (None, None)
    sqz_bounds = (np.log(0.1), np.log(10))
    disp_bounds = (-2/np.sqrt(2), 2/np.sqrt(2))
    bounds = []
    for _ in range(layers):
        # Passive optics bounds
        bounds += [passive_bounds for _ in range(2*N**2)]
        # Squeezing bounds
        bounds += [sqz_bounds for _ in range(N)]
        # Displacement bounds
        if not is_input_reupload:
            bounds += [disp_bounds for _ in range(2*N)]
    
    def callback(xk):
        '''
        Prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QONN tunable parameters
        '''
        e = training_QNN(xk)
        print(f'Training loss: {e}')
        val_e = validate_QNN(xk)
        print(f'Validation loss: {val_e}')
        loss_values.append(e)
        validation_loss.append(val_e)
        
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
        print(f"Best basinhopping iteration error so far: {best_loss_values[-1]}")
        print(f"Current basinhopping iteration error: {f}\n==========\n")
        if best_validation_loss[-1] > validation_loss[-1]:# and best_loss_values[-1] > loss_values[-1]:
            best_loss_values = loss_values.copy()
            best_validation_loss = validation_loss.copy()
        loss_values = [9999]
        validation_loss = [9999]
    
    qnn = QNN("model_N" + str(N) + "_L" + str(layers) + "_" + name, N, layers, n_inputs, n_outputs,
              photon_additions, observable, is_input_reupload, in_preprocs, out_prepocs, postprocs)
    train_inputs = reduce(lambda x, func: func(x), qnn.in_preprocessors, train_set[0])
    train_outputs = reduce(lambda x, func: func(x), qnn.out_preprocessors, train_set[1])
    
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
    
    training_QNN = partial(qnn.train_QNN, inputs_dataset=train_inputs, outputs_dataset=train_outputs, loss_function=loss_function)
    validate_QNN = partial(qnn.train_QNN, inputs_dataset=valid_inputs, outputs_dataset=valid_outputs, loss_function=loss_function)
    
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
    print(opt_result.fun)
    
    qnn.build_QNN(opt_result.x)

    qnn.print_qnn()
    print(qnn.qnn_profiling.avg_benchmark())
    qnn.qnn_profiling.avg_epochs()
    
    if save:
        qnn.qnn_profiling.clear_times()
        qnn.save_model(qnn.model_name + ".txt")
        
    if best_loss_values[0] == 9999:
        best_loss_values = best_loss_values[1:]
        best_validation_loss = best_validation_loss[1:]
    
    return qnn, best_loss_values, best_validation_loss

def train_entanglement_witness(name, N, layers, n_inputs, n_outputs, photon_additions, observable, hopping_iters=2, in_preprocs=[], out_prepocs=[], postprocs=[], init_pars=None, save=True):
    n_pars = layers*(2*N**2 + 3*N)
    if type(init_pars) == type(None):
        init_pars = np.random.rand(n_pars + N + 1) # TODO: Consider parameterize momentum too
    
    inp_disp_bounds = (-2, 2)
    passive_bounds = (None, None)
    sqz_bounds = (np.log(0.1), np.log(10))
    disp_bounds = (-2/np.sqrt(2), 2/np.sqrt(2))
    witness_par_bound = (0,1)
    bounds = []
    for _ in range(layers):
        # Passive optics bounds
        bounds += [passive_bounds for _ in range(2*N**2)]
        # Squeezing bounds
        bounds += [sqz_bounds for _ in range(N)]
        # Displacement bounds
        bounds += [disp_bounds for _ in range(2*N)]
    # Witness parameter bounds
    bounds += [witness_par_bound]
    # Input displacements bounds
    bounds += [inp_disp_bounds for _ in range(N)]
    
    qnn = QNN("model_N" + str(N) + "_L" + str(layers) + "_" + name, N, layers, n_inputs, n_outputs, photon_additions, observable)
    
    def callback(xk):
        '''
        Prints and stores the MSE error value for each QNN training epoch.
        
        :param xk: QONN tunable parameters
        '''
        e = train_witness(xk)
        print(f'Witness expected value: {e}')
        loss_values.append(e)
        
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
        print(f"Best basinhopping iteration value: {best_loss_values[-1]}")
        print(f"Current basinhopping iteration value: {f}\n==========\n")
        if best_loss_values[-1] > loss_values[-1]:# and best_loss_values[-1] > loss_values[-1]:
            best_loss_values = loss_values.copy()
        loss_values = [9999]
    
    global best_loss_values
    best_loss_values = [9999]
    global loss_values
    loss_values = [9999]
    
    train_witness = partial(qnn.train_ent_witness)
    
    training_start = time.time()
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "callback": callback}
    result = opt.basinhopping(train_witness, init_pars, niter=hopping_iters, minimizer_kwargs=minimizer_kwargs, callback=callback_hopping)
    print(bounds)
    print(f'Total training time: {time.time() - training_start} seconds')
    
    print(f'\nOPTIMIZED ENTANGLEMENT WITNESS VALUE FOR N={N}, L={layers}')
    print(result.fun)
    
    qnn.build_QNN(result.x)
    qnn.print_qnn()
    print(qnn.qnn_profiling.avg_benchmark())
    
    if save:
        qnn.qnn_profiling.clear_times()
        qnn.save_model(qnn.model_name + ".txt")
        
    if best_loss_values[0] == 9999:
        best_loss_values = best_loss_values[1:]
    
    return qnn, best_loss_values
