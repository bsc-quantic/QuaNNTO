from functools import partial, reduce
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from .dataset_gens.synth_datasets import *

from quannto.core.data_processors import get_range, rescale_data, trigonometric_feature_expressivity
from .dataset_gens.synth_datasets import generate_dataset_of, generate_linear_dataset_of, print_dataset, sin_cos_function

tf.experimental.numpy.experimental_enable_numpy_behavior()

# Define the number of parameters
neurons = [2,3,4]
n_ins = [1,1,1]
n_out = 1
input_range = (-3,3)
dataset_size = 70
testset_size = 200

target_function = test_linear_1in_1out

#train_df = pd.read_csv("curve_fitting_datasets/sincos_trainset.csv")
#in_train = train_df.iloc[:, 0:1].to_numpy()
#out_train = train_df.iloc[:, -n_out:].to_numpy()
#train_dataset = (in_train, out_train)
train_dataset = generate_dataset_of(target_function, 1, 1, dataset_size, input_range)

#eval_df = pd.read_csv("curve_fitting_datasets/sincos_testset_lin.csv")
#in_test = eval_df.iloc[:, 0:1].to_numpy()
#out_test = eval_df.iloc[:, -n_out:].to_numpy()
#test_dataset = (in_test, out_test)
test_dataset = generate_linear_dataset_of(target_function, 1, 1, testset_size, input_range)


in_norm_range = (0,1)
out_norm_range = (0,1)

# Custom loss function to be used with SciPy optimizer
def custom_loss(params):
    weights_hidden = params[:n_in * N].reshape((n_in, N))  # Reshape weights for hidden layer
    biases_hidden = params[n_in * N:n_in * N + N]  # Biases for hidden layer
    weights_output = params[n_in * N + N:n_in * N + N + N].reshape((N, 1))  # Reshape weights for output layer
    bias_output = params[-1]  # Bias for output layer
    model.set_weights([weights_hidden, biases_hidden, weights_output, np.array([bias_output])])
    y_pred = model(x_train)
    return np.mean(np.square(y_train - y_pred))

# Custom gradient function for the loss function
def custom_gradient(params):
    with tf.GradientTape() as tape:
        weights_hidden = tf.convert_to_tensor(params[:n_in * N], dtype=tf.float32).reshape((n_in, N))  # Reshape weights for hidden layer
        biases_hidden = tf.convert_to_tensor(params[n_in * N:n_in * N + N], dtype=tf.float32)  # Biases for hidden layer
        weights_output = tf.convert_to_tensor(params[n_in * N + N:n_in * N + N + N], dtype=tf.float32).reshape((N, 1))  # Reshape weights for output layer
        bias_output = tf.convert_to_tensor(params[-1], dtype=tf.float32)  # Bias for output layer
        model.set_weights([weights_hidden.numpy(), biases_hidden.numpy(), weights_output.numpy(), np.array([bias_output])])
        y_pred = model(x_train)
        loss = tf.reduce_mean(tf.square(y_train - y_pred))
    gradients = tape.gradient(loss, model.trainable_variables)
    flat_gradients = np.concatenate([tf.reshape(g, [-1]) for g in gradients])
    return flat_gradients

def callback(xk):
    '''
    Callback function that prints and stores the MSE error value for each QNN training epoch.
    
    :param xk: QNN tunable parameters
    '''
    e = custom_loss(xk)
    #print(e)
    loss_values.append(e)
    
def callback_hopping(x,f,accept):
    global best_loss_values
    global loss_values
    print(f"Error of basinhopping iteration: {f}")
    print(f"Previous iteration error: {best_loss_values[-1]}\n")
    if best_loss_values[-1] > f:
        best_loss_values = loss_values.copy()
    loss_values = []

plt.plot(test_dataset[1], 'go', label='Target function')

output_range = get_range(generate_linear_dataset_of(target_function, 1, 1, 2000, input_range)[1])
losses = []
colors = ['red','fuchsia', 'blue', 'purple','orange','brown']
for (N, n_in, color) in zip(neurons, n_ins, colors):    
    in_preprocessors = []
    #in_preprocessors.append(partial(trigonometric_feature_expressivity, num_final_features=n_in))
    #trig_range = (-N, N)
    in_preprocessors.append(partial(rescale_data, data_range=input_range, scale_data_range=in_norm_range))
    
    x_train = reduce(lambda x, func: func(x), in_preprocessors, train_dataset[0])
    
    out_preprocessors = []
    out_preprocessors.append(partial(rescale_data, data_range=output_range, scale_data_range=out_norm_range))
    
    y_train = reduce(lambda x, func: func(x), out_preprocessors, train_dataset[1])

    # Define the neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(N, activation='sigmoid', input_shape=(n_in,)),
        tf.keras.layers.Dense(1)
    ])
    global best_loss_values
    best_loss_values = [10]
    global loss_values
    loss_values = []
    print(f'For {N} neurons and {model.count_params()} parameters')
    
    initial_params = np.random.randn(model.count_params())
    result = opt.minimize(custom_loss, initial_params, jac=custom_gradient, method='L-BFGS-B', callback=callback)
    #minimizer_kwargs = {"method": "L-BFGS-B", "jac": custom_gradient, "callback": callback}
    #result = opt.basinhopping(custom_loss, initial_params, niter=19, minimizer_kwargs=minimizer_kwargs, callback=callback_hopping)
    print(f'OPTIMIZATION ERROR: {result.fun}\n')
    #losses.append(best_loss_values.copy())
    losses.append(loss_values.copy())
    optimized_params = result.x

    # Set the optimized parameters back to the model
    weights_hidden = optimized_params[:n_in * N].reshape((n_in, N))  # Reshape weights for hidden layer
    biases_hidden = optimized_params[n_in * N:n_in * N + N]  # Biases for hidden layer
    weights_output = optimized_params[n_in * N + N:n_in * N + N + N].reshape((N, 1))  # Reshape weights for output layer
    bias_output = optimized_params[-1]  # Bias for output layer

    model.set_weights([weights_hidden, biases_hidden, weights_output, np.array([bias_output])])

    test_inputs = reduce(lambda x, func: func(x), in_preprocessors, test_dataset[0])
    #x_test = eval_df.iloc[:, 0:n_in].to_numpy()
    #y_pred = model.predict(x_test)
    y_pred = model.predict(test_inputs)
    postprocessors = []
    postprocessors.append(partial(rescale_data, data_range=out_norm_range, scale_data_range=output_range))
    y_out = reduce(lambda x, func: func(x), postprocessors, y_pred)
    plt.plot(y_out, c=color, linestyle='dashed', label=f'N={N}')
plt.title(f'TESTING SET')
plt.xlabel('Input samples')
plt.ylabel('Output')
plt.legend()
plt.show()

for (loss, color, N) in zip(losses, colors, neurons):
    plt.plot(np.log(np.array(loss)+1), c=color, label=f'N={N}')
plt.ylim(bottom=0.0)
plt.xlabel('Epochs')
plt.ylabel('Logarithmic loss value')
plt.title(f'LOGARITHMIC LOSS FUNCTIONS')
plt.legend()
plt.show()