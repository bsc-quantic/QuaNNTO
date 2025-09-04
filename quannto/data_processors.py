import numpy as np
import tensorflow as tf
import jax.numpy as jnp
from sklearn.decomposition import PCA

def trigonometric_feature_expressivity(features, num_final_features):
    transf_feats = np.zeros((len(features), num_final_features))
    for i in range(len(features)):
        for j in range(num_final_features):
            transf_feats[i,j] = (j+1)*np.sin((j+1)*np.pi*features[i])
    return transf_feats

def replicate_inputs(inputs, num_repl):
    repl_inputs = np.zeros((len(inputs), num_repl))
    for inp_idx in range(len(inputs)):
        repl_inputs[inp_idx, :] = inputs[inp_idx]
    return repl_inputs

def trigonometric_one_input(inputs):
    return np.sin(inputs)

def polynomial_feature_expressivity(features, num_final_features):
    transf_feats = np.zeros((len(features), num_final_features))
    for i in range(len(features)):
        for j in range(num_final_features):
            transf_feats[i,j] = (j+1)*features[i]**(j+1)
    return transf_feats

def get_range(data):
    return (np.min(data), np.max(data))

def rescale_data(data, data_range, scale_data_range):
    data_range_dist = data_range[1] - data_range[0]
    norm_data_range_dist = scale_data_range[1] - scale_data_range[0]
    return scale_data_range[0] + norm_data_range_dist * (data - data_range[0]) / data_range_dist

def rescale_set(set, rescale_range):
    rescaled_set = np.zeros_like(set)
    for col in range(len(set[0])):
        rescaled_set[:,col] = rescale_data(set[:,col], get_range(set[:,col]), rescale_range)
    return rescaled_set

def rescale_set_with_ranges(set, data_ranges, rescale_range):
    rescaled_set = np.zeros_like(set)
    for col in range(len(set[0])):
        rescaled_set[:,col] = rescale_data(set[:,col], data_ranges[col], rescale_range)
    return rescaled_set

def pad_data(data, length):
    batch_dim, entries_dim = data.shape
    pad_width = ((0, 0),          # no padding on batch‐axis
                (0, length - entries_dim))  # pad (2N−M) zeros to the right of each row
    return jnp.pad(data, pad_width, mode="constant", constant_values=0)

def binning(data, data_range, num_categories):
    # TODO: center the binning in the numerical value used for training
    cat_step = (data_range[1] - data_range[0]) / num_categories
    cat = np.zeros_like(data)
    for row_idx in range(len(data)):
        for col_idx in range(len(data[0])):
            threshold = data_range[0] + cat_step
            while data[row_idx, col_idx] > threshold:
                cat[row_idx, col_idx] += 1
                threshold += cat_step
    return cat

def softmax_discretization(outputs):
    prob_outputs = np.array([np.e**out for out in outputs])
    norm_outputs = np.sum(prob_outputs, axis=1)
    return np.array([prob_outputs[i]/norm_outputs[i] for i in range(len(prob_outputs))])

def one_hot_encoding(values, num_cats):
    one_hot_set = np.zeros((len(values), num_cats))
    for i in range(len(values)):
        one_hot_set[i][values[i][0]] = 1
    return one_hot_set

def greatest_probability(probs):
    m = np.zeros((len(probs), 1))
    for i in range(len(probs)):
        max_prob_idx = 0
        for j in range(len(probs[i])):
            if probs[i, j] > probs[i, max_prob_idx]:
                max_prob_idx = j
        m[i, 0] = max_prob_idx
    return m
    
def binary_class_probability(output, output_range):
    output_mid = (output_range[1] - output_range[0]) / 2
    return 0 if output < output_range[0] + output_mid else 1

def filter_dataset_categories(inputs, outputs, categories):
    filtered_inputs = []
    filtered_outputs = []
    for idx in range(len(outputs)):
        if outputs[idx] in categories:
            filtered_inputs.append(inputs[idx])
            filtered_outputs.append(outputs[idx])
    return (np.array(filtered_inputs), np.array(filtered_outputs))

def autoencoder_mnist(latent_dim, categories):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train) = filter_dataset_categories(x_train, y_train, categories)
    (x_test, y_test) = filter_dataset_categories(x_test, y_test, categories)
    print(f"AUTOENCODER DATASET SIZE: {len(x_test)}")

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Reshape input data to 28x28 images
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Define the autoencoder architecture with a N-dimensional latent space
    # Encoder
    encoder_input = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(latent_dim, activation='relu')(x)

    # Decoder
    #decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(392, activation='relu')(encoded)
    x = tf.keras.layers.Reshape((7, 7, 8))(x)
    x = tf.keras.layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder model
    autoencoder = tf.keras.models.Model(encoder_input, decoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Train the autoencoder
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # Extract the encoder part of the autoencoder model
    encoder_model = tf.keras.models.Model(encoder_input, encoded)

    # Encode the MNIST images to obtain the compressed representations
    x_train_compressed = encoder_model.predict(x_train)
    x_test_compressed = encoder_model.predict(x_test)

    # Return the compressed inputs of the training dataset
    #return x_train_compressed, np.array([[y_elem] for y_elem in y_train])
    return x_test_compressed, np.array([[y_elem] for y_elem in y_test])

def pca_mnist(latent_dim, categories):
    """
    Drop-in replacement for the autoencoder function, using PCA instead.
    - latent_dim: number of principal components
    - categories: passed to your filter_dataset_categories helper
    Returns:
        x_test_compressed: (N_test, latent_dim)
        y_test_column:     (N_test, 1)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train) = filter_dataset_categories(x_train, y_train, categories)
    (x_test, y_test) = filter_dataset_categories(x_test, y_test, categories)
    print(f"PCA DATASET SIZE: {len(x_test)}")

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Keep these lines to match your original structure
    x_train = np.expand_dims(x_train, axis=-1)  # (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, axis=-1)    # (N, 28, 28, 1)

    # Flatten images to vectors for PCA: (N, 784)
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    # Fit PCA with 'latent_dim' components (AE bottleneck -> PCA comps)
    pca = PCA(n_components=latent_dim, svd_solver='randomized', random_state=42)
    pca.fit(x_train_flat)

    # "Encode" = transform to principal component space
    x_train_compressed = pca.transform(x_train_flat)
    x_test_compressed = pca.transform(x_test_flat)

    # Optional: quick reconstruction error / variance info
    # x_test_recon = pca.inverse_transform(x_test_compressed)
    # mse_test = np.mean((x_test_flat - x_test_recon) ** 2)
    # print(f"PCA cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    # print(f"PCA reconstruction MSE (test): {mse_test:.6f}")

    # Return compressed test representations and column vector labels
    return x_test_compressed, np.array([[y] for y in y_test])


    