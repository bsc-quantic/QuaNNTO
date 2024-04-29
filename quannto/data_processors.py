import numpy as np
import tensorflow as tf

def trigonometric_feature_expressivity(features, num_final_features):
    transf_feats = np.zeros((len(features), num_final_features))
    for i in range(len(features)):
        for j in range(num_final_features):
            transf_feats[i,j] = (j+1)*np.sin((j+1)*features[i])
    return transf_feats

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

def autoencoder_mnist(latent_dim):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
                    epochs=5,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # Extract the encoder part of the autoencoder model
    encoder_model = tf.keras.models.Model(encoder_input, encoded)

    # Encode the MNIST images to obtain the compressed representations
    x_train_compressed = encoder_model.predict(x_train)
    x_test_compressed = encoder_model.predict(x_test)

    # Return the compressed inputs of the training dataset
    return x_train_compressed, np.array([[y_elem] for y_elem in y_train])

    