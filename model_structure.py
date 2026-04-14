"""Defines the structure of a seq2point model and miscellaneous functions."""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, Input, Reshape


def create_model(input_window_length: int) -> tf.keras.Model:
    """Specifies the structure of a seq2point model using Keras' functional API.

    :param int input_window_length: The number of data points in the input window.
    :return tf.keras.Model model: The untrained seq2point model.
    """
    # Input layer
    input_layer = Input(shape=(input_window_length,))
    reshape_layer = Reshape((1, input_window_length, 1))(input_layer)

    # Convolutional layers
    kwargs = {"strides": (1, 1), "padding": "same", "activation": "relu"}
    conv_layer_1 = Convolution2D(filters=30, kernel_size=(10, 1), **kwargs)(reshape_layer)
    conv_layer_2 = Convolution2D(filters=30, kernel_size=(8, 1), **kwargs)(conv_layer_1)
    conv_layer_3 = Convolution2D(filters=40, kernel_size=(6, 1), **kwargs)(conv_layer_2)
    conv_layer_4 = Convolution2D(filters=50, kernel_size=(5, 1), **kwargs)(conv_layer_3)
    conv_layer_5 = Convolution2D(filters=50, kernel_size=(5, 1), **kwargs)(conv_layer_4)

    # Fully connected layers
    flatten_layer = Flatten()(conv_layer_5)
    label_layer = Dense(1024, activation="relu")(flatten_layer)
    output_layer = Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


# def save_model(model, network_type, algorithm, appliance, save_model_dir):  # Original version
    # model_name = f"saved_models/{appliance}_{algorithm}_{network_type}_model.h5"
def save_model(model: tf.keras.Model, save_model_dir: str) -> None:
    """Saves a model to a specified location.

    :param tf.keras.Model model: The model to save.
    :param str save_model_dir: The directory to save the model to.
    """
    model_path = save_model_dir
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(model_path):
        open((model_path), "a", encoding="utf-8").close()
    model.save(model_path)


# def load_model(model, network_type, algorithm, appliance, saved_model_dir):  # Original version
    # model_path = f"saved_models/{appliance}_{algorithm}_{network_type}_model.h5"
def load_model(model: tf.keras.Model, algorithm: str, saved_model_dir: str) -> tf.keras.Model:
    """Loads a model from a specified location.

    :param tf.keras.Model model: The model to which the loaded weights will be applied to.
    :param str algorithm: The pruning algorithm applied to the model.
    :param str saved_model_dir: The directory of the model to load.
    :return tf.keras.Model model: The model with the loaded weights.
    """
    model_path = saved_model_dir
    if not Path(model_path).is_file():
        model_path = saved_model_dir.replace(f"_{algorithm}_", "_default_")
    print(f"PATH NAME: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)
    except (FileNotFoundError, ValueError):
        model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])

    num_of_weights = model.count_params()
    print(f"Loaded model with {num_of_weights} weights")
    return model
