"""Tests a pre-trained seq2point model with or without pruning applied, and evaluates the results."""

import logging
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from appliance_data import appliance_data, mains_data
from data_feeder import TestSlidingWindowGenerator
from model_structure import create_model, load_model


class Tester:
    """Used to test and evaluate a pre-trained seq2point model with or without pruning applied.

    Parameters:
    _appliance (string): The target appliance.
    _algorithm (string): The (pruning) algorithm the model was trained with.
    _network_type (string): The architecture of the model.
    _crop (int): The maximum number of rows of data to evaluate the model with.
    _batch_size (int): The number of rows per testing batch.
    _window_size (int): The size of each sliding window
    _window_offset (int): The offset of the inferred value from the sliding window.
    _test_directory (string): The directory of the test file for the model.
    """

    def __init__(
        self,
        appliance,
        algorithm,
        crop,
        batch_size,
        network_type,
        test_directory,
        saved_model_dir,
        log_file_dir,
        input_window_length,
    ):
        self._appliance = appliance
        self._algorithm = algorithm
        self._network_type = network_type

        self._crop = crop
        self._batch_size = batch_size
        self._input_window_length = input_window_length
        self._window_size = self._input_window_length + 2
        self._window_offset = int(0.5 * self._window_size - 1)
        self._number_of_windows = 100

        self._test_directory = test_directory
        self._saved_model_dir = saved_model_dir

        self._log_file = log_file_dir
        logging.basicConfig(filename=self._log_file, level=logging.INFO)

    def test_model(self):
        """Tests a fully-trained model using a sliding window generator as an input.

        Measures inference time, gathers, and plots evaluation metrics.
        """
        test_input, test_target = self.load_dataset(self._test_directory)
        model = create_model(self._input_window_length)
        model = load_model(
            model,
            self._network_type,
            self._algorithm,
            self._appliance,
            self._saved_model_dir,
        )
        test_generator = TestSlidingWindowGenerator(
            number_of_windows=self._number_of_windows,
            inputs=test_input,
            targets=test_target,
            offset=self._window_offset,
        )

        # Calculate the optimum steps per epoch
        steps_per_test_epoch = np.round(
            int(test_generator.total_size / self._batch_size), decimals=0
        )

        # Test the model
        start_time = time.time()
        testing_history = model.predict(
            x=test_generator.load_dataset(), steps=steps_per_test_epoch, verbose=2
        )
        end_time = time.time()
        test_time = end_time - start_time

        evaluation_metrics = model.evaluate(
            x=test_generator.load_dataset(), steps=steps_per_test_epoch
        )
        self.log_results(model, test_time, evaluation_metrics)
        self.plot_results(testing_history, test_input, test_target)

    def load_dataset(self, directory):
        """Loads the testing dataset from the location specified by file_name.

        Parameters:
        directory (string): The location at which the dataset is stored, concatenated with the file name.

        Returns:
        test_input (numpy.array): The first n (crop) features of the test dataset.
        test_target (numpy.array): The first n (crop) targets of the test dataset.
        """
        data_frame = pd.read_csv(directory, nrows=self._crop, skiprows=0, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        test_target = np.round(
            np.array(
                data_frame.iloc[self._window_offset : -self._window_offset, 1], float
            ),
            6,
        )
        del data_frame
        return test_input, test_target

    def log_results(self, model, test_time, evaluation_metrics):
        """Logs the inference time, MAE and MSE of an evaluated model.

        Parameters:
        model (tf.keras.Model): The evaluated model.
        test_time (float): The time taken by the model to infer all required values.
        evaluation metrics (list): The MSE, MAE, and various compression ratios of the model.
        """
        logging.info(f"Inference Time: {test_time}")
        self.count_pruned_weights(model)

    def count_pruned_weights(self, model):
        """Counts the total number of weights, pruned weights, and weights in convolutional layers.

        Calculates the sparsity ratio of different layer types and logs these values.

        Parameters:
        model (tf.keras.Model): The evaluated model.
        """
        num_total_zeros = 0
        num_dense_zeros = 0
        num_dense_weights = 0
        num_conv_zeros = 0
        num_conv_weights = 0
        for layer in model.layers:
            # if np.shape(layer.get_weights())[0] != 0:  # Original. Does not work anymore
            if len(layer.get_weights()) != 0:
                layer_weights = layer.get_weights()[0].flatten()

                if "conv" in layer.name:
                    num_conv_weights += np.size(layer_weights)
                    num_conv_zeros += np.count_nonzero(layer_weights == 0)
                    num_total_zeros += np.size(layer_weights)
                else:
                    num_dense_weights += np.size(layer_weights)
                    num_dense_zeros += np.count_nonzero(layer_weights == 0)

        conv_zeros_string = "CONV. ZEROS: " + str(num_conv_zeros)
        conv_weights_string = "CONV. WEIGHTS: " + str(num_conv_weights)
        conv_sparsity_ratio = "CONV. RATIO: " + str(num_conv_zeros / num_conv_weights)

        dense_weights_string = "DENSE WEIGHTS: " + str(num_dense_weights)
        dense_zeros_string = "DENSE ZEROS: " + str(num_dense_zeros)
        dense_sparsity_ratio = "DENSE RATIO: " + str(
            num_dense_zeros / num_dense_weights
        )

        total_zeros_string = "TOTAL ZEROS: " + str(num_total_zeros)
        total_weights_string = "TOTAL WEIGHTS: " + str(model.count_params())
        total_sparsity_ratio = "TOTAL RATIO: " + str(
            num_total_zeros / model.count_params()
        )

        print(f"LOGGING PATH: {self._log_file}")

        logging.info(conv_zeros_string)
        logging.info(conv_weights_string)
        logging.info(conv_sparsity_ratio)
        logging.info("")
        logging.info(dense_zeros_string)
        logging.info(dense_weights_string)
        logging.info(dense_sparsity_ratio)
        logging.info("")
        logging.info(total_zeros_string)
        logging.info(total_weights_string)
        logging.info(total_sparsity_ratio)

    def plot_results(self, testing_history, test_input, test_target):
        """Generates and saves a plot of the test values against aggregate energy and appliance values.

        Parameters:
        testing_history (numpy.ndarray): The series of values inferred by the model.
        test_input (numpy.ndarray): The aggregate energy data.
        test_target (numpy.ndarray): The true energy values of the appliance.
        """
        params = appliance_data[self._appliance]
        testing_history = (testing_history * params["std"]) + params["mean"]
        test_target = (test_target * params["std"]) + params["mean"]
        test_agg = (test_input.flatten() * mains_data["std"]) + mains_data["mean"]
        test_agg = test_agg[: testing_history.size]

        # Can't have negative energy readings - set any results below 0 to 0.
        test_target[test_target < 0] = 0
        testing_history[testing_history < 0] = 0
        test_input[test_input < 0] = 0

        # Plot testing outcomes against ground truth.
        plt.figure(1)
        plt.plot(
            test_agg[self._window_offset : -self._window_offset], label="Aggregate"
        )
        plt.plot(
            test_target[: test_agg.size - (2 * self._window_offset)],
            label="Ground Truth",
        )
        plt.plot(
            testing_history[: test_agg.size - (2 * self._window_offset)],
            label="Predicted",
        )
        plt.title(
            self._appliance + " " + self._network_type + "(" + self._algorithm + ")"
        )
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        file_path = Path(
            self._appliance,
            "saved_models",
            f"{self._appliance}_{self._algorithm}_{self._network_type}_test_figure.png",
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=file_path)
        plt.show()
