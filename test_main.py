"""Test seq2point model."""

import argparse

from seq2point_test import Tester
from train_main import (
    DEFAULT_ALGORITHM,
    DEFAULT_APPLIANCE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CROP,
    DATASET_DIR,
    DEFAULT_INPUT_WINDOW_LENGTH,
    remove_space,
)


# Default values for terminal arguments
DEFAULT_TEST_FILE = f"{DATASET_DIR}{DEFAULT_APPLIANCE}_test_.csv"


if __name__ == "__main__":
    # The arguments that can be passed in from the terminal
    parser = argparse.ArgumentParser(
        description="Train a pruned neural network for energy disaggregation."
    )

    parser.add_argument(
        "--appliance_name",
        type=remove_space,
        default=DEFAULT_APPLIANCE,
        help=f"Appliance name. Default: {DEFAULT_APPLIANCE}. Other options: kettle, fridge, microwave.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"The batch size to use when training. Default is {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=DEFAULT_CROP,
        help=f"The number of rows of training data. Default is {DEFAULT_CROP}.",
    )
    parser.add_argument(
        "--algorithm",
        type=remove_space,
        default=DEFAULT_ALGORITHM,
        help=f"The pruning algorithm of the model to test. Default is {DEFAULT_ALGORITHM}.",
    )
    parser.add_argument(
        "--network_type",
        type=remove_space,
        default="",
        help="Seq2point architecture. Options: default, dropout, reduced, and reduced_dropout.",
    )
    parser.add_argument(
        "--input_window_length",
        type=int,
        default=DEFAULT_INPUT_WINDOW_LENGTH,
        help=f"Number of input data points. Default is {DEFAULT_INPUT_WINDOW_LENGTH}.",
    )
    parser.add_argument(
        "--test_directory",
        type=str,
        default=DEFAULT_TEST_FILE,
        help="Directory for test data.",
    )

    # Parses the arguments from the terminal
    arguments = parser.parse_args()

    # You need to provide the trained model
    appliance = arguments.appliance_name
    saved_model_dir = f"saved_models/{appliance}_{arguments.algorithm}_model.h5"

    # The logs including results will be recorded to this log file
    log_file_dir = (
        f"saved_models/{appliance}_{arguments.algorithm}_{arguments.network_type}.log"
    )

    tester = Tester(
        appliance=arguments.appliance_name,
        algorithm=arguments.algorithm,
        crop=arguments.crop,
        batch_size=arguments.batch_size,
        network_type=arguments.network_type,
        test_directory=arguments.test_directory,
        saved_model_dir=saved_model_dir,
        log_file_dir=log_file_dir,
        input_window_length=arguments.input_window_length,
    )
    tester.test_model()
