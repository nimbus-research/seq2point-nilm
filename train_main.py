"""Train seq2point model."""

import argparse

from seq2point_train import Trainer


DATASET = "UK-DALE_2015"

DEFAULT_APPLIANCE = "dishwasher"

# DATASET_DIR = f"dataset_management/ukdale/{DEFAULT_APPLIANCE}/"
DATASET_DIR = f"../../../data/processed/{DATASET}/{DEFAULT_APPLIANCE}/"

# Default values for terminal arguments
DEFAULT_BATCH_SIZE = 1000
DEFAULT_CROP = 100000
DEFAULT_ALGORITHM = "seq2point"
DEFAULT_NETWORK_TYPE = "default"
DEFAULT_EPOCHS = 10
DEFAULT_INPUT_WINDOW_LENGTH = 599
DEFAULT_VALIDATION_FREQUENCY = 1
DEFAULT_TRAIN_DIR = f"{DATASET_DIR}{DEFAULT_APPLIANCE}_training_.csv"
DEFAULT_VAL_DIR = f"{DATASET_DIR}{DEFAULT_APPLIANCE}_validation_.csv"


def remove_space(string):
    """Removes the spaces from a string. Used for parsing terminal inputs."""
    return string.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train sequence-to-point learning for energy disaggregation."
    )

    parser.add_argument(
        "--appliance_name",
        type=remove_space,
        default=DEFAULT_APPLIANCE,
        help=(
            f"Appliance name. Default is {DEFAULT_APPLIANCE}. Other options: "
            f"kettle, fridge, washing machine, and microwave."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"The batch size to use when training the network. Default is {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=DEFAULT_CROP,
        help=f"The number of rows of training data. Default is {DEFAULT_CROP}.",
    )
    # This is commented out in the original code:
    # parser.add_argument(
    #     '--pruning_algorithm', type=remove_space, default='default',
    #     help='Pruning algorithm. Default is none. Available are: spp, entropic, threshold.'
    # )
    parser.add_argument(
        "--network_type",
        type=remove_space,
        default=DEFAULT_NETWORK_TYPE,
        help=f"The seq2point architecture to use. Default is {DEFAULT_NETWORK_TYPE}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs. Default is {DEFAULT_EPOCHS}.",
    )
    parser.add_argument(
        "--input_window_length",
        type=int,
        default=DEFAULT_INPUT_WINDOW_LENGTH,
        help=f"Number of input data points. Default is {DEFAULT_INPUT_WINDOW_LENGTH}.",
    )
    parser.add_argument(
        "--validation_frequency",
        type=int,
        default=DEFAULT_VALIDATION_FREQUENCY,
        help=f"How often to validate model. Default is {DEFAULT_VALIDATION_FREQUENCY}.",
    )
    parser.add_argument(
        "--training_directory",
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help="The dir for training data.",
    )
    parser.add_argument(
        "--validation_directory",
        type=str,
        default=DEFAULT_VAL_DIR,
        help="The dir for validation data.",
    )

    arguments = parser.parse_args()

    # Need to provide the trained model
    save_model_dir = (
        f"saved_models/{arguments.appliance_name}_{arguments.network_type}_model.h5"
    )

    trainer = Trainer(
        appliance=arguments.appliance_name,
        batch_size=arguments.batch_size,
        crop=arguments.crop,
        network_type=arguments.network_type,
        training_directory=arguments.training_directory,
        validation_directory=arguments.validation_directory,
        save_model_dir=save_model_dir,
        epochs=arguments.epochs,
        input_window_length=arguments.input_window_length,
        validation_frequency=arguments.validation_frequency,
    )
    trainer.train_model()
