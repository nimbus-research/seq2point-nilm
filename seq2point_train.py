"""Trains a seq2point model on a user-selected appliance and architecture."""

# from pathlib import Path

import numpy as np
import tensorflow as tf

from data_feeder import TrainSlidingWindowGenerator
from model_structure import create_model, save_model


class Trainer:
    """Used to train a seq2point model with or without pruning applied.

    Parameters:
    _appliance (string): The target appliance.
    _network_type (string): The architecture of the model.
    _batch_size (int): The number of rows per testing batch.
    _window_size (int): The size of each sliding window
    _window_offset (int): The offset of the inferred value from the sliding window.
    _max_chunk_size (int): The largest possible number of row per chunk.
    _validation_frequency (int): The number of epochs between model validation.
    _training_directory (string): The directory of the model's training file.
    _validation_directory (string): The directory of the model's validation file.
    _training_chunker (TrainSlidingWindowGenerator): A sliding window provider
    that returns feature / target pairs. For training use only.
    _validation_chunker (TrainSlidingWindowGenerator): A sliding window provider
    that returns feature / target pairs. For validation use only.
    """

    def __init__(
        self,
        appliance,
        batch_size,
        crop,
        network_type,
        training_directory,
        validation_directory,
        save_model_dir,
        epochs=10,
        input_window_length=599,
        validation_frequency=1,
        patience=3,
        min_delta=1e-6,
        verbose=1,
    ):
        self._appliance = appliance
        self._algorithm = network_type
        self._network_type = network_type
        self._crop = crop
        self._batch_size = batch_size
        self._epochs = epochs
        self._patience = patience
        self._min_delta = min_delta
        self._verbose = verbose
        self._loss = "mse"
        self._metrics = ["mse", "msle", "mae"]
        self._learning_rate = 0.001
        self._beta_1 = 0.9
        self._beta_2 = 0.999
        self._save_model_dir = save_model_dir

        self._input_window_length = input_window_length
        self._window_size = 2 + self._input_window_length
        self._window_offset = int((0.5 * self._window_size) - 1)
        self._max_chunk_size = 5 * 10**2
        self._validation_frequency = validation_frequency
        self._ram_threshold = 5 * 10**5
        self._skip_rows_train = 10000000
        self._validation_steps = 100
        self._skip_rows_val = 0

        # Directories of the training and validation files.
        self._training_directory = training_directory
        self._validation_directory = validation_directory

        kwargs = {
            "chunk_size": self._max_chunk_size,
            "batch_size": self._batch_size,
            "crop": self._crop,
            "shuffle": True,
            "offset": self._window_offset,
            "ram_threshold": self._ram_threshold,
        }
        self._training_chunker = TrainSlidingWindowGenerator(
            file_name=self._training_directory,
            skip_rows=self._skip_rows_train,
            **kwargs
        )
        self._validation_chunker = TrainSlidingWindowGenerator(
            file_name=self._validation_directory,
            skip_rows=self._skip_rows_val,
            **kwargs
        )

    def train_model(self):
        """Trains an energy disaggregation model using a pruning algorithm (default is no pruning).
        Plots and saves the resulting model.
        """
        # Calculate the optimum steps per epoch.
        # self._training_chunker.check_if_chunking()
        # steps_per_training_epoch = np.round(int(self._training_chunker.total_size / self._batch_size), decimals=0)
        steps_per_training_epoch = np.round(
            int(self._training_chunker.total_num_samples / self._batch_size), decimals=0
        )

        model = create_model(self._input_window_length)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self._learning_rate,
                beta_1=self._beta_1,
                beta_2=self._beta_2,
            ),
            loss=self._loss,
            metrics=self._metrics,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=self._min_delta,
            patience=self._patience,
            verbose=self._verbose,  # type: ignore
            mode="auto",
        )

        # can use checkpoint ###############################################
        # checkpoint_filepath = "checkpoint/housedata/refit/"+ self._appliance + "/"
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath = checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
        #     save_weights_only=False, mode='auto', save_freq='epoch')
        # callbacks=[early_stopping, model_checkpoint_callback]
        ###################################################################

        callbacks = [early_stopping]
        training_history = self.default_train(
            model, callbacks, steps_per_training_epoch
        )
        training_history.history["val_loss"] = np.repeat(  # type: ignore
            training_history.history["val_loss"], self._validation_frequency
        )
        model.summary()
        save_model(
            model,
            self._network_type,
            self._algorithm,
            self._appliance,
            self._save_model_dir,
        )
        self.plot_training_results(training_history)

    def default_train(self, model, callbacks, steps_per_training_epoch):
        """The default training method the neural network will use. No pruning occurs.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.ndarray): Error metrics and loss values at the end of each epoch.
        """
        training_history = model.fit(
            self._training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=self._epochs,
            verbose=self._verbose,
            callbacks=callbacks,
            validation_data=self._validation_chunker.load_dataset(),
            validation_freq=self._validation_frequency,
            validation_steps=self._validation_steps,
        )
        return training_history

    def plot_training_results(self, training_history):
        """Plots and saves a graph of training loss against epoch.

        Parameters:
        training_history (numpy.ndarray): A timeseries of loss against epoch count.
        """
        # plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        # plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        # plt.title('Training History')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend()
        # file_name = Path(
        #     self._appliance, "saved_models",
        #     f'{self._appliance}_{self._pruning_algorithm}_{self._network_type}_training_results.png'
        # )
        # plt.savefig(fname=file_name)
