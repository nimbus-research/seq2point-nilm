"""Data feeder for training and testing a ConvNet."""

import numpy as np
import pandas as pd

# batch_size: the number of rows fed into the network at once.
# crop: the number of rows in the data set to be used in total.
# chunk_size: the number of lines to read from the file at once.


class TrainSlidingWindowGenerator:
    """Yields features and targets for training a ConvNet.

    Parameters:
    _file_name (string): The path where the training dataset is located.
    _batch_size (int): The size of each batch from the dataset to be processed.
    _chunk_size (int): The size of each chunk of data to be processed.
    _shuffle (bool): Whether the dataset should be shuffled before being returned.
    _offset (int):
    _crop (int): The number of rows of the dataset to return.
    _skip_rows (int): The number of rows of a dataset to skip before reading data.
    _ram_threshold (int): The maximum amount of RAM to utilise at a time.
    total_size (int): The number of rows read from the dataset.
    """

    def __init__(
        self,
        file_name,
        chunk_size,
        shuffle,
        offset,
        batch_size=1000,
        crop=100000,
        skip_rows=0,
        ram_threshold=5 * 10**5,
    ):
        self._file_name = file_name
        self._batch_size = batch_size
        self._chunk_size = 10**8
        self._shuffle = shuffle
        self._offset = offset
        self._crop = crop
        self._skip_rows = skip_rows
        self._ram_threshold = ram_threshold
        self.total_size = 0
        self._total_num_samples = crop

    @property
    def total_num_samples(self):
        return self._total_num_samples

    @total_num_samples.setter
    def total_num_samples(self, value):
        self._total_num_samples = value

    def check_if_chunking(self):
        """Determine whether dataset is longer than the chunking threshold or not."""
        # Loads the file and counts the number of rows it contains.
        print("Importing training file...")
        chunks = pd.read_csv(
            self._file_name,
            header=0,
            nrows=self._crop,  # skiprows=self._skip_rows
            # pandas.errors.EmptyDataError: No columns to parse from file:
            # self._file_name, header=0, nrows=self._crop, skiprows=self._skip_rows
        )
        print("Counting number of rows...")
        self.total_size = len(chunks)
        del chunks
        print("Done.")
        print("The dataset contains ", self.total_size, " rows")

        # Display a warning if there are too many rows to fit in the designated amount RAM.
        if self.total_size > self._ram_threshold:
            print(
                "Too much data to load into memory, so it will be loaded in chunks. "
                "Please note that this may result in decreased training times."
            )

    def load_dataset(self):
        """Yields pairs of features and targets for training.

        Yields:
        input_data (numpy.array): 1D array of size 'batch_size' containing features of a single input.
        output_data (numpy.array): 1D array containing the target values.
        """
        if self.total_size == 0:
            self.check_if_chunking()

        # If the data can be loaded in one go, don't skip any rows.
        if self.total_size <= self._ram_threshold:
            # Returns an array of the content from the CSV file.
            df_ = pd.read_csv(
                self._file_name,
                nrows=self._crop,
                header=0,  # skiprows=self._skip_rows
            )
            data_array = np.array(df_)
            inputs = data_array[:, 0]
            outputs = data_array[:, 1]

            maximum_batch_size = inputs.size - 2 * self._offset
            self.total_num_samples = maximum_batch_size
            if self._batch_size < 0:
                self._batch_size = maximum_batch_size

            indices = np.arange(maximum_batch_size)
            if self._shuffle:
                np.random.shuffle(indices)

            while True:
                for start_index in range(0, maximum_batch_size, self._batch_size):
                    splice = indices[start_index : start_index + self._batch_size]
                    data_ = [
                        inputs[index : index + 2 * self._offset + 1] for index in splice
                    ]
                    input_data = np.array(data_)
                    output_data = outputs[splice + self._offset].reshape(-1, 1)

                    yield input_data, output_data

        # Skip rows where needed to allow data to be loaded when there is not enough memory.
        if self.total_size >= self._ram_threshold:
            number_of_chunks = np.arange(self.total_size / self._chunk_size)
            if self._shuffle:
                np.random.shuffle(number_of_chunks)

            # Yield the data in sections.
            for index in number_of_chunks:
                data_array = np.array(
                    pd.read_csv(
                        self._file_name,
                        header=0,
                        nrows=self._crop,
                        skiprows=int(index) * self._chunk_size,
                    )
                )
                inputs = data_array[:, 0]
                outputs = data_array[:, 1]

                maximum_batch_size = inputs.size - 2 * self._offset
                self.total_num_samples = maximum_batch_size
                if self._batch_size < 0:
                    self._batch_size = maximum_batch_size

                indices = np.arange(maximum_batch_size)
                if self._shuffle:
                    np.random.shuffle(indices)

            while True:
                for start_index in range(0, maximum_batch_size, self._batch_size):
                    splice = indices[start_index : start_index + self._batch_size]
                    data_ = [
                        inputs[index : index + 2 * self._offset + 1] for index in splice
                    ]
                    input_data = np.array(data_)
                    output_data = outputs[splice + self._offset].reshape(-1, 1)
                    yield input_data, output_data


class TestSlidingWindowGenerator(object):
    """Yields features and targets for testing and validating a ConvNet.

    Parameters:
    _number_of_windows (int): The number of sliding windows to produce.
    _offset (int): The offset of the inferred value from the sliding window.
    _inputs (numpy.ndarray): The available testing / validation features.
    _targets (numpy.ndarray): The target values corresponding to _inputs.
    total_size (int): The total number of inputs.
    """

    def __init__(self, number_of_windows, inputs, targets, offset):
        self._number_of_windows = number_of_windows
        self._offset = offset
        self._inputs = inputs
        self._targets = targets
        self.total_size = len(inputs)

    def load_dataset(self):
        """Yields features and targets for testing and validating a ConvNet.

        Yields:
        input_data (numpy.array): An array of features to test / validate the network with.
        """
        self._inputs = self._inputs.flatten()
        max_number_of_windows = self._inputs.size - 2 * self._offset

        if self._number_of_windows < 0:
            self._number_of_windows = max_number_of_windows

        indices = np.arange(max_number_of_windows, dtype=int)
        for start_index in range(0, max_number_of_windows, self._number_of_windows):
            splice = indices[start_index : start_index + self._number_of_windows]
            input_data = np.array(
                [self._inputs[index : index + 2 * self._offset + 1] for index in splice]
            )
            target_data = self._targets[splice + self._offset].reshape(-1, 1)
            yield input_data, target_data
