"""
Cloud Toy ML

Toy training app to demonstrate machine learning in Google Cloud

Author:  Anshul Kharbanda
Created: 6 - 21 - 2018
"""
import tensorflow as tf

# The names of the input feature columns
CSV_COLUMN_NAMES = [
    'fixed acidity',
	'volatile acidity',
	'citric acid',
	'residual sugar',
	'chlorides',
	'free sulfur dioxide',
	'total sulfur dioxide',
	'density',
	'pH',
	'sulphates',
	'alcohol',
    'quality'
]
LABEL_COLUMN_NAME = 'quality'

# This dataset uses semicolons
CSV_DELIM = ';'

# Default csv columns
CSV_COLUMN_DEFAULTS = [
    [0.0], # fixed acidity
    [0.0], # volatile acidity
    [0.0], # citric acid
    [0.0], # residual sugar
    [0.0], # chlorides
    [0],   # free sulfur dioxide
    [0],   # total sulfur dioxide
    [0.0], # density
    [0.0], # pH
    [0.0], # sulphates
    [0.0], # alcohol
    [0]  # quality
]

def make_input_fn(filename, batch_size=100, epochs=None, shuffle=True):
    """
    Creates a data input function with the given parameters.

    This function will create a dataset from the csv data in the given filename,
    shuffle the data if shuffle is true, repeat the set for the given number of
    epochs (if epochs is not None) and break it into batches of the given batch
    size

    :param filename: the name of the csv file to parse data from
    :param batch_size: the batch size
    :param epochs: the number of epochs to repeat for (if None, repeat indefinitely)
    :param shuffle: if True, the data is shuffled

    :return: data input function
    """
    def _parse_csv(csv_row):
        """
        Parse csv row into feature and label tensor
        """
        # Parse columns
        columns = tf.decode_csv(csv_row, CSV_COLUMN_DEFAULTS, CSV_DELIM)

        # Extract features and labels
        features = dict(zip(CSV_COLUMN_NAMES, columns))
        labels = features.pop(LABEL_COLUMN_NAME)

        # Return features and labels
        return features, labels

    def _input_fn():
        """
        Input data function
        """
        # Parse csv into dataset of features and labels
        dataset = tf.data.TextLineDataset(filename).skip(1).map(_parse_csv)

        # Shuffle, repeat, and batch dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)

        # Return dataset
        return dataset

    # Return input function
    return _input_fn
