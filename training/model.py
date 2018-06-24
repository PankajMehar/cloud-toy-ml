"""
Cloud Toy ML

Toy training app to demonstrate machine learning in Google Cloud

Author:  Anshul Kharbanda
Created: 6 - 21 - 2018
"""
import tensorflow as tf
from . import FEATURE_COLUMNS, NUMBER_OF_CLASSES

def make_classifier(config):
    """
    Returns configured classifier

    :param config: runtime configuration

    :return: configured classifier
    """
    # Using a linear classifier for computational ease
    return tf.estimator.LinearClassifier(
        feature_columns=FEATURE_COLUMNS,
        n_classes=NUMBER_OF_CLASSES)
