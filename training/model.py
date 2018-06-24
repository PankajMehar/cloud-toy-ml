"""
Cloud Toy ML

Toy training app to demonstrate machine learning in Google Cloud

Author:  Anshul Kharbanda
Created: 6 - 21 - 2018
"""
import tensorflow as tf

def make_classifier(config):
    """
    Returns configured classifier

    :param config: runtime configuration

    :return: configured classifier
    """
    # Using a linear classifier for computational ease
    return tf.estimator.LinearClassifier(
        feature_columns=[
            tf.feature_column.numeric_column('fixed acidity', type=tf.float32),
            tf.feature_column.numeric_column('volatile acidity', type=tf.float32),
            tf.feature_column.numeric_column('citric acid', type=tf.float32),
            tf.feature_column.numeric_column('redidual sugar', type=tf.float32),
            tf.feature_column.numeric_column('chlorides', type=tf.float32),
            tf.feature_column.numeric_column('free sulfur dioxide', type=tf.int32),
            tf.feature_column.numeric_column('total sulfur dioxide', type=tf.int32),
            tf.feature_column.numeric_column('density', type=tf.float32),
            tf.feature_column.numeric_column('pH', type=tf.float32),
            tf.feature_column.numeric_column('sulphates', type=tf.float32),
            tf.feature_column.numeric_column('alcohol', type=tf.float32)
        ],
        n_classes=11)
