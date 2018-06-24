"""
Cloud Toy ML

Toy training app to demonstrate machine learning in Google Cloud

Author:  Anshul Kharbanda
Created: 6 - 21 - 2018
"""
import tensorflow as tf
from . import FEATURE_COLUMNS

def serving_input_fn():
    """
    Serving input reciever function

    :return: serving input reciever
    """
    # Inputs dict
    inputs = {
        'fixed acidity': tf.placeholder(shape=[None], dtype=tf.float32),
        'volatile acidity': tf.placeholder(shape=[None], dtype=tf.float32),
        'citric acid': tf.placeholder(shape=[None], dtype=tf.float32),
        'redidual sugar': tf.placeholder(shape=[None], dtype=tf.float32),
        'chlorides': tf.placeholder(shape=[None], dtype=tf.float32),
        'free sulfur dioxide': tf.placeholder(shape=[None], dtype=tf.int32),
        'total sulfur dioxide': tf.placeholder(shape=[None], dtype=tf.int32),
        'density': tf.placeholder(shape=[None], dtype=tf.float32),
        'pH': tf.placeholder(shape=[None], dtype=tf.float32),
        'sulphates': tf.placeholder(shape=[None], dtype=tf.float32),
        'alcohol': tf.placeholder(shape=[None], dtype=tf.float32)
    }

    # Return serving input reciever
    return tf.estimator.export.ServingInputReciever(inputs, inputs)
