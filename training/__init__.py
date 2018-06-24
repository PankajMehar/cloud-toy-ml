"""
Cloud Toy ML

Toy training app to demonstrate machine learning in Google Cloud

Author:  Anshul Kharbanda
Created: 6 - 21 - 2018
"""
import tensorflow as tf

# Feature columns and class number
FEATURE_COLUMNS = [
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
]
NUMBER_OF_CLASSES = 11
