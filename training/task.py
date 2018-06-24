"""
Cloud Toy ML

Toy training app to demonstrate machine learning in Google Cloud

Author:  Anshul Kharbanda
Created: 6 - 21 - 2018
"""
import tensorflow as tf
import argparse as ap
from .serving import serving_input_fn
from .model import make_classifier
from .data import make_input_fn

def run_train_job(hparams):
    """
    Runs training job. Trains the model, evaluates continuously,
    and exports the model.

    :param hparams: hyperparameters for training and evaluating
    """
    # We use a final exporter to handle exporting the model
    # This will export the final version or checkpoint of the model after training
    # The serving input function returns a tensor that handles input
    exporter = tf.estimator.FinalExporter('cloud-toy-ml', serving_input_fn)

    # Runtime configuration
    config = tf.estimator.RunConfig()
    config.replace(model_dir=hparams.job_dir)

    # The model itself!
    estimator = make_classifier(config)

    # Automatically train, evaluate and export the model using the given
    # hyperparameters
    #
    # Tensorflow provides the train_and_evaluate function which takes a train
    # spec and an eval spec along with the estimator and does all this for you

    # Training spec
    train_spec = tf.estimator.TrainSpec(
        make_input_fn(
            filename=hparams.train_filename,
            batch_size=hparams.train_batch_size,
            epochs=hparams.train_epochs,
            shuffle=True),
        max_steps=hparams.train_max_steps)

    # Evaluation spec
    eval_spec = tf.estimator.EvalSpec(
        make_input_fn(
            filename=hparams.eval_filename,
            batch_size=hparams.eval_batch_size,
            epochs=None,
            shuffle=False),
        exporters=[exporter],
        name='cloud-toy-ml')

    # Call routine
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def parse_arguments():
    """
    Parses command line arguments

    :return: parse argument dict
    """
    parser = ap.ArgumentParser()
    parser.add_argument('--job-dir',
        help='Location for exporting checkpoints and saving models',
        required=True)
    parser.add_argument('--train-filename',
        help='Filename for training data',
        required=True)
    parser.add_argument('--eval-filename',
        help='Filename for evaluation data',
        required=True)
    parser.add_argument('--train-batch-size',
        help='Batch size for training',
        type=int,
        default=100)
    parser.add_argument('--train-epochs',
        help='Epochs for training',
        type=int,
        default=1000)
    parser.add_argument('--train-max-steps',
        help='Max steps for training (if none, will train until all training epochs have completed)',
        type=int)
    parser.add_argument('--eval-batch-size',
        help='Batch size for evaluating',
        type=int,
        default=100)
    return parser.parse_args()

def main():
    """
    Main function
    """
    args = parse_arguments()
    hparams = tf.contrib.training.Hparams(**args.__dict__)
    run_train_job(hparams)

if __name__ == '__main__':
    main()
