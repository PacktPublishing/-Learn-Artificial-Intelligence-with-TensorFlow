"""
File: templates/estimator_template.py
Author: Brandon McKinzie
"""

import tensorflow as tf


def input_fn():
    """Estimators use notion of input_fn. We can use tf.data API to make
    such an input_fn.

    We call this manually when specifying `lambda: input_fn(...)` as argument
    to any of classifier.{train, evaluate, predict}.

    Returns:
        feature_cols: A dict containing key/value pairs that map feature column
            names to `Tensor`s (or `SparseTensor`s) containing the corresponding
            feature data.
        labels: A `Tensor` containing your label (target) values (the values
            your model aims to predict).
    """
    pass


def model_fn(features, labels, mode, params):
    """Required when making a custom Estimator, and passed as the first argument of
    tf.estimator.Estimator(...).

    Logic to do the following:
        1. Configure the model via TensorFlow operations
        2. Define the loss function for training/evaluation
        3. Define the training operation/optimizer
        4. Generate predictions
        5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object

    Args:
        features (required): A dict containing the features passed to the model via input_fn.
        labels (required): A Tensor containing the labels passed to the model via input_fn. Will be empty for predict() calls, as these are the values the model will infer.
        mode (required): One of the following tf.estimator.ModeKeys string values, which indicate
            the context in which the model_fn was invoked:
            - TRAIN: invoked in training mode, namely via a train() call.
            - EVAL: invoked in evaluation mode, namely via an evaluate() call.
            - PREDICT: invoked in predict mode, namely via a predict() call.
        params (optional): hyperparamer dictionary.

    Returns:
        mode (required). The mode in which the model was run. Typically, you will
            return the mode argument of the model_fn here.
        predictions (required in PREDICT mode). A dict that maps key names of your
            choice to Tensors containing the predictions from the model, e.g.:
            `predictions = {"results": tensor_of_predictions}`
            In PREDICT mode, the dict that you return in EstimatorSpec will
            then be returned by predict(), so you can construct it in the format
            in which you'd like to consume it.
        loss (required in EVAL and TRAIN mode). A Tensor containing a scalar loss value:
            the output of the model's loss function calculated over all the input examples.
            This is used in TRAIN mode for error handling and logging, and is
            automatically included as a metric in EVAL mode.
        train_op (required only in TRAIN mode). An Op that runs one step of training.
        eval_metric_ops (optional). A dict of name/value pairs specifying the metrics
            that will be calculated when the model runs in EVAL mode. The name is a
            label of your choice for the metric, and the value is the result of your
            metric calculation. The tf.metrics module provides predefined functions for
            a variety of common metrics. The following eval_metric_ops contains an
            "accuracy" metric calculated using tf.metrics.accuracy:
            `eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels, predictions) }`
            If you do not specify eval_metric_ops, only loss will be calculated during evaluation.
    """
    # return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
    pass
