"""
File: section3/util/hooks.py
Author: Brandon McKinzie
"""

import sys
import tensorflow as tf


class EarlyStoppingHook(tf.train.SessionRunHook):
    """Custom SessionRunHook that will terminate training when accuracy
    is found above some threshold.
    N.B. Relies on existense of an 'acc_metric' collection in the default
    graph.
    """

    def __init__(self, max_acc=0.99):
        """
        Args:
            acc: `Tensor` in the current graph that will contain the updated
                accuracy value after each session run call.
            max_acc: (float) largest permissible accuracy.
        """
        self._acc_tensor = None
        self._acc_op = None
        self._max_acc = max_acc

    def before_run(self, run_context):
        if tf.get_collection('acc_metric'):
            self._acc_tensor, self._acc_op = tf.get_collection('acc_metric')
            return tf.train.SessionRunArgs([self._acc_tensor, self._acc_op])
        else:
            return tf.train.SessionRunArgs()

    def after_run(self, run_context, run_values):
        if not run_values.results:
            return

        if run_values.results[0] > self._max_acc:
            tf.logging.info(
                'Early stopping -- Accuracy {:.3f} above threshold '
                'of {}.\n'.format(run_values.results[0], self._max_acc))
            sys.exit()


class EmbeddingVisualizerHook(tf.train.SessionRunHook):
    def __init__(self, embed_tensor):
        super(EmbeddingVisualizerHook, self).__init__()
        self._embed_tensor = embed_tensor

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=self._embed_tensor)

    def after_run(self, run_context, run_values):
        self._embeddings[0].extend(run_values[0][0])
        self._embeddings[1].extend(run_values[0][1])

    def get_embeddings(self):
        return {
            'values': self._embeddings[0],
            'labels': self._embeddings[1]}
