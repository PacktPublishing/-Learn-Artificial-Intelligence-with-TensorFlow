#!/usr/bin/env python3

import argparse
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.training import HParams
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='model_dir')
parser.add_argument('-B', '--batch_size', type=int, default=64)
parser.add_argument('-S', '--state_size', default=256, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--train_steps', default=8000, type=int)
args = parser.parse_args()


# For versions before 1.7:
# tfe.enable_eager_execution()
tf.enable_eager_execution()


def demo():
    print('Running demo() function:')
    # Multiply two 2x2 matrices
    x = tf.matmul([[1, 2],
                   [3, 4]],
                  [[4, 5],
                   [6, 7]])
    # Add one to each element
    # (tf.add supports broadcasting)
    y = x + 1
    # Create a random random 5x3 matrix
    z = tf.random_uniform([5, 3])
    print('x:', x, sep='\n')
    print('\n y = x + 1:', y, sep='\n')
    print('\nResult of tf.random_uniform:', z, sep='\n')


class SimpleNN(tfe.Network):
    """Reasons for wrapping with tfe.Network:
    - Gives us the `variables` property.

    """

    def __init__(self, params):
        super(SimpleNN, self).__init__()
        self.params = params
        self.hidden_layers = [
            self.track_layer(tf.layers.Dense(params.state_size, activation=tf.nn.relu))
            for _ in range(params.num_layers)]
        self.output_layer = self.track_layer(tf.layers.Dense(params.num_classes, name='output_layer'))
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.grads_fn = tfe.implicit_gradients(self.loss_fn)

    def call(self, input):
        x = input
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def loss_fn(self, X, y):
        logits = self(X)
        correct = tf.nn.in_top_k(tf.to_float(logits), y, k=1)
        self.num_correct = tf.reduce_sum(tf.to_int32(correct))
        onehot_labels = tf.one_hot(y, self.params.num_classes, name='onehot_labels')
        return tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits)

    def run_train_step(self, X, y):
        self.optimizer.apply_gradients(self.grads_fn(
            tf.constant(X),
            tf.constant(y)))

    def run_train_epoch(self, dataset):
        num_correct_total = 0
        for (x, y) in tfe.Iterator(dataset):
            self.run_train_step(x, y)
            num_correct_total += self.num_correct
        return num_correct_total


def main():

    hparams = HParams(**vars(args))
    hparams.hidden_size = 512
    hparams.num_classes = 10
    hparams.num_features = 100
    hparams.num_epochs = 200
    hparams.num_samples = 1234

    dataset = tf.data.Dataset.from_tensor_slices((
        np.random.random(size=(hparams.num_samples, hparams.num_features)),
        np.random.randint(0, hparams.num_classes, size=hparams.num_samples)))
    dataset = dataset.batch(hparams.batch_size)

    print('\n\nRunning SimpleNN model.')
    model = SimpleNN(hparams)
    for epoch_idx in range(hparams.num_epochs):
        num_correct_total = model.run_train_epoch(dataset)
        if epoch_idx % 5 == 0:
            print('Epoch {}: accuracy={:.3%}'.format(
                epoch_idx, float(num_correct_total) / hparams.num_samples))


if __name__ == '__main__':
    demo()
    main()
