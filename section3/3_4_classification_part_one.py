#!/usr/bin/env python3

import os
import argparse
from glob import glob
from pprint import pprint
import numpy as np

import tensorflow as tf
from tensorflow.contrib.training import HParams
tf.logging.set_verbosity(tf.logging.INFO)

import dataset
import components
from util import glove, newsgroups, tfrecords

parser = argparse.ArgumentParser(description="Train a RNN on 20NG dataset.")
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_dir', default='data')
parser.add_argument('--processed_data_dir', default='processed_data')
parser.add_argument('--model_dir', default='model_dir')

parser.add_argument('--predict_only', action='store_true', default=False)

parser.add_argument('-V', '--vocab_size', default=5000)
parser.add_argument('-T', '--max_seq_len', default=20)
parser.add_argument('-E', '--embed_size', default=100)

parser.add_argument('--hidden_size', default=256)
parser.add_argument('--num_iter', default=10, type=int)
parser.add_argument('--train_steps', default=1000, type=int)
parser.add_argument('--eval_steps', default=200, type=int)
parser.add_argument('--steps_per_print', default=200, type=int)
args = parser.parse_args()


def input_fn(hparams, mode):
    with tf.variable_scope('input_fn'):
        return dataset.Dataset(hparams.processed_data_dir, hparams).make_batch(mode)


def model_fn(features, labels, mode, params):
    # 20 Newsgroups dataset has 20 unique labels.
    num_classes = 20

    # Load and build the embedding layer, initialized with
    # pre-trained GloVe embeddings.
    # Has shape (batch_size, max_seq_len, embed_size)
    embedded_features = components.glove_embed(
        features,
        embed_shape=(params.vocab_size, params.embed_size),
        vocabulary=params.vocab)

    # Define LSTMCell with state size of 128.
    cell = tf.nn.rnn_cell.LSTMCell(128)

    # Use tf.nn.dynamic_rnn for efficient computation.
    # It utilizes TensorFlow's tf.while_loop to repeatedly
    # call cell(...) over the sequential embedded_features.
    #
    # Returns:
    #   the full output sequence as `outputs` tensor,
    #       which has shape (batch_size, max_seq_len, 128)
    #   the final LSTMStateTuple(c_final, h_final), where both
    #   c_final and h_final have shape (batch_size, 128)
    outputs, state = tf.nn.dynamic_rnn(
        cell=cell, inputs=embedded_features, dtype=tf.float32)

    # We project the final output state to obtain
    # the logits over each of our possible classes (labels).
    logits = tf.layers.Dense(num_classes)(state.h)

    # For PREDICT mode, compute predicted label for each example in batch.
    if mode == tf.estimator.ModeKeys.PREDICT:
        output_probs = tf.nn.softmax(logits)
        preds = tf.argmax(output_probs, -1)
        # Create table for converting prediction index -> label.
        table = tf.contrib.lookup.index_to_string_table_from_tensor(params.vocab)
        # Convert each prediction index to the corresponding label.
        preds_words = table.lookup(preds)
        return tf.estimator.EstimatorSpec(
            mode, predictions=preds_words)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    with tf.variable_scope('accuracy', values=[labels, logits]):
        output_probs = tf.nn.softmax(logits)
        num_correct = tf.to_float(tf.nn.in_top_k(
            output_probs, tf.to_int64(labels), 1))
        accuracy = tf.reduce_mean(num_correct)

    if mode == tf.estimator.ModeKeys.EVAL:
        preds = tf.argmax(output_probs, -1)
        eval_metric_ops = {'acc': tf.metrics.accuracy(labels, preds)}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss, global_step=tf.train.get_or_create_global_step(),
        learning_rate=1e-3,
        optimizer='Adam')

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('acc', accuracy)
        logging_hook = tf.train.LoggingTensorHook({
            'step': tf.train.get_global_step(),
            'loss': loss,
            'acc': accuracy
        }, every_n_iter=params.steps_per_print)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[logging_hook])


def main():
    # tf.estimator will load/reuse anything found in its model_dir, so
    # we make sure to clear its contents before every training run.
    # For predictions, however, we of course want to load the previously
    # trained model from disk.
    if tf.gfile.Exists(args.model_dir) and not args.predict_only:
        tf.gfile.DeleteRecursively(args.model_dir)
    tf.gfile.MakeDirs(args.model_dir)
    tf.gfile.MakeDirs(args.processed_data_dir)
    tf.gfile.Copy(os.path.join(args.data_dir, 'labels.txt'),
                  os.path.join(args.processed_data_dir, 'labels.txt'), overwrite=True)

    hparams = HParams(**vars(args))

    # Define the path to the file that we'll store our vocabulary in.
    # This file will have the same number of lines as our vocab_size.
    # Each line will contain a single word in our vocabulary, listed in
    # order of decreasing frequency seen in our training data.
    vocab_path = os.path.join(hparams.processed_data_dir, 'vocab.txt')

    # Data preparation: getting vocabulary and saving tfrecords format.
    if not tf.gfile.Exists(vocab_path):
        for mode in ['train', 'test']:
            data_file_path = os.path.join(
                hparams.data_dir, '20ng-{}-all-terms.txt'.format(mode))

            print('Extracting vocab, labels, and tokenized texts from data.')
            if mode == 'train':
                vocab, labels, texts = newsgroups.fit_and_extract(
                    data_file_path, hparams.vocab_size)
                print('Saving vocabulary to {}.'.format(vocab_path))
                with open(vocab_path, 'w+') as f:
                    f.write('\n'.join(vocab))
            else:
                _, labels, texts = newsgroups.fit_and_extract(
                    data_file_path, hparams.vocab_size)

            tfrecords_path = os.path.join(
                hparams.processed_data_dir, '20ng_simple_{}.tfrecords'.format(mode))
            print('Saving tfrecords to {}.'.format(tfrecords_path))
            tfrecords.save_tfrecords(
                out_path=tfrecords_path,
                labels=labels,
                texts=texts,
                vocab=vocab)
    else:
        print('Reading existing vocabulary from {}.'.format(vocab_path))
        with open(vocab_path) as f:
            vocab = [l.strip() for l in f.readlines()]

    hparams.vocab = vocab
    print('Creating classifier.')
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=hparams.model_dir,
        config=tf.estimator.RunConfig(
            log_step_count_steps=10000,
        ),
        params=hparams)

    if not args.predict_only:
        for i in range(hparams.num_iter):
            classifier.train(
                input_fn=lambda: input_fn(hparams, 'train'),
                steps=hparams.train_steps)
            classifier.evaluate(
                input_fn=lambda: input_fn(hparams, 'test'),
                steps=hparams.eval_steps)


if __name__ == '__main__':
    main()
