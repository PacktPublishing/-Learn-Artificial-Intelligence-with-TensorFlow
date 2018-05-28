#!/usr/bin/env python3

import sys
import os
import argparse
from glob import glob
from pprint import pprint
import numpy as np

import hooks
import dataset
import components
from util import glove, newsgroups, tfrecords

import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, LSTMCell, MultiRNNCell, LSTMStateTuple
from tensorflow.contrib.training import HParams

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description="Train a RNN on 20NG dataset.")
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--processed_data_dir', default='processed_data')
parser.add_argument('--model_dir', default='model_dir')

parser.add_argument('-B', '--batch_size', type=int, default=64)
parser.add_argument('-V', '--vocab_size', default=5000, type=int)
parser.add_argument('-T', '--max_seq_len', default=20, type=int)
parser.add_argument('-E', '--embed_size', default=100, type=int)
parser.add_argument('-S', '--state_size', default=256, type=int)

parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_iter', default=50, type=int)
parser.add_argument('--train_steps', default=500, type=int)
parser.add_argument('--eval_steps', default=100, type=int)
parser.add_argument('--steps_per_print', default=200, type=int)
args = parser.parse_args()


def input_fn(hparams, mode):
    with tf.variable_scope('input_fn'):
        return dataset.Dataset(hparams.processed_data_dir, hparams).make_batch(mode)


def model_fn(features, labels, mode, params):

    with tf.variable_scope('model_fn', values=[features, labels]):
        # 20 Newsgroups dataset has 20 unique labels.
        num_classes = 20

        # Load and build the embedding layer, initialized with
        # pre-trained GloVe embeddings.
        # Has shape (batch_size, max_seq_len, embed_size)
        embedded_features, emb_hook = components.glove_embed(
            features,
            embed_shape=(params.vocab_size, params.embed_size),
            vocabulary=params.vocab)

        with tf.variable_scope('deep_blstm'):
            def deep_lstm():
                if mode == tf.estimator.ModeKeys.TRAIN:
                    return MultiRNNCell([
                        DropoutWrapper(LSTMCell(params.state_size), state_keep_prob=0.5)
                        for _ in range(params.num_layers)])
                else:
                    return MultiRNNCell([
                        LSTMCell(params.state_size) for _ in range(params.num_layers)])

            cell_fw = deep_lstm()
            cell_bw = deep_lstm()

            # Use tf.nn.bidirectional_dynamic_rnn for efficient computation.
            # It utilizes TensorFlow's tf.while_loop to repeatedly
            # call cell(...) over the sequential embedded_features.
            #
            # Returns:
            #   outputs: tuple (output_fw, output_bw) containing fw and bw rnn output Tensor,
            #       where each has shape (batch size, max_seq_len, cell.output_size)
            #   output_states: tuple (output_state_fw, output_state_bw) containing fw and bw
            #       final states of bidirectional rnn.
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedded_features,
                dtype=tf.float32)

            # Each output_state is a tuple of length num_layers,
            # and the i'th element is an LSTMStateTuple, representing the
            # final state of the i'th layer.
            output_state_fw, output_state_bw = output_states

            def concat_lstms(lstms):
                """Merges list of LSTMStateTuple into a single LSTMStateTuple."""
                return LSTMStateTuple(
                    c=tf.concat([lstm.c for lstm in lstms], axis=-1),
                    h=tf.concat([lstm.h for lstm in lstms], axis=-1))

            # First, concatentate each output_state LSTMStateTuple, such that the
            # result is a single LSTMStatTuple for each (instead of num_layers many).
            output_state_fw = concat_lstms(output_state_fw)
            output_state_bw = concat_lstms(output_state_bw)

            # Then, combine the forward and backward output states.
            combined_final_state = tf.concat([
                output_state_fw.h, output_state_bw.h], axis=-1)

        # We project the final output state to obtain
        # the logits over each of our possible classes (labels).
        logits = tf.layers.Dense(num_classes)(combined_final_state)

        with tf.variable_scope('predictions'):
            output_probs = tf.nn.softmax(logits)
            preds = tf.argmax(output_probs, -1)

        # For PREDICT mode, compute predicted label for each example in batch.
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Create table for converting prediction index -> label.
            table = tf.contrib.lookup.index_to_string_table_from_tensor(params.vocab)
            # Convert each prediction index to the corresponding label.
            preds_words = table.lookup(preds)
            return tf.estimator.EstimatorSpec(
                mode, predictions=preds_words)

        with tf.variable_scope('train_and_eval', values=[labels, logits]):
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits)
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss, global_step=tf.train.get_or_create_global_step(),
                learning_rate=1e-3,
                clip_gradients=5.0,
                optimizer='Adam')

        with tf.variable_scope('metrics', values=[labels, preds]):
            accuracy = tf.metrics.accuracy(labels, preds, name='acc_op')
            metrics = {'acc': accuracy}
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.histogram('output_probs', output_probs)
            tf.summary.histogram('combined_rnn_state', combined_final_state)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        if mode == tf.estimator.ModeKeys.TRAIN:

            print('PDD:', params.processed_data_dir)
            vocab_path = os.path.join(
                os.path.realpath(params.processed_data_dir),
                'vocab_{}.txt'.format(params.vocab_size))
            print('VP:', vocab_path)

            config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            embz = config.embeddings.add()
            embz.tensor_name = emb_hook._embed_tensor.name
            embz.metadata_path = vocab_path
            writer = tf.summary.FileWriter(params.model_dir)
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

            def my_formatter(tag_to_tensor):
                res = ''
                for tag, tensor in tag_to_tensor.items():
                    res += '  {}={:.2f}'.format(tag, tensor)
                return res

            logging_hook = tf.train.LoggingTensorHook({
                'step': tf.train.get_global_step(),
                'loss': loss,
                'acc': accuracy[0]
            }, every_n_iter=params.steps_per_print,
                formatter=my_formatter)
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
    if tf.gfile.Exists(args.model_dir):
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
    vocab_path = os.path.join(hparams.processed_data_dir,
                              'vocab_{}.txt'.format(hparams.vocab_size))

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
                hparams.processed_data_dir,
                '20ng_advanced_{}_{}.tfrecords'.format(mode, hparams.vocab_size))
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
            save_summary_steps=100,
            save_checkpoints_steps=500,
            log_step_count_steps=10000,
        ),
        params=hparams)

    for i in range(hparams.num_iter):
        classifier.train(
            input_fn=lambda: input_fn(hparams, 'train'),
            hooks=[tf.train.ProfilerHook(save_steps=100, output_dir=hparams.model_dir)],
            steps=hparams.train_steps)
        classifier.evaluate(
            input_fn=lambda: input_fn(hparams, 'test'),
            steps=hparams.eval_steps)


if __name__ == '__main__':
    main()
