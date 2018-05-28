#!/usr/bin/env python3

"""
File: section3/3_1_embeddings.py
Author: Brandon McKinzie
Description: Simple autoencoder model to illustrate embeddings with TensorFlow.
    1. Build {word => counts} dictionary from training dataset.
    2. Store as vocabulary file (always useful to have on-hand).
"""

import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.contrib.training import HParams

import dataset
import components
from util import newsgroups, tfrecords

tf.logging.set_verbosity(tf.logging.INFO)

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
args = parser.parse_args()


def input_fn(data_dir, params):
    text_batch, _ = dataset.Dataset(data_dir, params).make_batch(params.batch_size)
    return text_batch, text_batch


def model_fn(features, labels, mode, params):
    """
    Args:
        features: Tensor with shape (batch_size, max_seq_len) with dtype int.
        labels: same as features, since we are training an autoencoder.
        params: tf.contrib.HParams object containing...
            embed_size, vocab_size
    """
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['x']

    # Load and build the embedding layer, initialized with pre-trained
    # GloVe embeddings.
    embedded_features = components.glove_embed(
        features,
        embed_shape=(params.vocab_size, params.embed_size),
        vocabulary=params.vocab)

    hidden_layer = tf.layers.Dense(
        params.hidden_size, activation=tf.tanh)(embedded_features)
    logits = tf.layers.Dense(params.vocab_size)(hidden_layer)

    if mode == tf.estimator.ModeKeys.PREDICT:
        output_probs = tf.nn.softmax(logits)
        preds = tf.argmax(output_probs, -1)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(params.vocab)
        preds_words = table.lookup(preds)
        return tf.estimator.EstimatorSpec(
            mode, predictions=preds_words)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss, global_step=tf.train.get_or_create_global_step(),
        learning_rate=0.01,
        optimizer='Adam')

    with tf.variable_scope('accuracy', values=[labels, logits]):
        flattened_logits = tf.reshape(
            logits, [params.batch_size * params.max_seq_len, -1])
        flattened_labels = tf.reshape(
            labels, [params.batch_size * params.max_seq_len,])
        output_probs = tf.nn.softmax(flattened_logits)
        num_correct = tf.to_float(tf.nn.in_top_k(
            output_probs, tf.to_int64(flattened_labels), 1))
        accuracy = tf.reduce_mean(num_correct)

    tf.summary.scalar('accuracy', accuracy)
    if mode == tf.estimator.ModeKeys.TRAIN:
        logging_hook = tf.train.LoggingTensorHook({
            'loss': loss,
            'acc': accuracy
        }, every_n_iter=100)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[logging_hook])


def main(_):

    # tf.estimator will load/reuse anything found in its model_dir, so
    # we make sure to clear its contents before every training run.
    # For predictions, however, we of course want to load the previously
    # trained model from disk.
    if tf.gfile.Exists(args.model_dir) and not args.predict_only:
        tf.gfile.DeleteRecursively(args.model_dir)
    tf.gfile.MakeDirs(args.model_dir)

    hparams = HParams(**vars(args))

    # We will use the 20 newsgroups dataset to train our model.
    # Note that we won't be using the labels, since our model is simply
    # learning to reconstruct its inputs as its output.
    train_file_path = os.path.join(hparams.data_dir, '20ng-train-all-terms.txt')

    # Define the path to the file that we'll store our vocabulary in.
    # This file will have the same number of lines as our vocab_size.
    # Each line will contain a single word in our vocabulary, listed in
    # order of decreasing frequency seen in our training data.
    vocab_path = os.path.join(hparams.processed_data_dir, 'vocab.txt')

    # Data preparation: getting vocabulary and saving tfrecords format.
    if not tf.gfile.Exists(vocab_path):
        print('Extracting vocab, labels, and tokenized texts from data.')
        vocab, labels, texts = newsgroups.fit_and_extract(
            train_file_path, hparams.vocab_size)
        print('Saving vocabulary to {}.'.format(vocab_path))
        with open(vocab_path, 'w+') as f:
            f.write('\n'.join(vocab))

        tfrecords_path = os.path.join(hparams.processed_data_dir, 'embed.tfrecords')
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
    print('Creating autoencoder.')
    autoencoder = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=hparams.model_dir,
        config=tf.estimator.RunConfig(log_step_count_steps=10000),
        params=hparams)

    if not args.predict_only:
        print('Training autoencoder.')
        autoencoder.train(
            input_fn=lambda: input_fn(hparams.processed_data_dir, hparams),
            steps=1000)

    sample_sentences = [
        'i like dogs',
        'i am a test sentence',
        'TensorFlow is a fun library to use']
    pred_inputs = []
    for sent in sample_sentences:
        token_ids = [vocab.index(w)
                     for w in sent.split()[:args.max_seq_len]
                     if w in vocab]
        # Pad if necessary.
        if len(token_ids) < args.max_seq_len:
            token_ids.extend([0] * (args.max_seq_len - len(token_ids)))
        pred_inputs.append(token_ids)

    pred_inp_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.asarray(pred_inputs)}, shuffle=False)
    predictions = autoencoder.predict(input_fn=pred_inp_fn)

    print('Sample predictions:')
    for i, prediction in enumerate(predictions):
        clean_prediction = ' '.join([tok.decode() for tok in prediction if tok != b'_UNK'])
        print('\nExpected:', sample_sentences[i], sep='\t')
        print('Actual:  ', clean_prediction, sep='\t')


if __name__ == '__main__':
    tf.app.run()

