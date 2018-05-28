#!/usr/bin/env python3

import os
import argparse
import functools
from glob import glob

import hooks
import dataset
import tensorflow as tf
from tensorflow.contrib.training import HParams
import components
from optimize_loss import optimize_loss
from tensorflow.python import debug as tfdbg

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--processed_data_dir', default='processed_data')
parser.add_argument('--model_dir', default='model_dir')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--device', default='gpu', type=str)

parser.add_argument('-B', '--batch_size', type=int, default=64)
parser.add_argument('-V', '--vocab_size', default=5000, type=int)
parser.add_argument('-T', '--max_seq_len', default=20, type=int)
parser.add_argument('-E', '--embed_size', default=100, type=int)
parser.add_argument('-S', '--state_size', default=256, type=int)

parser.add_argument('--l2_decay', default=0.0, type=float)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout_prob', default=0.5, type=float)
parser.add_argument('--learning_rate', default=1e-3, type=float)

parser.add_argument('--train_steps', default=8000, type=int)
parser.add_argument('--eval_steps', default=50, type=int)
parser.add_argument('--throttle_secs', default=30, type=int)
parser.add_argument('--steps_per_print', default=25, type=int)
parser.add_argument('--steps_per_ckpt', default=200, type=int)
args = parser.parse_args()


def l2_regularize(t, scale, name):
    if scale == 0.:
        return t
    t_reg = tf.multiply(scale, tf.nn.l2_loss(t))
    tf.add_to_collection(t_reg, tf.GraphKeys.REGULARIZATION_LOSSES)
    tf.summary.histogram(name, t_reg)
    return t_reg

def input_fn(hparams, mode):
    return dataset.Dataset(hparams.processed_data_dir, hparams).make_batch(mode)

def model_fn(features, labels, mode, params):
    print('=' * 50, '\n', 'MODEL_FN CALLED IN MODE', mode, '\n', '=' * 50)

    # Place any special preprocessing needed for PREDICT and/or serving here:
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_doc = features['x']
        table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=params.vocab_path,
            vocab_size=params.vocab_size,
            default_value=1)
        features = table.lookup(input_doc)

    # Load and build the embedding layer, initialized with
    # pre-trained GloVe embeddings.
    # Has shape (batch_size, max_seq_len, embed_size)
    with tf.device('/cpu:0'):
        embedded_features = components.glove_embed(
            features,
            embed_shape=(params.vocab_size, params.embed_size),
            vocab_path=params.vocab_path,
            projector_path=params.model_dir)

    with tf.device('/{}:0'.format(args.device)):
        # Ensure that `embedded_features` has rank 3 before executing any further.
        with tf.control_dependencies([tf.assert_rank(embedded_features, 3)]):
            _, final_state = components.deep_blstm(
                inputs=embedded_features,
                mode=mode,
                state_size=params.state_size,
                num_layers=params.num_layers,
                dropout_prob=params.dropout_prob)

        if params.l2_decay:
            l2_regularize(final_state, params.l2_decay, 'final_state_l2')

        # We project the final output state to obtain
        # the logits over each of our possible classes (labels).
        logits = tf.layers.Dense(
            params.num_labels,
            kernel_regularizer=functools.partial(
                l2_regularize, scale=params.l2_decay, name='logits_kernel_l2')
        )(final_state)

        with tf.variable_scope('predictions'):
            output_probs = tf.nn.softmax(logits, name='output_probs')
            preds = tf.argmax(output_probs, -1, name='preds')
            if params.debug:
                preds = tf.Print(
                    preds, [preds], 'Preds print: ',
                    first_n=10, summarize=5)

    # Create table for converting prediction index -> label.
    table = tf.contrib.lookup.index_to_string_table_from_file(params.labels_path)
    # Convert each prediction index to the corresponding label.
    preds_words = tf.identity(table.lookup(preds), 'preds_words')
    # For PREDICT mode, compute predicted label for each example in batch.
    if mode == tf.estimator.ModeKeys.PREDICT:
        pred_out = tf.estimator.export.PredictOutput({'preds_words': preds_words})
        return tf.estimator.EstimatorSpec(
            mode, predictions=preds_words,
            export_outputs={'export_outputs': pred_out},
            prediction_hooks=[hooks.FreezingHook(
                'preds_words', os.path.join(params.model_dir, 'frozen_model.pb'))],
            scaffold=tf.train.Scaffold(init_op=tf.tables_initializer()))

    with tf.device('/{}:0'.format(args.device)):
        # from tensorflow.contrib.layers import optimize_loss, l2_regularizer
        l2 = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) \
            if args.l2_decay else 0.0
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits) + l2
        train_op = optimize_loss(
            loss,
            learning_rate=params.learning_rate,
            clip_gradients=5.0,
            optimizer='Adam')

    accuracy = tf.metrics.accuracy(labels, preds, name='acc_op')
    tf.summary.scalar('acc', tf.identity(accuracy[1], 'acc_tensor'))
    tf.summary.scalar('loss', tf.identity(loss, 'loss_tensor'))
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('output_probs', output_probs)
    tf.summary.histogram('combined_rnn_state', final_state)

    if mode == tf.estimator.ModeKeys.TRAIN:
        ops_hook = hooks.CustomOpsHook([accuracy])

        def over_fifty(datum, tensor):
            """Returns True if there exists an entry in output_probs over 0.5.
            Args:
                datum: DebugTensorDatum
                tensor: dumped tensor value a `numpy.ndarray`
            """
            if 'predictions/output_probs' in datum.tensor_name:
                return tensor.max() > 0.5
            return False

        training_hooks = [ops_hook]
        if params.debug:
            debug_hook = tfdbg.LocalCLIDebugHook()
            debug_hook.add_tensor_filter('over_fifty', over_fifty)
            training_hooks.append(debug_hook)

        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            train_op=train_op,
            training_hooks=training_hooks)
    elif mode == tf.estimator.ModeKeys.EVAL:
        early_stopping_hook = hooks.EarlyStoppingHook(
            metric=accuracy,
            max_metric=0.98,
            patience=20000)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            evaluation_hooks=[early_stopping_hook],
            eval_metric_ops={'acc': accuracy})


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
                  os.path.join(args.processed_data_dir, 'labels.txt'),
                  overwrite=True)

    hparams = HParams(**vars(args))
    hparams.labels_path = os.path.join(args.processed_data_dir, 'labels.txt')
    with open(hparams.labels_path) as f:
        hparams.num_labels = len(f.readlines())
        print('Num labels =', hparams.num_labels)

    # Define the path to the file that we'll store our vocabulary in.
    # This file will have the same number of lines as our vocab_size.
    # Each line will contain a single word in our vocabulary, listed in
    # order of decreasing frequency seen in our training data.
    hparams.vocab_path = os.path.join(
        os.path.realpath(hparams.processed_data_dir),
        'vocab_{}.txt'.format(hparams.vocab_size))
    os.makedirs(os.path.dirname(hparams.vocab_path), exist_ok=True)

    # Data preparation: getting vocabulary and saving tfrecords format.
    if not tf.gfile.Exists(hparams.vocab_path):
        for mode in ['train', 'test']:
            data_file_path = glob('{}/*{}*'.format(hparams.data_dir, mode))[0]

            print('Extracting vocab, labels, and tokenized texts from data.')
            if mode == 'train':
                vocab, labels, texts = data_util.fit_and_extract(
                    data_file_path, hparams.vocab_size)
                print('Saving vocabulary to {}.'.format(hparams.vocab_path))
                with open(hparams.vocab_path, 'w+') as f:
                    f.write('\n'.join(vocab))
            else:
                _, labels, texts = data_util.fit_and_extract(
                    data_file_path, hparams.vocab_size)

            tfrecords_path = os.path.join(
                hparams.processed_data_dir,
                '{}_{}.tfrecords'.format(mode, hparams.vocab_size))
            print('Saving tfrecords to {}.'.format(tfrecords_path))
            tfrecords_util.save_tfrecords(
                out_path=tfrecords_path,
                labels=labels,
                texts=texts,
                vocab=vocab)

    def my_formatter(tag_to_tensor):
        res = ''
        for tag, tensor in tag_to_tensor.items():
            res += '  {}={:.2f}'.format(tag, tensor)
        return res

    logging_hook = tf.train.LoggingTensorHook({
        'loss': 'loss_tensor',
        'acc': 'acc_tensor'},
        every_n_iter=hparams.steps_per_print,
        formatter=my_formatter)
    profiler_hook = tf.train.ProfilerHook(
        save_steps=100,
        show_memory=True,
        output_dir=hparams.model_dir)
    # tensorboard_hook = tfdbg.TensorBoardDebugHook(
    #     grpc_debug_server_addresses='localhost:6064'
    # )

    run_config = tf.estimator.RunConfig(
        model_dir=hparams.model_dir,
        save_summary_steps=hparams.steps_per_ckpt,
        save_checkpoints_steps=hparams.steps_per_ckpt,
        keep_checkpoint_max=1,
        log_step_count_steps=10000,
        session_config=tf.ConfigProto(
            allow_soft_placement=True))
            # gpu_options=tf.GPUOptions(allow_growth=True)))
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        params=hparams)

    # Returns a callable (zero-argument function) that, when called, will
    # return a ServingInputReceiver. Our input features have the general shape
    # of (None, None), which means (unspecified batch size, unspecified max seq len),
    # which allows us to pass variable-length batches and sequences to our model
    # at serving time.
    #
    # In this simple case, we could actually implement this identically as
    # def receiver_fn():
    #     features = {'x': tf.placeholder(tf.int64, (None, None), 'x')}
    #     return tf.estimator.export.ServingInputReceiver(features, features.copy())
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={'x': tf.placeholder(
            dtype=tf.string,
            shape=[None, None],
            name='x')})


    try:
        tf.estimator.train_and_evaluate(
            estimator=classifier,
            train_spec=tf.estimator.TrainSpec(
                input_fn=lambda: input_fn(hparams, 'train'),
                hooks=[logging_hook, profiler_hook],
                max_steps=hparams.train_steps),
            eval_spec=tf.estimator.EvalSpec(
                input_fn=lambda: input_fn(hparams, 'test'),
                throttle_secs=hparams.throttle_secs,
                steps=hparams.eval_steps))
    except SystemExit:
        print('Early stopping raised SystemExit. '
              'Terminating train_and_evaluate.')
    except KeyboardInterrupt:
        print('Received KeyboardInterrupt. Exporting model then exiting.')
    classifier.export_savedmodel(
        serving_input_receiver_fn=receiver_fn,
        export_dir_base=os.path.join(hparams.model_dir, 'exports'))


if __name__ == '__main__':
    main()
