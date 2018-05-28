#!/usr/bin/env python3

"""
File: section3/cifar10_estimator.py
Author: Brandon McKinzie
Description: custom tf.estimator.Estimator implemention of the CNN.
             This is the main file that should be run. It uses `cifar10_input`
             to get the data and build the input pipeline, and uses
             `cifar10_model` to link the logits, loss, and training operations
             together in the custom estimator's `model_fn`.
"""

import os
import argparse
from glob import glob
from pprint import pprint

import cifar10_model
import cifar10_input
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 dataset.")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_dir', default='data/cifar')
parser.add_argument('--model_dir', default='models/cifar')
args = parser.parse_args()

# CIFAR-10 consists of 50K training examples and 10K eval examples.
# Each image has size 32x32 (and depth 3 for RGB).
CIFAR_TRAIN_SIZE = 50000
CIFAR_EVAL_SIZE = 10000
CIFAR_IMAGE_SIZE = 32


def input_fn(data_dir, batch_size):
    filenames = glob(os.path.join(data_dir, 'cifar-10-batches-bin', 'data_batch_*.bin'))
    pprint(filenames)

    depth = 3
    height = width = CIFAR_IMAGE_SIZE
    label_bytes = 1
    image_bytes = height * width * depth
    # Every record consists of a label followed by the image,
    # with a fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    def decode_line(value):
        """Additional processing to perform on each line in dataset."""
        record_bytes = tf.decode_raw(value, tf.uint8)
        # The first bytes represent the label, which we convert from uint8->int32.
        label = tf.to_int32(tf.strided_slice(record_bytes, [0], [label_bytes]))
        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
            [depth, height, width])
        # Convert from [depth, height, width] to [height, width, depth].
        uint8image = tf.transpose(depth_major, [1, 2, 0])
        reshaped_image = tf.cast(uint8image, tf.float32)
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(reshaped_image)
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)
        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        label.set_shape([1])
        return float_image, label

    # Repeat infinitely.
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes).repeat()
    dataset = dataset.map(decode_line, num_parallel_calls=batch_size)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(CIFAR_TRAIN_SIZE * min_fraction_of_examples_in_queue)
    dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    # Ensure we don't have any shape dimensions equal to None...
    image_batch.set_shape([batch_size, height, width, 3])
    label_batch = tf.squeeze(label_batch)
    return image_batch, label_batch


def model_fn(features, labels, mode, params):
    logits = cifar10_model.inference(
        image_batch=features,
        batch_size=params.get('batch_size'))
    loss = cifar10_model.loss(logits, labels)
    train_op = cifar10_model.train(loss, batch_size=params.get('batch_size'))

    if mode == tf.estimator.ModeKeys.TRAIN:
        logging_hook = tf.train.LoggingTensorHook({'loss': loss}, every_n_iter=1000)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[logging_hook])


def main():
    # Ensure data_dir has dataset and model_dir is cleared before training.
    cifar10_input.maybe_download_and_extract(data_dir=args.data_dir)
    if tf.gfile.Exists(args.model_dir):
        tf.gfile.DeleteRecursively(args.model_dir)
    tf.gfile.MakeDirs(args.model_dir)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params={'batch_size': args.batch_size})
    classifier.train(
        input_fn=lambda: input_fn(args.data_dir, args.batch_size),
        steps=10000)


if __name__ == '__main__':
    main()