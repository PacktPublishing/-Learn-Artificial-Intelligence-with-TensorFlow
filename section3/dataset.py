"""
File: section3/dataset.py
Author: Brandon McKinzie
"""

import os
import tensorflow as tf


class Dataset(object):
    """Works on amazon_reviews and 20_newsgroups.

    Should rename.
    """

    def __init__(self, data_dir, params):
        self.data_dir = data_dir
        self.params = params

    def get_filenames(self, mode):
        # TODO: make compatible with:
        # - "embed.tfrecords"
        # - "20ng_simple_{}.tfrecords"
        return [os.path.join(self.data_dir, '{}_{}.tfrecords'.format(
            mode, self.params.vocab_size))]

    def parser(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={'label': tf.FixedLenFeature([], dtype=tf.string),
                      'sequence': tf.VarLenFeature(dtype=tf.int64)})
        features['sequence'] = tf.sparse_tensor_to_dense(features['sequence'])
        label = features['label']
        sequence = features['sequence']
        return sequence[self.params.max_seq_len], label

    def vectorize_labels(self, labels):
        table = tf.contrib.lookup.index_table_from_file(
            os.path.join(self.data_dir, 'labels.txt'))
        ids = table.lookup(labels)
        return ids

    def make_batch(self, mode):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames(mode)
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames).repeat()
        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls=self.params.batch_size)
        # Batch it up.
        dataset = dataset.shuffle(buffer_size=10 * self.params.batch_size)
        dataset = dataset.padded_batch(
            self.params.batch_size,
            padded_shapes=(tf.TensorShape(self.params.max_seq_len), ()))
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        labels = self.vectorize_labels(labels)
        return features, labels

