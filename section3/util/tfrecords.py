import tensorflow as tf
import os
import re
import nltk
import numpy as np
from collections import OrderedDict

INT_TYPES = (int, np.int, np.int16, np.int32, np.int64)
FLOAT_TYPES = (float, np.float, np.float32, np.float64, np.float128)
_PAD_TOKEN = '_PAD'
_UNK_TOKEN = '_UNK'
START_VOCAB = OrderedDict([(_PAD_TOKEN, 0), (_UNK_TOKEN, 1)])


def to_feature(values):
    """Wrap values inside tf.train.Feature of appropriate type.
    Args:
        values: list(int/float/str).
    Returns:
        tf.train.Feature
    """
    if isinstance(values[0], INT_TYPES):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    elif isinstance(values[0], FLOAT_TYPES):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))
    elif isinstance(values[0], str):
        values = [bytes(item, 'utf-8') for item in values]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
    elif isinstance(values[0], bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
    else:
        raise ValueError(
            "Values not a recognized type; v: %s type: %s" %
            (str(values[0]), str(type(values[0]))))


def to_feature_list(values):
    """
    Args:
        values: list(list(int/float/str))
    """
    return tf.train.FeatureList(feature=[
        to_feature(v) for v in values])


def to_features(dictionary):
    """Helper: build tf.train.Features from str => list(int/float/str) dict."""
    features = {}
    for k, v in dictionary.items():
        if not v:
            raise ValueError("Empty generated field: %s", str((k, v)))
        features[k] = to_feature(v)
    return tf.train.Features(feature=features)


def to_feature_lists(dictionary):
    """
    Args:
        dictionary: str => list(list(int/float/str))
    """
    feature_list = {}
    for k, values in dictionary.items():
        feature_list[k] = to_feature_list(values)
    return tf.train.FeatureLists(feature_list=feature_list)


def save_tfrecords(out_path, labels, texts,  vocab):
    vocab_dict = {w: i for i, w in enumerate(vocab)}
    vectorized_texts = vectorize(texts, vocab_dict)
    print('Writing...')
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for i, (label, token_ids) in enumerate(zip(labels, vectorized_texts)):
            if not token_ids:
                print('skipping token_ids:', token_ids)
                continue
            if i % 100 == 0:
                print('\rSerializing example {}...'.format(i), end='', flush=True)

            example = tf.train.Example(features=to_features(
                {'label': [label],
                 'sequence': token_ids}))
            if example is not None:
                writer.write(example.SerializeToString())


def vectorize(texts, vocab_dict):
    vectorized_texts = []
    print('Vectorizing...')
    for i, tokens in enumerate(texts):
        if i % 100 == 0:
            print('\rVectorizing example {}...'.format(i), end='', flush=True)

        token_ids = [vocab_dict.get(t, START_VOCAB[_UNK_TOKEN]) for t in tokens]
        vectorized_texts.append(token_ids)
    return vectorized_texts



