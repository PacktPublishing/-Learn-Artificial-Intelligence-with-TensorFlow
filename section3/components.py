from tensorflow.contrib.tensorboard.plugins import projector

import tensorflow as tf
from util import glove
import numpy as np
from tensorflow.contrib.rnn import DropoutWrapper, LSTMCell, MultiRNNCell, LSTMStateTuple


def init_embeddings_projector(vocab_path, tensor_name, logdir):
    """Saves the projector_config.pbtxt file in `logdir`, to be used
    by the TensorBoard Projector.
    """
    # Define the protobuf object that will eventually be saved in text format
    # as projector_config.pbtxt, within `logdir`. TensorBoard will automatically
    # read that file and associate the elements of `tensor_name` with the
    # vocabulary words specified in the `vocab_path` file.
    config = projector.ProjectorConfig(embeddings=[projector.EmbeddingInfo(
        tensor_name=tensor_name,
        metadata_path=vocab_path)])
    # Call visualize_embeddings to execute the saving of the .pbtxt file.
    writer = tf.summary.FileWriter(logdir)
    projector.visualize_embeddings(writer, config)


def get_embedding_matrix(embed_shape, vocab_path):
    """Loads tokens from `vocab_path` into a numpy matrix with
    shape `embed_shape`.
    """
    # Get list of tokens in the vocabulary.
    with open(vocab_path) as f:
        vocabulary = list(map(str.strip, f.readlines()))

    # Get dictionary for mapping word => vector.
    word_vec = glove.get_glove(dim=embed_shape[-1])

    # Fill emb_matrix[i] with word vector for ith word in vocabulary.
    emb_matrix = np.zeros(embed_shape)
    for i, word in enumerate(vocabulary):
        embed_vec = word_vec.get(word)
        if embed_vec is not None:
            emb_matrix[i] = embed_vec

    return emb_matrix


def glove_embed(features, embed_shape, vocab_path, projector_path=None):
    """Loads and builds an embedding layer initialized with pre-trained
    GloVe embeddings, using only the words given in `vocab_path`.

    Args:
        features: int64 Tensor with shape (batch_size, max_seq_len) containing
            the integer ids we want to embed into vectors.
        embed_size: (int) dimensionality of the embedding layer to build.
        vocab_path: (str) full path to text file where each line contains
            a single word, and the number of lines defines the size of the vocabulary.
        projector_path (optional): path to store embedding information needed to
            link TensorBoard projector with word labels.

    Returns:
        embedded_features: float32 Tensor with shape
            (batch_size, max_seq_len, embed_size) containing the embedded `features`.
    """
    with tf.variable_scope('glove_embed', values=[features]):
        embedding_matrix = get_embedding_matrix(embed_shape, vocab_path)
        embed_tensor = tf.get_variable(
            name='embed_tensor',
            initializer=tf.constant_initializer(embedding_matrix),
            dtype=tf.float32,
            trainable=False,
            shape=embed_shape)
        tf.summary.histogram('embed_tensor', embed_tensor)
        embedded_features = tf.nn.embedding_lookup(embed_tensor, features)

        # Sync vocabulary labels with TensorBoard Projector, if
        # projector_path has been provided.
        if projector_path is not None:
            tf.logging.info('Setting up TensorBoard Projector.')
            init_embeddings_projector(
                vocab_path=vocab_path,
                tensor_name=embed_tensor.name,
                logdir=projector_path)

        return embedded_features


def trainable_embed(features, embed_shape):
    """Creates trainable embedding layer, as opposed to loading pretrained GloVe."""
    embed_tensor = tf.get_variable(
        name='embed_tensor', dtype=tf.float32,
        shape=embed_shape)
    tf.summary.histogram('embed_tensor', embed_tensor)
    embedded_features = tf.nn.embedding_lookup(embed_tensor, features)
    return embedded_features


def deep_blstm(inputs, mode, state_size, num_layers, dropout_prob):
    with tf.variable_scope('deep_blstm', values=[inputs]):
        def deep_lstm():
            if mode == tf.estimator.ModeKeys.TRAIN:
                return MultiRNNCell([
                    DropoutWrapper(LSTMCell(state_size), state_keep_prob=1.-dropout_prob)
                    for _ in range(num_layers)])
            else:
                return MultiRNNCell([
                    LSTMCell(state_size) for _ in range(num_layers)])

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
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
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

        return outputs, combined_final_state


def stacked_blstm(inputs, num_layers=1):
    """Builds a stack_bidirectional_dynamic_rnn layer, which has a slightly
    different architecture than the deep_lstm function."""
    def get_cell():
        return tf.nn.rnn_cell.LSTMCell(128)

    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        [get_cell() for _ in range(num_layers)],
        [get_cell() for _ in range(num_layers)],
        inputs,
        dtype=tf.float32)
    return outputs

