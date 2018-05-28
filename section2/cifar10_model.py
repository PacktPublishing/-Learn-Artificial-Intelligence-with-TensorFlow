"""
File: section3/cifar10_model.py
Author: Brandon McKinzie
Description: Builds the CIFAR-10 network.
"""

import tensorflow as tf


def l2_regularizer(wd):
    def _l2_regularizer(var, name=None):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=name)
        tf.add_to_collection('losses', weight_decay)
        return weight_decay
    return _l2_regularizer


def conv_block(inputs, filters, kernel_size, name):
    """Builds a ConvBlock as seen in the section 2.3 slides.

    Args:
        inputs:
        filters: {N}
        kernel_size: {KS}
        name:

    Assumes we are using the default data format: NHWC

    For the stuff below, let:
        PS = pool_size
        KS = kernel_size
        S = stride for pooling

    Output shapes can be computed with standard formulas, given stride=1:
        Output height H <= (H - KS + 1 - PS)/S + 1
        Output width: W <= (W - KS + 1 - PS)/S + 1
    """
    with tf.variable_scope(name, 'conv_block'):
        x = tf.layers.Conv2D(
            filters, kernel_size,
            padding='same',
            use_bias=False)(inputs)
        x = tf.layers.BatchNormalization(
            epsilon=1e-5,
            fused=True,
            name='batch_norm')(x, training=True)
        x = tf.nn.relu(x, name='relu')
        x = tf.layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            padding='same',
            name='max_pool')(x)
        return x


def inference(image_batch, batch_size=128):
    """Build the CIFAR-10 model.

    Args:
      image_batch: Images returned from distorted_inputs() or inputs().
      batch_size: (int) number of examples per batch.

    Returns:
      Logits.
    """
    kernel_size = 5
    num_kernels_per_conv = 64
    conv_block_1 = conv_block(
        image_batch,
        filters=num_kernels_per_conv,
        kernel_size=kernel_size,
        name='conv_layer_1')
    conv_block_2 = conv_block(
        conv_block_1,
        filters=num_kernels_per_conv,
        kernel_size=kernel_size,
        name='conv_layer_2')

    with tf.variable_scope('fc_relu_1'):
        # Move everything into depth so we can perform a single matrix multiply.
        # reshape = tf.reshape(conv2, [batch_size, -1])
        flattened_inputs = tf.layers.Flatten()(conv_block_2)
        fc_relu_1 = tf.layers.Dense(
            384,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.04),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=l2_regularizer(wd=0.004))(flattened_inputs)

    with tf.variable_scope('fc_relu_2'):
        fc_relu_2 = tf.layers.Dense(
            192,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.04),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=l2_regularizer(wd=0.004))(fc_relu_1)

    with tf.variable_scope('fc_logits'):
        logits = tf.layers.Dense(
            10,  # CIFAR-10 has 10 classes.
            kernel_initializer=tf.truncated_normal_initializer(stddev=1/192.0),
            bias_initializer=tf.constant_initializer(0.0),
            kernel_regularizer=l2_regularizer(wd=0.0))(fc_relu_2)

    return logits


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, batch_size):
    global_step = tf.train.get_or_create_global_step()
    num_examples_per_epoch_for_train = 50000
    num_epochs_per_decay = 350.0
    num_batches_per_epoch = num_examples_per_epoch_for_train / batch_size
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    lr = tf.train.exponential_decay(
        0.1, global_step, decay_steps,
        decay_rate=0.1, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=global_step,
            learning_rate=lr,
            optimizer='SGD')
    return train_op
