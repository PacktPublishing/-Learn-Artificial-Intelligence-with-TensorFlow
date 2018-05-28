"""
File: section1/tensors_and_variables.py
Author: Brandon McKinzie
Description: snippets from the slides in video 1.3.
"""

import tensorflow as tf

# Tensors: basic usage.
x = tf.constant(3, name='x')
y = tf.constant([4., 5., 6.])
z = tf.convert_to_tensor([[7., 8.], [9., 10.]])

# Variables: basic usage.
x = tf.Variable(initial_value=3, name='x')
y = tf.get_variable('y', initializer=[4., 5., 6.])
z = y + 3

