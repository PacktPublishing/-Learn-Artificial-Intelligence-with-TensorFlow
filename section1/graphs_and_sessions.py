"""
File: section1/graphs_and_sessions.py
Author: Brandon McKinzie
Description: snippets from the slides in video 1.4.
"""

import tensorflow as tf

# -----------------------------------
# Graphs.
# -----------------------------------

c = tf.constant(4.0, name='c')
assert c.graph is tf.get_default_graph()
initial_default_graph = tf.get_default_graph()

g = tf.Graph()
with g.as_default():
    c = tf.constant(4.0, name='c')
    assert c.graph is g
    assert c.graph is not initial_default_graph

# -----------------------------------
# Sessions.
# -----------------------------------

# 1. Define some computations.
x = tf.get_variable('x', shape=[2, 3])
y = tf.get_variable('y', shape=[3, 1])
z = tf.matmul(x, y, name='z')

with tf.Session() as sess:
    # 2. Initialize our variables.
    sess.run(tf.global_variables_initializer())
    # 3. Execute computations.
    output = sess.run(z)
    
