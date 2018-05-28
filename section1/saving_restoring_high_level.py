"""
File: section1/saving_restoring_high_level.py
Author: Brandon McKinzie
Description: snippets from the slides in video 1.5.
"""

import tensorflow as tf

x = tf.get_variable('x', shape=[2, 3])
y = tf.get_variable('y', shape=[3, 1])
z = tf.matmul(x, y, name='z')

# -----------------------------------
# SavedModelBuilder
# -----------------------------------

builder = tf.saved_model.builder.SavedModelBuilder('out/0')
with tf.Session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.TRAINING])
    builder.save()

# -----------------------------------
# Loader
# -----------------------------------

export_dir = 'my_export_dir'

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(
        sess=sess,
        tags=[tf.saved_model.tag_constants.TRAINING],
        export_dir='out')
# OR...
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'out/0/variables/variables')

# -----------------------------------
# OLD SKEWL
# -----------------------------------

# Saving:
saver = tf.train.Saver()
with tf.Session() as sess:
    save_path = saver.save(
        sess, '/tmp/model.ckpt')

# Restoring:
saver = tf.train.Saver()
with tf.Session() as sess:
    # Instead of initializing vars, do:
    saver.restore(
        sess, '/tmp/model.ckpt')

