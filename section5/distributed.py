#!/usr/bin/env python3

"""
File: section5/distributed.py
Author: Brandon McKinzie
Description: snippets from the slides in video 5.1.

TensorFlow terminology:
    - cluster: either (1) a set of "tasks" or (2) a set of "jobs", where...
        - task: associated with a TensorFlow server, and participates in the
                distributed execution of a TensorFlow graph. Each task does...
                (1) Create a tf.train.ClusterSpec that describes all the tasks
                    in the cluster. This is the same for all tasks.
                (2) Create a tf.train.Server, passing the ClusterSpec to the
                    constructor, and identifying the local task with a job name
                    and task index.
        - job: a set of one or more tasks.
    - server: contains...
        - master: can be used to create sessions.
        - worker: executes operations in the graph.

"""
import tensorflow as tf
from pprint import pprint

job_name = 'my_job'


def long_device_name(short_name):
    return '/job:{}/replica:0/task:0/device:{}:0'.format(
        job_name, short_name.upper())


def device_fn(n):
    if 'Variable' in n.type:
        return long_device_name('CPU')
    else:
        return long_device_name('GPU')


with tf.device(device_fn):
    x = tf.get_variable(
        'x', shape=(), dtype=tf.int32,
        initializer=tf.constant_initializer(5))
    y = tf.get_variable(
        'y', shape=(), dtype=tf.int32,
        initializer=tf.constant_initializer(3))
    z = x + y

# Creates a single-process cluster, with an in-process server.
server = tf.train.Server({job_name: ['localhost:1234']})

with tf.Session(server.target) as sess:
    sess.run(tf.global_variables_initializer())
    print('Result: z = ', sess.run(z))

    print('Available devices:')
    pprint(sess.list_devices())

