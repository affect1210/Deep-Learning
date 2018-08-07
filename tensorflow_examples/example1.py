#!/usr/bin/env python
# -*- coding:utf-8 -*

import tensorflow as tf
import numpy as np
import os,sys
from tensorflow.contrib.tensorboard.plugins import projector

# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
tf.flags.DEFINE_string("log_dir",os.path.join(current_path, 'log'),"logdir")
FLAGS = tf.flags.FLAGS

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### create tensorflow structure end ###

tf.summary.scalar('loss', loss)

with tf.Session() as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
    session.run(init)# Very important

    for setup in range(201):
        session.run(train)
        if setup % 20 == 0:
            print(setup,session.run(Weights),session.run(biases))


