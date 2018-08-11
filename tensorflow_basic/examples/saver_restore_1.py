#!/usr/bin/env python
# -*- coding:utf-8 -*

import tensorflow as tf
import numpy as np
import os, sys

current_path = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
save_net_path = os.path.join(current_path, "log/save_net.ckpt")
tf.flags.DEFINE_string("save_net", save_net_path, "save_net.ckpt")
FLAGS = tf.flags.FLAGS

# restore variables
# redefine the same shape and same type for your variables

W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# not need init setup
saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, FLAGS.save_net)
    print("weights:", session.run(W))
    print("biases:", session.run(b))
