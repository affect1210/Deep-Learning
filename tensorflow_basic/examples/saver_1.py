#!/usr/bin/env python
# -*- coding:utf-8 -*

import tensorflow as tf
import os, sys
from tensorflow.contrib.tensorboard.plugins import projector

# sys.argv[0] 表示代码本身文件路径
current_path = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
tf.flags.DEFINE_string("log_dir", os.path.join(current_path, 'log'), "logdir")
tf.flags.DEFINE_string("save_net", os.path.join(current_path, 'log/save_net.ckpt'), "save_net.ckpt")
FLAGS = tf.flags.FLAGS

## Save to file
W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='weights')
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)
    save_path = saver.save(session, FLAGS.save_net)
    print("Save to path:",save_path)
