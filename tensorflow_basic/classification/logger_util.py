#!/usr/bin/env python
# -*- coding:utf-8 -*
import os,sys
import tensorflow as tf
'''
    common logger module
'''
basepath = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
logdirpath = os.path.join(basepath,'log')
tf.flags.DEFINE_string("log_dir",logdirpath,"log_dir")
FLAGS = tf.flags.FLAGS

# create the directory for Tensorboard variable if there is not
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)