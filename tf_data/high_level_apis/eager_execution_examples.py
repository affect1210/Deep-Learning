#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: eager_execution_examples.py
@time: 2018/12/16 20:01
"""
import tensorflow as tf
import numpy as np
from tf_data.high_level_apis.logging_util import info_logger

# Charging Disable or Enable behave: "MatMul:0"  or  [[16. 21.],[28. 37.]]
# tf.enable_eager_execution()

def matmul_example1():
    # Enabling eager execution changes how TensorFlow operations behave
    #  — now they immediately evaluate and return their values to Python
    m = tf.constant([[2.0, 3.0], [4.0, 5.0]])
    info_logger.info("Enabling Eager Execution test matmul {}".format(tf.matmul(m, m)))
