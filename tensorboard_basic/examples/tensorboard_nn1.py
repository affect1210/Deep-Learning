#!/usr/bin/env python
# -*- coding:utf-8 -*

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorboard_basic.examples.logger_util as logger
# Tensorboard Summaries Logger
FLAGS = logger.FLAGS

def add_layer(inputs, in_size, out_size, activation_function=None):

    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
          Weigths = tf.Variable(tf.truncated_normal(mean=0.0, stddev=0.1, shape=[in_size, out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weigths) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 一个":"代表开始,"::"代表结束

# 加入一些噪点,使不完全按二次方的曲线连续分布
noise = np.random.normal(0, 0.01, x_data.shape)  # 正态分布
y_data = np.square(x_data) - 0.5 + noise

# 1. define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, shape=(None, 1), name='x_input')
    ys = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 输入层1个神经元  隐层10个  输出层1个
l1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)

prediction = add_layer(l1, 20, 1, activation_function=None)


# 方差是衡量随机变量或一组数据离散程度的度量, 在概率论中方差用来度量随机变量和其数学期望(均值)之间的偏离程度。
# reduction_indices=1 按行求和
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    writer = tf.summary.FileWriter(FLAGS.log_dir,session.graph)
    session.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()

    for i in range(10000):
        session.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # print("prediction %s"%session.run(prediction, feed_dict={xs: x_data, ys: y_data}))
            print(session.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = session.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
