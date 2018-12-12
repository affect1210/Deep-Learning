#!/usr/bin/env python
# -*- coding:utf-8 -*
from cnn_sentence_classification.test_cases import *
import unittest
import tensorflow as tf
import numpy as np

"""
    softmax:指数是一种骤增的函数，这将扩大向量中每个元素的差异。
            它也会迅速地产生一个巨大的值。之后，当进行向量的标准化时，最大的元素、也就是主导（dominate）了范数（norm）的那个元素，
            将会被标准化为一个接近 1 的数字；其他的元素将除以较大的值并，于是被标准化为一个接近 0 的数字。
            最终得到的向量清楚地显示出了哪个是其最大的值，即 “max”，
            但是却又保留了其值的原始的相对排列顺序，因此即为 “soft”
    cross_entropy:
"""
class TestSoftMax_CrossEntropy(TestCase):
    input_x = np.array([[0., 2., 1.], [0., 0., 1.]])
    label = np.array([[0., 0., 1.], [0., 0., 1.]])

    def softmax(self, logits):
        sf = np.exp(logits)
        sf = sf / np.sum(sf, axis=1).reshape(-1, 1)
        return sf

    def cross_entropy(self, softmax, labels):
        return -np.sum(labels * np.log(softmax), axis=1)

    def loss(self, cross_entropy):
        return np.mean(cross_entropy)

    @unittest.skip("skip softmax")
    def test_softmax(self):
        print(self.softmax(self.input_x))

    def test_cross_entropy(self):
        softmax = self.softmax(self.input_x)
        log_softmax = np.log(softmax)
        cross_entropy = self.cross_entropy(softmax, self.label)
        loss = self.loss(cross_entropy)
        print(
            "softmax {} , log_softmax {}, cross_entropy {} , loss {}".format(softmax, log_softmax, cross_entropy, loss))

    def test_tensorflow_softmax_cross_entropy_with_logits(self):
        graph = tf.Graph()
        with graph.as_default():
            tf_input_x = tf.constant(self.input_x)
            tf_label = tf.constant(self.label)
            tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf_input_x, labels=tf_label))
            tf_loss = tf.reduce_mean(tf.losses.log_loss())
        with tf.Session(graph=graph) as session:
            x, l, loss = session.run([tf_input_x, tf_label, tf_loss])
            print("x {} , l {} , loss {}".format(x, l, loss))
