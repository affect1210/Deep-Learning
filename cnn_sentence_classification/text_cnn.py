#!/usr/bin/env python
# -*- coding:utf-8 -*
import tensorflow as tf
from cnn_sentence_classification.LoggerUtil import *


class TextCNN(object):
    '''
        文本分类卷积类
        分为 嵌入层,卷积层,池化层,分类输出层
    '''

    def __init__(self, max_sentence_length, num_classes,
                 vocabulary_size, embedding_size, region_size, num_filters, l2_reg_lambda=0.0):
        # 将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符
        self.input_x = tf.placeholder(tf.int32, [None, max_sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.EW = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],
                                                    -1.0, 1.0, dtype=tf.float32), name="EW")
            self.embedded_chars = tf.nn.embedding_lookup(self.EW, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # 为每个不同region大小的卷积核创建 convolution + maxpool 层
        pooled_outputs = []
        for i, region in enumerate(region_size):
            with tf.name_scope("conv-maxpool-{}".format(region)):
                # Convolution Layer
                # 卷积核矩阵的形状
                filter_shape = [region, embedding_size, 1, num_filters]
                CW = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="CW")
                cb = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="cb")
                # convolution funcation
                conv_value = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    CW, strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_func")
                # activate funcation for nonlinear transformation output frature map
                feature_map = tf.nn.relu(tf.nn.bias_add(conv_value, cb), name="relu")
                # Maxpooling over the outputs
                # ksize: 池化窗口的大小
                # strides: 滑动步长
                pooled_output = tf.nn.max_pool(
                    feature_map,
                    ksize=[1, max_sentence_length - region + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name="max_pool")
                # 日志
                info_logger.info(pooled_output)
                pooled_outputs.append(pooled_output)
        # self.tensor_pooled_outputs = tf.convert_to_tensor(pooled_outputs)
        # 合并所有池化后的特征
        # num_filters_total = num_filters * len(region)
        # self.feature_pooled =
