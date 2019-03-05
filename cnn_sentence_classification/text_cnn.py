#!/usr/bin/env python
# -*- coding:utf-8 -*
import tensorflow as tf
from cnn_sentence_classification.LoggerUtil import *
import numpy as np

np.set_printoptions(threshold=np.inf)


class TextCNN(object):
    '''
        文本分类卷积类
        分为 嵌入层,卷积层,池化层,分类输出层
    '''

    def __init__(self, max_sentence_length,
                 num_classes,
                 vocabulary_size,
                 embedding_size,
                 region_size,
                 num_filters,
                 l2_reg_lambda=0.0,
                 word_embeddings=None):

        # if word_embeddings & vocabulary_size != word_embeddings.shape[0]:
        #     raise RuntimeError("'vocabulary_size' need the same length as 'embedding_words' shape 0 dimension size")
        #
        # if word_embeddings & embedding_size != word_embeddings.shape[1]:
        #     raise RuntimeError("'embedding_size' need the same length as 'embedding_words' shape 1 dimension size")

        # 将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符
        # shape[num_sentence_per_batch,56]
        self.input_x = tf.placeholder(tf.int32, [None, max_sentence_length], name="input_x")
        # shape[num_sentence_per_batch,2]
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        # Embedding layer , use '/cpu:0' to run in cpu
        with tf.device('/gpu:0'), tf.name_scope("embedding"):

            if word_embeddings is not None:
                # TODO 用词向量字典替换随机字典EW
                # 词向量由word_to_vector模块生成
                self.EW = tf.Variable(word_embeddings, dtype=tf.float32, name="EW")
            else:
                # 词典矩阵
                # shape[词典中词数,128]
                self.EW = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],
                                                        -1.0, 1.0, dtype=tf.float32), name="EW")

            # 用词典矩阵词向量替换 batch中所有的输入源矩阵
            # 在参数EW中查找word_id所对应的表示，embedded_chars_shape[?,56,128]
            self.embedded_chars = tf.nn.embedding_lookup(self.EW, self.input_x)

            # 为了给卷积用的加一维
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 为每个不同region大小的卷积核创建 convolution + maxpool 层
        pooled_outputs = []
        for i, region in enumerate(region_size):
            with tf.name_scope("conv-maxpool-{}".format(region)):
                # Convolution Layer
                # 卷积核矩阵的形状 [卷积核高度，卷积核宽度，图像通道数，卷积核个数]
                filter_shape = [region, embedding_size, 1, num_filters]
                CW = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="CW")
                cb = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="cb")

                # convolution funcation
                # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
                # input = [batch, in_height, in_width, in_channels]
                # Returns: A `Tensor`.Has the same type as `input`.
                conv_value = tf.nn.conv2d(
                    input=self.embedded_chars_expanded,
                    filter=CW,
                    strides=[1, 1, 1, 1],
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
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="max_pool")

                # 日志
                info_logger.info(tf.convert_to_tensor(pooled_output))
                pooled_outputs.append(pooled_output)

        # 合并所有池化后的特征
        num_filters_total = num_filters * len(region_size)
        self.feature_pooled = tf.concat(pooled_outputs, 3)
        self.feature_pooled_flat = tf.reshape(self.feature_pooled, [-1, num_filters_total])

        info_logger.info(self.feature_pooled_flat)

        # 添加DropOut
        with tf.name_scope("dropout"):
            self.feature_pooled_dropout = tf.nn.dropout(self.feature_pooled_flat, self.dropout_keep_prob)

        # 评分和预测
        with tf.name_scope("output"):
            # “Xavier”初始化方法是一种很有效的神经网络初始化方法，方法来源于2010年的一篇论文
            # 《Understanding the difficulty of training deep feedforward neural networks》，
            # 可惜直到近两年，这个方法才逐渐得到更多人的应用和认可。
            W_1 = tf.get_variable(
                "classifer_w_1",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            b_1 = tf.Variable(tf.constant(0.1, shape=[num_filters_total], name="classifer_b_1"))

            W_2 = tf.get_variable(
                "classifer_w_2",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            b_2 = tf.Variable(tf.constant(0.1, shape=[num_filters_total], name="classifer_b_2"))

            W_3 = tf.get_variable(
                "classifer_w_3",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b_3 = tf.Variable(tf.constant(0.1, shape=[num_classes], name="classifer_b_3"))

            # 这个函数的作用是利用L2范数来计算张量的误差值，但是没有开方并且只L2范数的值的一半
            l2_loss += tf.nn.l2_loss(W_1)
            l2_loss += tf.nn.l2_loss(b_1)
            l2_loss += tf.nn.l2_loss(W_2)
            l2_loss += tf.nn.l2_loss(b_2)
            l2_loss += tf.nn.l2_loss(W_3)
            l2_loss += tf.nn.l2_loss(b_3)

            layer1 = tf.matmul(self.feature_pooled_dropout, W_1) + b_1
            act1 = tf.nn.relu(layer1)
            layer2 = tf.matmul(act1, W_2) + b_2
            act2 = tf.nn.relu(layer2)

            # 相当于matmul(x, weights) + biases.
            self.scores = tf.nn.xw_plus_b(act2, W_3, b_3, name="scores")

            # 返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # 1. 第一步是先对网络最后一层的输出做一个softmax
            # 2. 第二步是softmax的输出向量[Y1，Y2,Y3…]和样本的实际标签做一个交叉熵
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # 如果不指定第二个参数，那么就在所有的元素中取平均值 l2_reg_lambda: 正则化率
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
