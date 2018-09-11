#!/usr/bin/env python
# -*- coding:utf-8 -*
from cnn_sentence_classification.cnn_params_flags import FLAGS
from cnn_sentence_classification import data_parser
from cnn_sentence_classification.text_cnn import TextCNN
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np


# 数据预处理
def preprocess():
    print("Loading data ...")
    x_text, y_lables = data_parser.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_sentence_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    x_text = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_lables)))
    x_shuffle = x_text[shuffle_indices]
    y_shuffle = y_lables[shuffle_indices]

    # 分割训练集与测试集
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_lables)))
    x_train, x_dev = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
    y_train, y_dev = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]

    del x_text, y_lables, x_shuffle, y_shuffle

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # 训练集  建立中文词汇表和把文本转为词ID序列   测试集
    return x_train, y_train, vocab_processor, x_dev, y_dev


'''
    模型训练核心方法
'''


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():
        # log_device_placement=True
        # allow_soft_placement=True
        session_conf = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                                      allow_soft_placement=FLAGS.allow_soft_placement)
        session = tf.Session(config=session_conf)
        with session.as_default():
            cnn = TextCNN(max_sentence_length=x_train.shape[1], num_classes=y_train.shape[1],
                          vocabulary_size=len(vocab_processor.vocabulary_), embedding_size=FLAGS.embedding_dims,
                          region_size=list(map(int, FLAGS.filter_size.split(","))),
                          num_filters=FLAGS.num_filter_per_region, l2_reg_lambda=FLAGS.l2_reg_lambda)
        # Define Training procedure
        # 变量在计算过程中是可变的，并且在训练过程中会自动更新或优化。如果只想在 tf 外手动更新变量，那需要声明变量是不可训练的，比如 not_trainable = tf.Variable(0, trainable=False)
        global_step = tf.Variable(0, name="global_step", trainable=False)
