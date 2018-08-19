#!/usr/bin/env python
# -*- coding:utf-8 -*
from cnn_sentence_classification.cnn_params_flags import FLAGS
from cnn_sentence_classification import data_parser
from tensorflow.contrib import learn
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

    return x_train, y_train, vocab_processor, x_dev, y_dev
