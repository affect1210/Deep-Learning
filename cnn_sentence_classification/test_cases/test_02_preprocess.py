#!/usr/bin/env python
# -*- coding:utf-8 -*
from cnn_sentence_classification.test_cases import *
from cnn_sentence_classification.train import preprocess
'''
    数据预处理单元测试
'''
class TestPreprocess(TestCase):

    def test_preprocess(self):
        x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
        test_logger.info("x_train, y_train, vocab_processor, x_dev, y_dev: {0},{1}".format(len(x_train),len(x_dev)))
