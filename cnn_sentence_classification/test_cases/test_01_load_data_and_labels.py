#!/usr/bin/env python
# -*- coding:utf-8 -*
from cnn_sentence_classification.test_cases import *
from cnn_sentence_classification.data_parser import load_data_and_labels

'''
    数据加载单元测试
'''
class TestLoad_data_and_labels(TestCase):

    def test_load_data_and_labels(self):
        x_text, y_lables = load_data_and_labels(FLAGS.positive_data_file,FLAGS.negative_data_file)
        test_logger.info(x_text[:])
