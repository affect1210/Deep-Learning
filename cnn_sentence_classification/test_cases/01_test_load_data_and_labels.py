#!/usr/bin/env python
# -*- coding:utf-8 -*
from unittest import TestCase
from cnn_sentence_classification.data_parser import load_data_and_labels
from cnn_sentence_classification.cnn_params_flags import FLAGS

'''
    数据加载单元测试
'''
class TestLoad_data_and_labels(TestCase):

    def test_load_data_and_labels(self):
        x_y_examples = load_data_and_labels(FLAGS.positive_data_file,FLAGS.negative_data_file)
        print(x_y_examples[0][0])
        print(x_y_examples[1][0])