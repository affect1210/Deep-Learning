#!/usr/bin/env python
# -*- coding:utf-8 -*
from cnn_sentence_classification.test_cases.test_base import TestBase
from cnn_sentence_classification.data_parser import *
from cnn_sentence_classification.train import *
import numpy as np

# 测试生成整个迭代的每批数据
class TestAll_batches_generator(TestBase):

    def test_all_batches_generator(self):
        x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
        gen_all_batches = all_batches_generator(list(zip(x_train,y_train)),64,5)
        for batches in gen_all_batches:
            x_train_batch,y_train_batch = zip(*batches)
            print("---------------------")
            print("batches {} {}".format(len(x_train_batch),len(y_train_batch)))
            print("---------------------")
