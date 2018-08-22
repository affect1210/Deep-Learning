#!/usr/bin/env python
# -*- coding:utf-8 -*
from unittest import TestCase
from cnn_sentence_classification.data_parser import *
from cnn_sentence_classification.cnn_params_flags import *
from cnn_sentence_classification.train import *
import numpy as np

# 测试生成整个迭代的没批数据
class TestAll_batches_generator(TestCase):
    def test_all_batches_generator(self):
        # sample test
        # x_train = [np.random.permutation([ii for ii in range(10)]) for i in range(200)]
        # y_train = [[0,1] for j in range(200)]
        # datasets = list(zip(x_train,y_train))
        x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
        gen_all_batches = all_batches_generator(list(zip(x_train,y_train)),128,5)
        for batches in gen_all_batches:
            x_train_batch,y_train_batch = zip(*batches)
            print("---------------------")
            print("batches {} {}".format(len(x_train_batch),len(y_train_batch)))
            print("---------------------")
