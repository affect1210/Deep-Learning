#!/usr/bin/env python
import unittest
from cnn_sentence_classification.text_cnn import TextCNN
from cnn_sentence_classification.train import *
from cnn_sentence_classification.test_cases.test_base import TestBase
from cnn_sentence_classification.data_parser import *
import tensorflow as tf
import os,sys

# -*- coding:utf-8 -*
class TestTextCNN(TestBase):

    @unittest.skip('skip embedding_layer')
    def test_embedding_layer(self):
        x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
        gen_all_batches = all_batches_generator(list(zip(x_train, y_train)), 128, 5)
        with tf.Session() as session:
            cnn = TextCNN(56, 2, 20000, embedding_size=128, region_size=[3, 4, 5], num_filters=128, l2_reg_lambda=0.0)
            for batches in gen_all_batches:
                x_train_batch, y_train_batch = zip(*batches)
                print("---------------------")
                print("batches {} {}".format(len(x_train_batch), len(y_train_batch)))
                print("---------------------")
                session.run(tf.global_variables_initializer())
                print(session.run(cnn.embedded_chars, feed_dict={cnn.input_x: x_train_batch, cnn.input_y: y_train_batch,
                                                                 cnn.dropout_keep_prob: 0.5}))

    def test_train(self):
        x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
        # gen_all_batches = all_batches_generator(list(zip(x_train, y_train)), 64, 1)
        train(x_train, y_train, vocab_processor, x_dev, y_dev)


