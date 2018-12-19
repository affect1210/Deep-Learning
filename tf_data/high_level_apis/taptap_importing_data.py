#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: taptap_importing_data.py
@time: 2018/12/16 18:03
"""
import tensorflow as tf
import numpy as np
from tf_data.high_level_apis.logging_util import *
from tf_data.app_root import get_root_path
from tensorflow.contrib import learn

# Eager execution provides an imperative interface to TensorFlow
tf.enable_eager_execution()


# 1.0 read the file_dataset of this taptap data dir
def dataset_list_files():
    project_root = get_root_path()
    taptap_file_neg = project_root + "\data\*.neg"
    taptap_file_pos = project_root + "\data\*.pos"
    info_logger.info("dataset list_files taptap_file_neg: {}".format(taptap_file_neg))
    info_logger.info("dataset list_files taptap_file_pos: {}".format(taptap_file_pos))

    neg_files = tf.data.Dataset.list_files(taptap_file_neg)
    pos_files = tf.data.Dataset.list_files(taptap_file_pos)
    # 1.1. pure create list or tuple
    # info_logger.info(list(neg_files))
    # info_logger.info(tuple(pos_files))

    # 1.2. list comprehensions : List comprehensions provide a concise way to create lists
    # [info_logger.info(file_tenor.numpy()) for file_tenor in files]

    # 1.3. what is the difference between unicode and bytes
    [info_logger.info(str(neg_file_tenor.numpy(), encoding='utf-8')) for neg_file_tenor in neg_files]
    [info_logger.info(str(pos_file_tenor.numpy(), encoding='utf-8')) for pos_file_tenor in pos_files]

    # 1.4. the goal of __repr__ is ti be unambiguous
    # [info_logger.info(repr(neg_file_tenor.numpy())) for neg_file_tenor in neg_files]
    # [info_logger.info(repr(pos_file_tenor.numpy())) for pos_file_tenor in pos_files]

    #----------------------------------------------------------------------------

    # 2.1 A Dataset comprising lines from one or more text files.

    # map_func
    def read_lines(text_row, lables):
        return (text_row, lables)

    neg_dataset = neg_files.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
                .map(lambda line: read_lines(line, [1, 0]))))

    pos_dataset = pos_files.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
                .map(lambda line: read_lines(line, [0, 1]))))

    dataset = neg_dataset.concatenate(pos_dataset)

    # 2,2 read the value of this tuple(TextTensor,LableTensor)
    if info_logger.isEnabledFor(DEBUG):
        [info_logger.debug(line) for line in dataset]

    return dataset

# 2.0 build vocabulary of the taptap review
def build_vocabulary(dataset,vocab_file):

    max_sentence_length = max([len(str(line[0].numpy(),'utf-8').split(" ")) for line in dataset])
    info_logger.debug("max_sentence_length {}".format(max_sentence_length))
    vocabularyProcessor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    vocabularyProcessor.fit([str(line[0].numpy(),'utf-8') for line in dataset])
    vocabularyProcessor.save(vocab_file)

    return vocabularyProcessor

def load_vocabulary(vocab_filename):
    vocabularyProcessor = learn.preprocessing.VocabularyProcessor.restore(vocab_filename)

    if info_logger.isEnabledFor(DEBUG):
        info_logger.debug("load vocabulary is success. max_document_length:{}".format(vocabularyProcessor.max_document_length))

    return vocabularyProcessor




















