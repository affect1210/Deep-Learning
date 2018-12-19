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
from tf_data.high_level_apis.logging_util import info_logger
from tf_data.app_root import get_root_path
from tensorflow.contrib import learn

# Eager execution provides an imperative interface to TensorFlow
tf.enable_eager_execution()


# 1.0 read the file_dataset of this taptap data dir
def dataset_list_files():
    project_root = get_root_path()
    taptap_file_path = project_root + "\data\*"
    info_logger.info("dataset list_files taptap_file_path: {}".format(taptap_file_path))

    files = tf.data.Dataset.list_files(taptap_file_path)
    # 1.1. pure create list or tuple
    # info_logger.info(list(files))
    # info_logger.info(tuple(files))

    # 1.2. list comprehensions : List comprehensions provide a concise way to create lists
    # [info_logger.info(file_tenor.numpy()) for file_tenor in files]

    # 1.3. what is the difference between unicode and bytes
    [info_logger.info(str(file_tenor.numpy(),encoding='utf-8')) for file_tenor in files]

    # 1.4. the goal of __repr__ is ti be unambiguous
    # [info_logger.info(repr(file_tenor.numpy())) for file_tenor in files]

    #----------------------------------------------------------------------------

    # 2.1 A Dataset comprising lines from one or more text files.

    # map_func
    def read_lines(text_row):
        text_row_str = tf.string_strip(text_row)
        return text_row_str

    dataset = files.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
                .map(lambda line: read_lines(line))))

    # 2,2 read the value of this Tensor
    [info_logger.debug(str(line.numpy(),'utf-8')) for line in dataset]

    return dataset

# 2.0 build vocabulary of the taptap review
def build_vocabulary(dataset,vocab_file):

    max_sentence_length = max([len(str(line.numpy(),'utf-8').split(" ")) for line in dataset])
    info_logger.debug("max_sentence_length {}".format(max_sentence_length))
    vocabularyProcessor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    vocabularyProcessor.fit([str(line.numpy(),'utf-8') for line in dataset])
    vocabularyProcessor.save(vocab_file)

    return vocabularyProcessor

def load_vocabulary(vocab_file):
    pass




















