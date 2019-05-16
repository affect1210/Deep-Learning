#!/usr/bin/env python
# -*- coding:utf-8 -*
import re
import numpy as np
import math
from contextlib import ExitStack

'''
    ExitStack是一个上下文管理器，允许你很容易地与其它上下文管理结合或者清除。
    ExitStack维护一个寄存器的栈。当我们退出with语句时，文件就会关闭，栈就会按照相反的顺序调用这些上下文管理器。
'''


def clean_str(string):
    # 切词
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    加载影评的正负数据文件,进行分词和生成标签,返回分词的句子与标签。
    :param positive_data_file:
    :param negative_data_file:
    :return:
    """
    # ExitStack()上下文管理器被设计为使得可以容易地以编程方式组合其他上下文管理器和清除功能，特别是那些是可选的或以其他方式由输入数据驱动的
    with ExitStack() as exitstack:
        # 加载数据文件
        # 1、readlines:返回文件所有行,区别于readline
        # 2、strip() 清除首尾的空格与换行符
        positive_examples = exitstack.enter_context(open(positive_data_file, "r", encoding='utf-8')).readlines()
        # 列表解析是python迭代机制的一种应用,用于创建新列表,语法: [expression for iter_val in iterable]
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = exitstack.enter_context(open(negative_data_file, "r", encoding='utf-8')).readlines()
        negative_examples = [s.strip() for s in negative_examples]
        # 分词
        x_text = positive_examples + negative_examples
        # x_text = [clean_str(sentence) for sentence in x_text]
        x_text = [sentence for sentence in x_text]
        # 生成标签
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_lables = [[1, 0] for _ in negative_examples]

        y_lables = np.concatenate((positive_labels, negative_lables), axis=0)

    return [x_text, y_lables]


def all_batches_generator(all_x_y_train, num_sentence_per_batch, num_epochs, shuffle=True):
    '''
        生成整个迭代需要的所有batch的数据集
    :param all_x_y_train:
    :param num_sentence_per_batch:
    :param num_epoch:
    :param shuffle:
    :return:
    '''
    datasets = np.array(all_x_y_train)
    datasets_size = len(datasets)
    num_batches_per_epoch = math.ceil(datasets_size / num_sentence_per_batch)  # 每个epoch中有多少个batch
    for epoch in range(num_epochs):
        # 保证每个epoch中的句子顺序也都是不同的
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(datasets_size))
            shuffle_datasets = datasets[shuffle_indices]
        else:
            shuffle_datasets = datasets

        # 组装每个batch的句子内容
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * num_sentence_per_batch
            end_index = min((batch_num + 1) * num_sentence_per_batch, datasets_size)
            yield shuffle_datasets[start_index:end_index]
