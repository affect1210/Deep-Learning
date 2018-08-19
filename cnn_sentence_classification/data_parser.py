#!/usr/bin/env python
# -*- coding:utf-8 -*
import re
import numpy as np
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
        x_text = [clean_str(sentence) for sentence in x_text]
        # 生成标签
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_lables = [[1, 0] for _ in negative_examples]

        y_lables = np.concatenate((positive_labels, negative_lables), axis=0)

    return [x_text, y_lables]
