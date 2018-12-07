#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: word2vec_data_util.py
@time: 2018/12/5 18:07
"""
import numpy as np
import os

def build_word2id(file):
    '''
    :param file: word2id保存地址
    :return: None
    '''
    word2id = {'_pad_',0}
    # data_path =


