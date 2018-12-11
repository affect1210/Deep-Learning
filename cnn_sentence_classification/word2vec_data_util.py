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
from cnn_sentence_classification.app_root import get_root_path


"""
    0.1     构建字典
        ------------------
        内容如:
            _pad_	0
            游戏	    1
            本该   	2
        ------------------
        注意: 其中的标点符号, 表情字符应该去除.
"""
def build_word2id(output_file, corpus_root_dir):
    '''
    :param output_file:    word2id输出文件存储路径
    :param corpus_root_dir:   预料库根目录
    :return:
    '''
    word2id = {'_pad_':0}
    data_path = [corpus_root_dir + "/" + w for w in os.listdir(corpus_root_dir)]
    for _path in data_path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[0:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    with open(output_file,'w',encoding='utf-8') as file:
        for w in word2id:
            file.write(w+'\t')
            file.write(str(word2id[w]))
            file.write('\n')
        return output_file
    return "build word2id file fail."
"""
    0.1.1 加载词典
"""
def load_word2id(word2id_path):
    word2id = {}
    with open(word2id_path,'r',encoding='utf-8') as file:
        for line in file.readlines():
            sp = line.strip().split()
            word = sp[0]
            wid = sp[1]
            if word not in word2id:
                word2id[word] = wid
    return word2id

