#!/usr/bin/env python
# -*- coding:utf-8 -*
import tensorflow as tf

class TextCNN(object):
    '''
        文本分类卷积类
        分为 嵌入层,卷积层,池化层,分类输出层
    '''
    def __init__(self,max_sentence_length,num_classes,vocabulary_size,embedding_size,region_size):
        #将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符
        self.input_x = tf.placeholder()

