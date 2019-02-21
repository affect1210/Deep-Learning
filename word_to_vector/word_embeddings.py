from tensorflow.contrib.learn.python.learn.preprocessing.categorical_vocabulary import CategoricalVocabulary
from cnn_sentence_classification.cnn_params_flags import FLAGS
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tensorflow.contrib import learn

import tensorflow as tf
import tensorlayer as tl
import collections
import random
import pickle
from tensorflow.python.platform import gfile
from contextlib import ExitStack
import numpy as np

"""
Gensim 方法
"""


class GensimVocabulary(CategoricalVocabulary):

    def __init__(self, model, unknown_token="<UNK>"):
        CategoricalVocabulary.__init__(self, unknown_token, True)

        freq = {"<UNK>": 0}
        mapping = {"<UNK>": 0}
        reverse_mapping = ["<UNK>"]

        for key in model.wv.vocab:
            freq[key] = model.wv.vocab[key].count
            mapping[key] = model.wv.vocab[key].index + 1
            reverse_mapping.append(key)

        self._freq = freq
        self._mapping = mapping
        self._reverse_mapping = reverse_mapping
        self._freeze = True


class GensimProcessor(object):

    def __init__(self):
        self.model = None
        self.vocabulary = None

    def fit(self, path):
        self.model = Word2Vec(LineSentence(path),  #
                              sg=1,  #
                              size=200,  #
                              window=8,  #
                              min_count=5,  #
                              negative=3,  #
                              sample=0.0001,  #
                              hs=0,  #
                              workers=5,  #
                              iter=5,  #
                              compute_loss=True  #
                              )

        self.vocabulary = GensimVocabulary(self.model)

    def save(self, path, binary=False):
        self.model.wv.save_word2vec_format(path, binary=binary)

    def load(self, path=FLAGS.gensim_model_file):
        self.model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(path)
        self.vocabulary = GensimVocabulary(self.model)

    @property
    def word_embeddings(self):
        return self.model.wv.vectors

    def vocab_processor(self, max_document_length):
        return learn.preprocessing.VocabularyProcessor(max_document_length, vocabulary=self.vocabulary)


"""
tensorflow 方式
"""


# ######################################### #
# TODO 使用tensorflow进行词向量训练，未完成 #
# ######################################### #
class Vocabulary(CategoricalVocabulary):

    def __init__(self, model, unknown_token="<UNK>"):
        CategoricalVocabulary.__init__(self, unknown_token, True)

        freq = {"<UNK>": 0}
        mapping = {"<UNK>": 0}
        reverse_mapping = ["<UNK>"]

        self._freq = freq
        self._mapping = mapping
        self._reverse_mapping = reverse_mapping
        self._freeze = True


class TensorProcessor(object):
    def __init__(self):
        self.vocabulary = None

    def fit(self, path):
        pass

    def save(self, path, binary=False):
        pass

    def load(self, path=FLAGS.gensim_model_file):
        pass

    @property
    def word_embeddings(self):
        return None

    def vocab_processor(self, max_document_length):
        return learn.preprocessing.VocabularyProcessor(max_document_length, vocabulary=self.vocabulary)
