#!/usr/bin/env python
import unittest
from word_to_vector.test import *
from word_to_vector.word_embeddings import *
from word_to_vector.tl_word_embeddings_model import *
from word_to_vector import get_root_path


class TestWordEmbeddings(TestCase):

    def test_gensim_processor_fit_save(self):
        processor = GensimProcessor()
        processor.fit(get_root_path() + "/data/word.csv")
        processor.save(get_root_path() + "/data/word2vec.model")
        print("vocabulary size is {}".format(len(processor.vocabulary)))

    def test_gensim_processor_load(self):
        processor = GensimProcessor()
        processor.load(path=get_root_path() + "/data/word2vec.model")
        vocab_processor = processor.vocab_processor(5)
        print("load finish")
        print("vocabulary size is {}".format(len(processor.vocabulary)))
        print("word embeddings shape is ({0[0]},{0[1]})".format(processor.word_embeddings.shape))
        print("VocabularyProcessor generate success")

    def test_tensor_processor_train_save(self):
        processor = TensorProcessor()
        processor.fit(get_root_path() + "/data/part_fucun.csv")
        processor.save(get_root_path() + "/data/tf_word2vec.model")
        print("vocabulary size is {}".format(len(processor.vocabulary)))

    def test_tensor_processor_load(self):
        processor = TensorProcessor()
        processor.load(path=get_root_path() + "/data/tf_word2vec.model")
        vocab_processor = processor.vocab_processor(5)
        print("load finish")
        print("vocabulary size is {}".format(len(processor.vocabulary)))
        print("word embeddings shape is ({0[0]},{0[1]})".format(processor.word_embeddings.shape))
        print("VocabularyProcessor generate success")
