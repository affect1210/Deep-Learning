#!/usr/bin/env python
# encoding: utf-8
from cnn_sentence_classification.test_cases import *
from cnn_sentence_classification.word2vec_data_util import build_word2id,load_word2id

class TestBuild_word2id(TestCase):

    def test_build_word2id(self):
        result = build_word2id(output_file=FLAGS.word2id_output_file,corpus_root_dir=FLAGS.corpus_root_dir)
        test_logger.info("build word2id result {}".format(result))

    def test_load_word2id(  self):
        word2id = load_word2id(word2id_path=FLAGS.word2id_output_file)
        test_logger.info("load word2id result {}".format(word2id))
