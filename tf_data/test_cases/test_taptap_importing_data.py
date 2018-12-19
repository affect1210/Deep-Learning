#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: test_taptap_importing_data.py
@time: 2018/12/16 18:43
"""
from unittest import TestCase

import pytest

from tf_data.high_level_apis.logging_util import test_logger
from tf_data.high_level_apis.taptap_importing_data import *

class TestTaptap_importing_data(TestCase):
    @pytest.mark.skip(reason="no way of currnetly testing this")
    def test_dataset_iterator_test(self):
        pass

    # @pytest.mark.skip(reason="ignore")
    def  test_dataset_list_files(self):
        dataset_list_files()
        test_logger.info("test_dataset_list_files is ok")

    def test_build_vocabulary(self):
        project_root = get_root_path()
        vocab_file_path = project_root + "/runs/taptap_review_vocab"
        build_vocabulary(dataset_list_files(),vocab_file_path)

    def test_load_vocabulary(self):
        project_root = get_root_path()
        vocab_file_path = project_root + "/runs/taptap_review_vocab"
        load_vocabulary(vocab_file_path)
