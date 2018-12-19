#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: test_eager_execution.py
@time: 2018/12/16 20:07
"""
from unittest import TestCase
from tf_data.high_level_apis.logging_util import test_logger
from tf_data.high_level_apis.eager_execution_examples import matmul_example1

class TestEagerExecution(TestCase):

    def test_matmul_example1(self):
        matmul_example1()

