#!/usr/bin/env python
# -*- coding:utf-8 -*
from unittest import TestCase
import os
from cnn_sentence_classification.app_root import get_root_path

project_root_path = get_root_path()
log_path = os.path.join(project_root_path, "config")
import logging
import logging.config

info_logger = logging.getLogger('infoLogger')
error_logger = logging.getLogger('errorLogger')

class TestBase(TestCase):

    def __init_subclass__(cls, **kwargs):
        log_file_name = "logging.conf"
        filepath = os.path.join(log_path, log_file_name)
        if os.path.isfile(filepath) is False:
            raise Exception("Config file {} not found".format(filepath))
        else:
            logging.config.fileConfig(filepath)

    def logging_info(self, content):
        info_logger.info(content)

    def logging_error(self, content):
        error_logger.error(content)

    def test(self):
        self.logging_info("info")
        self.logging_error("error")
