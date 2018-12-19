#!/usr/bin/env python
# -*- coding:utf-8 -*

import os
from tf_data.app_root import get_root_path
import logging.config

project_root_path = get_root_path()
log_config_path = os.path.join(project_root_path, "config")

log_file_name = "logging.conf"
file_path = os.path.join(log_config_path, log_file_name)
if os.path.isfile(file_path) is False:
    raise Exception("Config file{} not Found!".format(file_path))
else:
    logging.config.fileConfig(file_path)

info_logger = logging.getLogger('infoLogger')
error_logger = logging.getLogger('errorLogger')
test_logger = logging.getLogger('testLogger')


