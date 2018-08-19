#!/usr/bin/env python
# -*- coding:utf-8 -*
import tensorflow as tf
from cnn_sentence_classification.app_root import get_root_path

'''
    tf.app.flags.FLAGS 使用全局变量
    python train.py --positive_data_file "./data/en_polaritydata/rt-polarity.pos"
    --negative_data_file "./data/en_polaritydata/rt-polarity.neg"
    如果不传参数,采用默认设置
'''
project_root_path = get_root_path()
print(project_root_path)

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", project_root_path + "/data/en_polaritydata/rt-polarity.pos",
                       "positive data file path")
tf.flags.DEFINE_string("negative_data_file", project_root_path + "/data/en_polaritydata/rt-polarity.neg",
                       "negative data file path")


def test_params():
    positive_data_file = FLAGS.positive_data_file
    negative_data_file = FLAGS.negative_data_file
    print("positive_data_file %s" % positive_data_file)
    print("negative_data_file %s" % negative_data_file)


if __name__ == "__main__":
    test_params()
