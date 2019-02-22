#!/usr/bin/env python
# -*- coding:utf-8 -*
from cnn_sentence_classification.cnn_params_flags import FLAGS
from cnn_sentence_classification import data_parser
from cnn_sentence_classification.text_cnn import TextCNN
from cnn_sentence_classification.app_root import get_root_path
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import time, os, datetime
from cnn_sentence_classification.LoggerUtil import *
from word_to_vector.word_embeddings import GensimProcessor
from word_to_vector.word_embeddings import TensorProcessor

project_root_path = get_root_path()


# 数据预处理
def preprocess():
    print("Loading data ...")
    x_text, y_labels = data_parser.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_sentence_length = max([len(x.split(" ")) for x in x_text])
    # tflearn.data_utils.VocabularyProcessor (max_document_length, min_frequency=0, vocabulary=None, tokenizer_fn=None)
    # max_document_length: 文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充.
    # min_frequency: 词频的最小值，出现次数小于最小词频则不会被收录到词表中
    # vocabulary: CategoricalVocabulary object.
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    # 文本转为词ID序列，未知或填充用的词ID为0
    x_text = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_labels)))
    x_shuffle = x_text[shuffle_indices]
    y_shuffle = y_labels[shuffle_indices]

    # 分割训练集与测试集
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_labels)))
    x_train, x_dev = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
    y_train, y_dev = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]

    del x_text, y_labels, x_shuffle, y_shuffle

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # 训练集  建立中文词汇表和把文本转为词ID序列   测试集
    return x_train, y_train, x_dev, y_dev, vocab_processor


# 使用gensim生成的词向量进行训练
def preprocess_with_gensim():
    print("==============================================start to preprocess data==============================================")

    print("load data and labels...")
    x_text, y_labels = data_parser.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    max_sentence_length = max([len(x.split(" ")) for x in x_text])

    print("load word embeddings...")

    # new GensimProcessor
    gensim_processor = GensimProcessor()
    # load gensim word embeddings
    gensim_processor.load()
    # word embeddings matrix
    word_embeddings = gensim_processor.word_embeddings
    # generate VocabularyProcessor
    vocab_processor = gensim_processor.vocab_processor(max_sentence_length)

    print("transform data to word embeddings...")
    # transform train set
    x_text = np.array(list(vocab_processor.transform(x_text)))

    print("generate test and valid data set...")
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_labels)))
    x_shuffle = x_text[shuffle_indices]
    y_shuffle = y_labels[shuffle_indices]

    # 分割训练集与测试集
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_labels)))
    x_train, x_dev = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
    y_train, y_dev = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]

    del x_text, y_labels, x_shuffle, y_shuffle

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    print("==============================================finish to preprocess data=============================================")

    # 训练集  建立中文词汇表和把文本转为词ID序列   测试集
    return x_train, y_train, x_dev, y_dev, vocab_processor, word_embeddings


# 使用gensim生成的词向量进行训练
def preprocess_with_tensorflow():
    print("==============================================start to preprocess data==============================================")

    print("load data and labels...")
    x_text, y_labels = data_parser.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    max_sentence_length = max([len(x.split(" ")) for x in x_text])

    print("load word embeddings...")

    # new GensimProcessor
    gensim_processor = TensorProcessor()
    # load gensim word embeddings
    gensim_processor.load()
    # word embeddings matrix
    word_embeddings = gensim_processor.word_embeddings
    # generate VocabularyProcessor
    vocab_processor = gensim_processor.vocab_processor(max_sentence_length)

    print("transform data to word embeddings...")
    # transform train set
    x_text = np.array(list(vocab_processor.transform(x_text)))

    print("generate test and valid data set...")
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_labels)))
    x_shuffle = x_text[shuffle_indices]
    y_shuffle = y_labels[shuffle_indices]

    # 分割训练集与测试集
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_labels)))
    x_train, x_dev = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
    y_train, y_dev = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]

    del x_text, y_labels, x_shuffle, y_shuffle

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    print("==============================================finish to preprocess data=============================================")

    # 训练集  建立中文词汇表和把文本转为词ID序列   测试集
    return x_train, y_train, x_dev, y_dev, vocab_processor, word_embeddings


'''
    模型训练核心方法
'''


def train(x_train, y_train, x_dev, y_dev, vocab_processor, word_embeddings=None):
    with tf.Graph().as_default():
        # log_device_placement=True
        # allow_soft_placement=True
        session_conf = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                                      allow_soft_placement=FLAGS.allow_soft_placement)

        session = tf.Session(config=session_conf)

        with session.as_default():
            cnn = TextCNN(max_sentence_length=x_train.shape[1],
                          num_classes=y_train.shape[1],
                          vocabulary_size=len(vocab_processor.vocabulary_),
                          embedding_size=FLAGS.embedding_dims,
                          word_embeddings=word_embeddings,
                          region_size=list(map(int, FLAGS.filter_size.split(","))),
                          num_filters=FLAGS.num_filter_per_region,
                          l2_reg_lambda=FLAGS.l2_reg_lambda)
            """
                 参数选择性训练方法:
                    1、定义Variable时设置trainable=False
                    2、通知optimizer只更新部分梯度：opt.compute_gradients(loss, var_list)
            """
            # Define Training procedure
            # 变量在计算过程中是可变的，并且在训练过程中会自动更新或优化。如果只想在 tf 外手动更新变量，
            # 那需要声明变量是不可训练的，比如 not_trainable = tf.Variable(0, trainable=False)
            # 计数器
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
            # Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳.
            optimizer = tf.train.AdamOptimizer()
            # compute_gradients 返回的是：A list of (gradient, variable) pairs
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # minimize() = compute_gradients() + apply_gradients()
            # 拆分成计算梯度和应用梯度两个步骤
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(project_root_path, "runs", timestamp))
            info_logger.info("Writing to {}\n".format(out_dir))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # save vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            session.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy, flat, embedded = session.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,
                     cnn.feature_pooled_flat, cnn.embedded_chars_expanded],
                    feed_dict)

                # print("flat shape:{}".format(flat.shape))
                # print("embedded shape:{}".format(embedded.shape))

                time_str = datetime.datetime.now().isoformat()
                # 在保证六位有效数字的前提下，使用小数方式，否则使用科学计数法
                info_logger.info("{}: step {},loss {:g} , accuracy {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            # batch 训练
            batches = data_parser.all_batches_generator(list(zip(x_train, y_train)),
                                                        num_sentence_per_batch=FLAGS.num_sentence_per_batch,
                                                        num_epochs=FLAGS.num_epochs)
            # Training loop. For each batch...
            for i, batch in enumerate(batches):
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(session, global_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(session, checkpoint_prefix, global_step=current_step)
                    info_logger.info("Save model checkpoint to {}".format(path))


def train_with_random_embeddings():
    """
    使用随机词向量进行模型训练
    """
    x_train, y_train, x_dev, y_dev, vocab_processor = preprocess()
    train(x_train, y_train, x_dev, y_dev, vocab_processor)


def train_with_word_embeddings():
    """
    使用gensim训练的词向量进行模型训练
    """
    x_train, y_train, x_dev, y_dev, vocab_processor, word_embeddings = preprocess_with_gensim()
    train(x_train, y_train, x_dev, y_dev, vocab_processor, word_embeddings)


def train_with_tensor_embeddings():
    """
    使用tensorlayer训练的词向量进行模型训练
    """
    x_train, y_train, x_dev, y_dev, vocab_processor, word_embeddings = preprocess_with_tensorflow()
    train(x_train, y_train, x_dev, y_dev, vocab_processor, word_embeddings)


if __name__ == "__main__":
    train_with_word_embeddings()
