#!/usr/bin/env python
# -*- coding:utf-8 -*

import tensorflow as tf
import os, csv
import numpy as np
from cnn_sentence_classification.cnn_params_flags import FLAGS
from cnn_sentence_classification import data_parser
from cnn_sentence_classification.LoggerUtil import info_logger
from tensorflow.contrib import learn


def evaluation():
    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train:
        x_raw, y_test = data_parser.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw = ["a masterpiece four years in the making", "I feel good."]
        y_test = [1, 0]

    # 恢复vocabulary字典
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    info_logger.info("Evaluating...")

    # 评估
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session = tf.Session(config=session_conf)
        with session.as_default():
            # 加载 .meta 图与变量
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(session, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches  for one epoch
            batches = data_parser.all_batches_generator(list(x_test), batch_sentence_size=FLAGS.batch_sentence_size,
                                                        num_epochs=1, shuffle=False)
            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = session.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # logging accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        info_logger.info("Total number of test example: {}".format(len(y_test)))
        info_logger.info("Accuracy {:g}".format(correct_predictions / float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.csv")
    info_logger.info("Saving evaluation to {}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)

if __name__ == "__main__":
    info_logger.info("Evaluation Application ...")
    evaluation()