import tensorflow as tf
import tensorlayer as tl
import numpy as np
import collections
import random
import pickle
from tensorflow.python.platform import gfile
from cnn_sentence_classification.cnn_params_flags import FLAGS
from tensorflow.contrib import learn

from word_to_vector import get_root_path


class TFWordEmbeddings(object):
    def __init__(self):
        pass

    def generate_train_set(self, data):
        words = self.data_to_list(data)
        self.inputs, self.labels = self.build_word_set(data)
        del data
        self.count, self.dictionary, self.reverse_dictionary = self.build_words_dict(words)
        del words

    def data_to_list(self, data):
        words = []
        for item in data:
            words += item.split(" ")
        return words

    def build_words_dict(self, words):
        """
        建立词典，是用tensorlayer中的nlp.build_words_dataset
        :param words: 所有词的列表
        :return: 词对应id的字典
        """
        size = len(collections.Counter(words).keys())
        data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words,
                                                                                 vocabulary_size=size,
                                                                                 printable=True,
                                                                                 unk_key="<UNK>")
        return count, dictionary, reverse_dictionary

    def build_word_set(self, data):
        """
        :param data: 句子集合
        :return: x -> y
        """
        x = []
        y = []

        for item in data:
            sentence = []
            for i in item:
                if i in self.dictionary:
                    sentence.append(self.dictionary[i])
                else:
                    sentence.append(0)

            batch, labels = self.get_word_pair(sentence, num_skips=self.num_skips)

            if batch is not None and labels is not None:
                x += batch
                y += labels

        return x, y

    def get_word_pair(self, sentence, num_skips, data_index=0):
        """
        使用skip-gram的方式获取《词对》
        """
        batch = list()
        labels = list()
        sentence_length = len(sentence)
        skip_window = num_skips // 2
        span = 2 * skip_window + 1
        mid = span // 2
        lr = num_skips // 2
        buffer = collections.deque(maxlen=span)

        if sentence_length < span:
            return None, None

        for _ in range(span):
            buffer.append(sentence[data_index])
            data_index += 1

        start = skip_window
        end = sentence_length - skip_window

        for i in range(start, end):
            for j in range(lr):
                batch.append(buffer[mid])
                labels.append([buffer[mid + j + 1]])
                batch.append(buffer[mid])
                labels.append([buffer[mid - j - 1]])

            if data_index < sentence_length:
                buffer.append(sentence[data_index])
                data_index += 1

        return batch, labels

    def batch(self, batch_size):
        start = random.randint(0, len(self.inputs) - batch_size)
        end = start + batch_size
        return self.inputs[start:end], self.labels[start:end]

    def train(self, data, num_skips=2, batch_size=512, learning_rate=0.1, steps=10000, embedding_size=200, num_sampled=64):

        """
        :param data: 训练用句子语料
        :param num_skips:
        """
        self.num_skips = num_skips
        self.generate_train_set(data)

        train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
        train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
        net = tl.layers.Word2vecEmbeddingInputlayer(name='word2vec',
                                                    inputs=train_inputs,
                                                    train_labels=train_labels,
                                                    vocabulary_size=len(self.dictionary),
                                                    embedding_size=embedding_size,
                                                    num_sampled=num_sampled)

        cost = net.nce_cost
        train_params = net.all_params
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=train_params)

        num_steps = steps
        check_num = int(steps / 10)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = self.batch(batch_size)

                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = sess.run([train_op, cost], feed_dict=feed_dict)
                average_loss += loss_val

                if step % check_num == 0:
                    if step > 0:
                        average_loss /= check_num
                    # The average loss is an estimate of the loss over the last check_num batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

            normalized_embeddings = net.normalized_embeddings
            self.word_embeddings = normalized_embeddings.eval()

            return self.word_embeddings

    def save(self, filename):
        with gfile.Open(filename, 'wb') as f:
            f.write(pickle.dumps(self))

    @classmethod
    def load(cls, filename):
        with gfile.Open(filename, 'rb') as f:
            return pickle.loads(f.read())


if __name__ == "__main__":
    f = open(get_root_path() + "/data/word.csv", "r", encoding="UTF-8")
    data = f.readlines()

    w2v = TFWordEmbeddings()
    w2v.train(data)
