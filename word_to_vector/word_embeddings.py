from tensorflow.contrib.learn.python.learn.preprocessing.categorical_vocabulary import CategoricalVocabulary
from cnn_sentence_classification.cnn_params_flags import FLAGS
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from word_to_vector.tl_word_embeddings_model import *

"""
Gensim 方法
"""


class GensimVocabulary(CategoricalVocabulary):

    def __init__(self, model, unknown_token="<UNK>"):
        CategoricalVocabulary.__init__(self, unknown_token, True)

        freq = {"<UNK>": 0}
        mapping = {"<UNK>": 0}
        reverse_mapping = ["<UNK>"]

        for key in model.wv.vocab:
            freq[key] = model.wv.vocab[key].count
            mapping[key] = model.wv.vocab[key].index + 1
            reverse_mapping.append(key)

        self._freq = freq
        self._mapping = mapping
        self._reverse_mapping = reverse_mapping
        self._freeze = True


class GensimProcessor(object):

    def __init__(self):
        self.model = None
        self.vocabulary = None

    def fit(self, path):
        self.model = Word2Vec(LineSentence(path),  # 训练数据
                              sg=1,  # 1表示使用skip-gram模式，其他表示CBOW模式
                              size=200,  # 词向量维度大小
                              window=8,  # 窗口大小
                              min_count=5,  # 忽略出现次数小于值的词
                              negative=3,  #
                              sample=0.0001,  #
                              hs=0,  # 1表示层次softmax优化，0表示负采样优化
                              workers=5,  # 工作线程数
                              iter=5,  # 迭代次数
                              compute_loss=True  #
                              )

        self.vocabulary = GensimVocabulary(self.model)

    def save(self, path, binary=False):
        self.model.wv.save_word2vec_format(path, binary=binary)

    def load(self, path=FLAGS.gensim_model_file):
        self.model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(path)
        self.vocabulary = GensimVocabulary(self.model)

    @property
    def word_embeddings(self):
        return self.model.wv.vectors

    def vocab_processor(self, max_document_length):
        return learn.preprocessing.VocabularyProcessor(max_document_length, vocabulary=self.vocabulary)


"""
tensorflow 方式
"""


# ######################################### #
# 使用tensorflow进行词向量训练
# TODO 疑似存在问题，待探究
# ######################################### #
class Vocabulary(CategoricalVocabulary):

    def __init__(self, model, unknown_token="<UNK>"):
        CategoricalVocabulary.__init__(self, unknown_token, True)

        freq = {"<UNK>": 0}
        mapping = {"<UNK>": 0}
        reverse_mapping = ["<UNK>"]

        self._freq = freq
        self._mapping = mapping
        self._reverse_mapping = reverse_mapping
        self._freeze = True


class TFVocabulary(learn.preprocessing.CategoricalVocabulary):
    """
    use tensorlayar.nlp.build_words_dataset to create vocabulary
    """

    def __init__(self, model, unknown_token="<UNK>"):
        learn.preprocessing.CategoricalVocabulary.__init__(self, unknown_token, True)

        self._freq = model.count
        self._mapping = model.dictionary
        self._reverse_mapping = list(model.dictionary.keys())
        self._freeze = True


class TensorProcessor(object):
    def __init__(self):
        self.vocabulary = None
        self.model = TFWordEmbeddings()

    def fit(self, path):
        f = open(path, mode="r", encoding="UTF-8")
        data = f.readlines()
        self.model.train(data)
        self.vocabulary = TFVocabulary(self.model)

    def save(self, path, binary=False):
        self.model.save(path)

    def load(self, path=FLAGS.gensim_model_file):
        self.model = TFWordEmbeddings.load(path)
        self.vocabulary = TFVocabulary(self.model)

    @property
    def word_embeddings(self):
        return self.model.word_embeddings

    def vocab_processor(self, max_document_length):
        return learn.preprocessing.VocabularyProcessor(max_document_length, vocabulary=self.vocabulary)
