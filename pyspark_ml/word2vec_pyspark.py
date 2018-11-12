#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http:#www.changyou.com
@software: Game Prophet System)
@file: word2vec_pyspark.py
@time: 2018/11/10 17:28
"""
'''
    内存问题：spark默认的framesize仅为10M
    -- conf "spark.akak.frameSize=500"
    
'''
import os
import sys

try:
    from pyspark.ml.feature import Word2Vec, Word2VecModel
    from pyspark.ml.feature import StopWordsRemover
    from pyspark.ml.feature import Tokenizer
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    from pyspark_ml.app_root import get_root_path
except ImportError as e:
    sys.exit(1)

project_root_path = get_root_path()

spark = SparkSession.builder.appName('word2vec_pyspark').getOrCreate()
filepath = project_root_path + '/data/word2vec/word2vec_df.csv'
csv_lines = spark.read.csv(filepath, header='true', inferSchema='true', sep=',').toDF("id","sentence")
print(csv_lines)

# initializing a Tokenizer
tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
wordsDF = tokenizer.transform(csv_lines)
wordsDF.show(truncate=False)

# Initialize a StopWordsRemoval
remover = StopWordsRemover(inputCol='words', outputCol='filteredWords')

# invoke transform function
no_stopwords_df = remover.transform(wordsDF)
no_stopwords_df.show(truncate=False)

# output dataset showing only sentence and filtered words
no_stopwords_df.select("sentence", "filteredWords").show(n=5, truncate=False)

# initialize
word2vec = Word2Vec(inputCol="words", outputCol='wordvector', vectorSize=10, minCount=0, maxIter=100)

# fit
word2vecModel = word2vec.fit(no_stopwords_df)

# save model
modelPath = project_root_path + "/models/word2vec-model"
word2vecModel.save(modelPath)

# transform()
word2vec_df = word2vecModel.transform(no_stopwords_df)

# output dataset
word2vec_df.show(truncate=False)
