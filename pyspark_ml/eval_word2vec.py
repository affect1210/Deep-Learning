#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: eval_word2vec.py
@time: 2018/11/10 19:22
"""
import numpy as np
import pandas as pd
import time
import math
from pyspark.ml.feature import Word2VecModel
from pyspark_ml.app_root import get_root_path
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from mpl_toolkits.mplot3d import Axes3D

project_root_path = get_root_path()
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('evel_word2vec').getOrCreate()

#load model
modelPath = project_root_path+"/models/word2vec-model"
loadedModel = Word2VecModel.load(modelPath)
# loadedModel.findSynonyms('beijing',15).show(truncate=False)
wordVectorsDF=loadedModel.getVectors()
vocabSize = wordVectorsDF.count()
print("Vocabulary Size: ", vocabSize)
loadedModel.findSynonyms('coffee',3).show(truncate=False)



# loadedModel.getVectors().show(truncate=False)

dfW2V = wordVectorsDF.select('vector').withColumnRenamed('vector','features')

numComponents = 3
pca = PCA(k = numComponents, inputCol = 'features', outputCol = 'pcaFeatures')
model = pca.fit(dfW2V)
dfComp = model.transform(dfW2V).select("pcaFeatures")


def topNwordsToPlot(dfComp, wordVectorsDF, word, nwords):
    compX = np.asarray(dfComp.rdd.map(lambda vec: vec[0][0]).collect())
    compY = np.asarray(dfComp.rdd.map(lambda vec: vec[0][1]).collect())
    compZ = np.asarray(dfComp.rdd.map(lambda vec: vec[0][2]).collect())

    words = np.asarray(wordVectorsDF.select('word').toPandas().values.tolist())
    Feat = np.asarray(wordVectorsDF.select('vector').rdd.map(lambda v: np.asarray(v[0])).collect())

    Nw = words.shape[0]  # total number of words
    ind_star = np.where(word == words)  # find index associated to 'word'
    wstar = Feat[ind_star, :][0][0]  # vector associated to 'word'
    nwstar = math.sqrt(np.dot(wstar, wstar))  # norm of vector assoicated with 'word'

    dist = np.zeros(Nw)  # initialize vector of distances
    i = 0
    for w in Feat:  # loop to compute cosine distances between 'word' and the rest of the words
        den = math.sqrt(np.dot(w, w)) * nwstar  # denominator of cosine distance
        dist[i] = abs(np.dot(wstar, w)) / den  # cosine distance to each word
        i = i + 1

    indexes = np.argpartition(dist, -(nwords + 1))[-(nwords + 1):]
    di = []
    for j in range(nwords + 1):
        di.append((words[indexes[j]], dist[indexes[j]], compX[indexes[j]], compY[indexes[j]], compZ[indexes[j]]))

    result = []
    for elem in sorted(di, key=lambda x: x[1], reverse=True):
        result.append((elem[0][0], elem[2], elem[3], elem[4]))

    return pd.DataFrame(result, columns=['word', 'X', 'Y', 'Z'])

def show3D():
    word = 'coffee'
    nwords = 5

    #############

    r = topNwordsToPlot(dfComp, wordVectorsDF, word, nwords)

    ############
    fs = 20  # fontsize
    w = r['word']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    height = 10
    width = 10
    fig.set_size_inches(width, height)

    ax.scatter(r['X'], r['Y'], r['Z'], color='red', s=100, marker='o', edgecolors='black')
    for i, txt in enumerate(w):
        if (i < 5):
            ax.text(r['X'].ix[i], r['Y'].ix[i], r['Z'].ix[i], '%s' % (txt), size=30, zorder=1, color='k')

    ax.set_xlabel('1st. Component', fontsize=fs)
    ax.set_ylabel('2nd. Component', fontsize=fs)
    ax.set_zlabel('3rd. Component', fontsize=fs)
    ax.set_title('Visualization of Word2Vec via PCA', fontsize=fs)
    ax.grid(True)
    plt.show()

show3D()


t0 = time.time()

K = int(math.floor(math.sqrt(float(vocabSize) / 2)))
# K ~ sqrt(n/2) this is a rule of thumb for choosing K,
# where n is the number of words in the model
# feel free to choose K with a fancier algorithm

dfW2V = wordVectorsDF.select('vector').withColumnRenamed('vector', 'features')
kmeans = KMeans(k=K, seed=1)
modelK = kmeans.fit(dfW2V)
labelsDF = modelK.transform(dfW2V).select('prediction').withColumnRenamed('prediction', 'labels')


print("Number of Clusters (K) Used: ", K)
print("Elapsed time (seconds) :", time.time() - t0)