#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: pca_example.py
@time: 2018/11/9 17:56
"""
import os
import sys
import pyspark
"""
PAC  - Principal Component Analysis
       主成分分析的pyspark.ml的样例
提取最有价值的信息（基于方差）
数据的主成分（即特征向量）与它们的权值（即特征值）
基变换
如何找到最合适的基？
协方差矩阵
希望投影后的投影值尽可能的分散
协方差


"""
try:
    from pyspark.ml.feature import PCA
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import SparkSession

    print("Successfully  imported Spark Modules")
except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

spark = SparkSession.builder.appName("PACExample").getOrCreate()

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
print(data)
df = spark.createDataFrame(data, ["features"])
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
# 表示用数据df来训练PCA模型
model = pca.fit(df)
# 当模型训练好后,对于新输入的数据,都可以用transform方法来降维.
result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
spark.stop()

