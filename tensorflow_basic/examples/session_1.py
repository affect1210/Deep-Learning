#!/usr/bin/env python
# -*- coding:utf-8 -*

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2) # matrix multiply

# method 1
session = tf.Session()
result = session.run(product)

print("method1: %s"%result)
session.close()

#method 2
with tf.Session() as session:
    result2 = session.run(product)
    print("method2: %s"%result2)

