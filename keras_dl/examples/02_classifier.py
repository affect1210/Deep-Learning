#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: 02_classifier.py
@time: 2018/11/26 16:03
"""

from keras.datasets import mnist
from keras.utils import np_utils,plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation
#  Momentum(惯性原则) + AdaGrad(对错误反向的阻力) = RMSProp
from keras.optimizers import RMSprop
from keras_dl.app_root import get_root_path
app_root_path = get_root_path()
import numpy as np

np.set_printoptions(threshold=np.inf)

# X shape(60000,28*28) , y shape(10000,)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# print('X_train 1st image:', X_train[0])
# print('X_test 1st image:', X_test[0])
print('y_train:', y_train[:3])
print('y_train shape:', y_train.shape)
# print('y_test:', y_test[:3])
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# Another way to build your neural net
model = Sequential([Dense(32, input_dim=784), Activation('relu'), Dense(10), Activation('softmax')])


# save show model
model_file = app_root_path + '\classifier.png'
plot_model(model, show_shapes=True, to_file=model_file)

# Another way to define your optimizer
'''
lr: float >= 0. 学习率.
rho: float >= 0. RMSProp梯度平方的移动均值的衰减率.
epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
decay: float >= 0. 每次参数更新后学习率衰减值.
'''
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# define training
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

# define testing
loss,accuracy = model.evaluate(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)