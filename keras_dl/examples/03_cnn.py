#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: 03_cnn.py
@time: 2018/11/26 18:41
"""
# Convolutional Neural NetWork
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils, plot_model
from keras_dl.app_root import get_root_path
from keras.optimizers import Adam

app_root_path = get_root_path()

# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1,1,28,28) / 255
X_test = X_test.reshape(-1,1,28,28) / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# 0.1 build CNN
model = Sequential()

# 0.2 Conv layer 1
model.add(Convolution2D(batch_input_shape=(None, 1, 28, 28), filters=32, kernel_size=5, strides=1, padding='same',
                        data_format='channels_first'))

model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape(32,14,14)
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

# Conv layer 2 output shape (64,14,14)
model.add(Convolution2D(
    64, 5, strides=1,
    padding='same', data_format='channels_first'
))
model.add(Activation('relu'))
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))
# Flatten Layer 1
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Flatten Layer 2
model.add(Dense(10))
model.add(Activation('softmax'))

# Optimizer
adam = Adam(lr=1e-4)

# add Metrics and Compile
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

# save show model
model_file = app_root_path + '\cnn.png'
plot_model(model, show_shapes=True, to_file=model_file)

print('Training -------------- ')
# train model
model.fit(X_train, y_train, epochs=2, batch_size=64,)

print('\n Testing ---------------')
# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print('\n test loss:',loss)
print('\n test accuracy:',accuracy)




