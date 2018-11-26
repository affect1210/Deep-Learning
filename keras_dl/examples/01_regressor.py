#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@time: 2018/11/23 10:25
"""
# import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
np.random.seed(1337)  # for reproduciblity
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(loc=0, scale=0.05, size=(200,))  # loc是均值 scale为标准差 size是要生成点的个数
# plot data
# plt.scatter(X,Y)
# plt.show()

X_train,Y_train = X[:160],Y[:160]
X_test,Y_test = X[160:],Y[160:]


# 0.1 build a neural network from the 1st layer to the last layer
model = Sequential()

model.add(Dense(units=1,input_dim=1))

# 0.2 choose loss function and optimizing method
model.compile(loss="mse",optimizer='sgd')

# 0.3 training
print("Training ...")
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step % 100 == 0:
        print('train cast:',cost)

# 0.4 test
print("\n Testing ...")
cost = model.evaluate(X_test,Y_test,batch_size=40)
print("test cost:",cost)
W,b = model.layers[0].get_weights()
print("Weights=",W,"\n biaes=",b)

# 0.5 plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()

































