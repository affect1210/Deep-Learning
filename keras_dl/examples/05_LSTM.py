#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0.0
@author: tianhonghan
@license: COPYRIGHT @ 2015 CHANGYOU.COM LIMITED
@contact: tianhonghan@cyou-inc.com
@site: http://www.changyou.com
@software: Game Prophet System)
@file: 05_LSTM.py
@time: 2018/12/4 16:02
"""
import numpy as np

np.random.seed(1337)
from keras_dl.app_root import get_root_path
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

app_root_path = get_root_path()

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (1 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    plt.show()
    # return

get_batch()