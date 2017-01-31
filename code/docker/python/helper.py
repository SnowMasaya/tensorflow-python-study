# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf


def weight_variable(shape, name=None):
    """
    Setting the Weiught Value
    :param shape(int): Setting the weight dimension
    :param name(str): Setting Weight Name
    :return: weight value
    """
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=""):
    """
    Setting the Bias Name
    :param shape(int): Setting the bias dimension
    :param name(str): Setting Bias Name
    :return: bias value
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    """
    convoluation 2 dimension
    :param x: input value
    :param W: weight value
    :return: convoloation value
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    """
     Pooling Value
    :param x: input value
    :return: pooling value
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def prewitt_filter():
    """

    :return:
    """
    v = np.array([[1, 0, -1]] * 3)
    h = v.swapaxes(0, 1)
    f = np.zeros(3 * 3 * 1 * 2).reshape(3, 3, 1, 2)
    f[:, :, 0, 0] = v
    f[:, :, 0, 1] = h
    return tf.constant(f, dtype = tf.float32)
