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

    :param shape:
    :param name:
    :return:
    """
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=""):
    """

    :param shape:
    :param name:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    """

    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    """

    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

