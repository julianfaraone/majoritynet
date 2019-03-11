# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from tensorpack import *
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected)

import numpy as np


@layer_register()
def Conv2D_mask(inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        mask=None):

    # group conv implementation
    data_format = get_data_format(data_format, tfmode=False)
    in_shape = inputs.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

    out_channel = filters
    assert dilation_rate == (1, 1) or get_tf_version_tuple() >= (1, 5), 'TF>=1.5 required for group dilated conv'

    kernel_shape = shape2d(kernel_size)
    filter_shape = kernel_shape + [in_channel, out_channel]
    stride = shape4d(strides, data_format=data_format)

    kwargs = dict(data_format=data_format)
    if get_tf_version_tuple() >= (1, 5):
        kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

    W = tf.get_variable(
        'W', filter_shape, initializer=kernel_initializer)

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

    #md = calc_mask_density(mask)
    if mask is not None:
        #logger.info("Apply mask with density=%.2f"%(md))
        m = tf.constant(mask)
        W = W*m

    kernels = W
    #outputs = [tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs) for i, k in zip(inputs, kernels)]
    outputs = [tf.nn.conv2d(inputs, kernels, stride, padding.upper(), **kwargs)]
    conv = tf.concat(outputs, channel_axis)
    if activation is None:
        activation = tf.identity
    ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret

@layer_register()
def Conv2D_matmul(inputs,
        filters,
        kernel_size,
        strides=1,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        data_format="channels_last"
        ):

    # only works for channels last
    assert (data_format == 'NHWC' or data_format == "channels_last"), "error: need to transpose inputs"

    data_format = get_data_format(data_format, tfmode=False)
    in_shape = inputs.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    out_channel = filters

    # get filter weights and flatten
    kernel_shape = shape2d(kernel_size)
    filter_shape = kernel_shape + [in_channel, out_channel]
    W = tf.get_variable('W', filter_shape, 
        initializer=kernel_initializer)
    W_f = tf.reshape(W, [kernel_size*kernel_size*in_channel, out_channel])

    # get input patches and flatten
    patches = tf.extract_image_patches(
        inputs,
        ksizes=[1, kernel_size, kernel_size, 1],
        strides=[1, strides, strides, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
        )
    patches_f = tf.reshape(patches, [-1, in_shape[1]//strides, in_shape[2]//strides, 
        kernel_size*kernel_size*in_channel])
    patches_f = tf.reshape(patches_f, [-1, kernel_size*kernel_size*in_channel])

    # This is about 3x slower
    feature_maps = tf.matmul(patches_f, W_f)
    features = tf.reshape(feature_maps, 
        [-1, in_shape[1]//strides, in_shape[2]//strides, out_channel] )

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

    if activation is None:
        activation = tf.identity
    ret = activation(tf.nn.bias_add(features, b, data_format=data_format) if use_bias else features, name='output')

    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret



@layer_register()
def Conv2D_muladd(inputs,
        filters,
        kernel_size,
        strides=1,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        data_format="channels_last"
        ):
    # Takes very long

    # only works for channels last
    assert (data_format == 'NHWC' or data_format == "channels_last"), "error: need to transpose inputs"

    data_format = get_data_format(data_format, tfmode=False)
    in_shape = inputs.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    out_channel = filters

    # get filter weights and flatten
    kernel_shape = shape2d(kernel_size)
    filter_shape = kernel_shape + [in_channel, out_channel]
    W = tf.get_variable('W', filter_shape, 
        initializer=kernel_initializer)
    W_f = tf.reshape(W, [kernel_size*kernel_size*in_channel, out_channel])

    # get input patches and flatten
    patches = tf.extract_image_patches(
        inputs,
        ksizes=[1, kernel_size, kernel_size, 1],
        strides=[1, strides, strides, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
        )
    patches_f = tf.reshape(patches, [-1, in_shape[1]//strides, in_shape[2]//strides, 
        kernel_size*kernel_size*in_channel])
    patches_f = tf.reshape(patches_f, [-1, kernel_size*kernel_size*in_channel])

    # for each patch, right multiply the filter matrix and the image patch vector
    # This is about 40x slower
    #feature_maps = []
    #for i in range(out_channel):
    #    feature_map = tf.reduce_sum(tf.multiply(W_f[:, i], patches_f), axis=1, keep_dims=True)
    #    feature_maps.append(feature_map)
    #feature_maps = tf.concat(feature_maps, axis=1)

    feature_maps = tf.reduce_sum(tf.multiply(tf.expand_dims(patches_f, 1), tf.transpose(W_f, [1,0])), axis=2)


    features = tf.reshape(feature_maps, 
        [-1, in_shape[1]//strides, in_shape[2]//strides, out_channel] )

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

    if activation is None:
        activation = tf.identity
    ret = activation(tf.nn.bias_add(features, b, data_format=data_format) if use_bias else features, name='output')

    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret
