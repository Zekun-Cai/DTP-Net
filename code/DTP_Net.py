# -*- coding: utf-8 -*-
'''
@Time    : 2020/06/24 20:26
@Author  : Zekun Cai
@File    : preddiffusion_DTM.py
@E-mail  : caizekun@csis.u-tokyo.ac.jp
'''

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, concatenate
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import backend as K
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import tensorflow as tf
from Param import *


def my_mse(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true))
    return mse_weight * mse


def my_mape(y_true, y_pred):
    mape = 100 * K.mean(K.abs((y_true - y_pred) / K.clip((K.abs(y_pred) + K.abs(y_true)) * 0.5, EPSILON, None)))
    return mape_weight * mape


def mse_mape(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true))
    mape = 100 * K.mean(K.abs((y_true - y_pred) / K.clip((K.abs(y_pred) + K.abs(y_true)) * 0.5, EPSILON, None)))
    loss = mse_weight * mse + mape_weight * mape
    return loss


class GraphConvolution(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',  # Gaussian distribution
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        input_dim = features_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(GraphConvolution, self).build(input_shapes)

    def call(self, inputs, mask=None):
        features = inputs[0]
        links = inputs[1]

        result = K.batch_dot(links, features, axes=[2, 1])
        output = K.dot(result, self.kernel)

        if self.bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


def sequence_GCN(input_seq, adj_seq, unit, act='relu', **kwargs):
    GCN = GraphConvolution(unit, activation=act, **kwargs)
    embed = []
    for n in range(input_seq.shape[2]):
        frame = Lambda(lambda x: x[:, :, n, :])(input_seq)
        adj = Lambda(lambda x: x[:, n, :, :])(adj_seq)
        embed.append(GCN([frame, adj]))
    output = Lambda(lambda x: tf.stack(x, axis=2))(embed)
    return output


def shared_ConvLSTM(x_in, filters, in_channel, return_sequences=True, activation='tanh'):
    x = Lambda(lambda x: K.reshape(x, (-1, TIMESTEP, HEIGHT, WIDTH, in_channel)))(x_in)
    x = ConvLSTM2D(filters, (convlstm_ks, convlstm_ks), padding='same',
                   return_sequences=return_sequences, activation=activation)(x)

    if return_sequences:
        x = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT * WIDTH, TIMESTEP, HEIGHT * WIDTH, filters)))(x)
    else:
        x = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT * WIDTH, HEIGHT * WIDTH, filters)))(x)

    return x


def getModel():
    ins, fused = [], []
    x_input = Input(batch_shape=(BATCHSIZE, HEIGHT * WIDTH, TIMESTEP, HEIGHT * WIDTH, CHANNEL))
    ins.append(x_input)
    fused.append(x_input)

    if Metadata:
        X_meta = Input(batch_shape=(BATCHSIZE, DAY_FEA))
        ins.append(X_meta)

        dens1 = Dense(units=10, activation='relu')(X_meta)
        dens2 = Dense(units=TIMESTEP * HEIGHT * WIDTH * 1, activation='relu')(dens1)
        hmeta = Reshape((TIMESTEP, HEIGHT * WIDTH, 1))(dens2)
        hmeta = Lambda(lambda x: K.concatenate([x[:, np.newaxis, :, :, :]] * HEIGHT * WIDTH, axis=1))(hmeta)
        fused.append(hmeta)

    if GCN:
        X_adj = Input(batch_shape=(BATCHSIZE, TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH))
        X_seman = Input(batch_shape=(BATCHSIZE, HEIGHT * WIDTH, TIMESTEP, local_window * local_window, CHANNEL))
        ins.append(X_adj), ins.append(X_seman)

        order = []
        for i in range(CHANNEL):
            order_fea = Lambda(lambda x: x[:, :, :, :, i])(X_seman)
            order_gcn = sequence_GCN(order_fea, X_adj, HEIGHT * WIDTH)
            order.append(order_gcn)
        X_gcn = Lambda(lambda x: tf.stack(x, axis=-1))(order)
        fused.append(X_gcn)

    if Metadata or GCN:
        x = concatenate(fused, axis=-1)
    else:
        x = x_input
    in_channel = CHANNEL + (CHANNEL if GCN else 0) + (1 if Metadata else 0)

    x = shared_ConvLSTM(x, convlstm_filter, in_channel, return_sequences=True, activation='tanh')

    if MultiTask:
        outs = []
        for i in range(CHANNEL):
            out = shared_ConvLSTM(x, 1, convlstm_filter, return_sequences=False, activation='relu')
            outs.append(out)
    else:
        outs = shared_ConvLSTM(x, CHANNEL, convlstm_filter, return_sequences=False, activation='relu')

    model = Model(inputs=ins, outputs=outs)
    model.compile(loss=mse_mape, optimizer=OPTIMIZER)
    model.summary()
    return model

if __name__=='__main__':
    model = getModel()