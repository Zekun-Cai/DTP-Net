# -*- coding: utf-8 -*-
'''
@Time    : 2020/06/24 20:26
@Author  : Zekun Cai
@File    : preddiffusion_DTM.py
@E-mail  : caizekun@csis.u-tokyo.ac.jp
'''
import numpy as np
from Param import *

def mse_func(y_true, y_pred):
    return np.mean(np.array((y_pred - y_true).todense()) ** 2)


def rmse_func(y_true, y_pred):
    return np.sqrt(np.mean(np.array((y_pred - y_true).todense()) ** 2))


def mae_func(y_true, y_pred):
    return np.mean(np.abs(np.array((y_pred - y_true).todense())))


def mape_func(y_true, y_pred):
    y_true = np.array(y_true.todense())
    y_pred = np.around(np.array(y_pred.todense()))

    y_true = np.clip(y_true, a_min=EPSILON, a_max=None)
    y_pred = np.clip(y_pred, a_min=EPSILON, a_max=None)
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)) * 0.5)
