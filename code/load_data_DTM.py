# -*- coding: utf-8 -*-
'''
@Time    : 2020/06/24 20:26
@Author  : Zekun Cai
@File    : preddiffusion_DTM.py
@E-mail  : caizekun@csis.u-tokyo.ac.jp
'''
import numpy as np
from sklearn.preprocessing import normalize
from Param import *


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    try:
        lam = np.linalg.eigvals(L).max().real
    except:
        lam = 2
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def data_generator(data, temporal_data, batch, return_meta=True, return_multi=True, return_gcn=True,
                   desc_type='Local', adj_type='Semantic'):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP
    seed = 0

    if adj_type == 'Dis':
        adj = np.load(dis_adjFile)
        adj = 1 / adj
        adj[adj == np.inf] = 0
        adj[range(adj.shape[0]), range(adj.shape[1])] = 1
        adj = normalize(adj, norm='l1')
        adj = np.array([adj for i in range(TIMESTEP)])
        adj = np.array([adj for i in range(BATCHSIZE)])
    elif adj_type == 'Semantic':
        adj = np.load(hisflow_adjFile)
        adj = normalize(adj, norm='l1')
        adj = np.array([adj for i in range(TIMESTEP)])
        adj = np.array([adj for i in range(BATCHSIZE)])

    while True:

        time_random = np.arange(num_time)
        np.random.seed(seed)
        np.random.shuffle(time_random)
        seed += 1

        trainX, temporal, desc, trainY = [], [], [], []
        batch_num = 0

        for t in time_random:
            x = data[t:t + TIMESTEP]
            x = x.transpose((1, 0, 2, 3))
            y = data[t + TIMESTEP]

            # Build Grid Descriptor
            if return_gcn:
                if desc_type == 'Local':
                    x_temp = x.reshape((x.shape[0], x.shape[1], HEIGHT, WIDTH, x.shape[-1]))
                    window = np.zeros(shape=(x_temp.shape[0], x_temp.shape[1],
                                             local_window * local_window, x_temp.shape[-1]))
                    pad_x = np.pad(x_temp, ((0, 0), (0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
                    for n in range(x.shape[0]):
                        loc_x = n // 20
                        loc_y = n % 20
                        cut = pad_x[n, :, loc_x:loc_x + local_window, loc_y:loc_y + local_window, :]
                        window[n] = cut.reshape((cut.shape[0], local_window * local_window, cut.shape[-1]))
                elif desc_type == 'Global':
                    window = x
                else:
                    raise NameError('Wrong Descriptor Name!')
            else:
                window = None

            temp_fea = temporal_data[t + TIMESTEP]
            trainX.append(x), temporal.append(temp_fea), desc.append(window), trainY.append(y)
            batch_num += 1

            if batch_num == batch:
                trainX, temporal, desc, trainY = \
                    np.array(trainX), np.array(temporal), np.array(desc), np.array(trainY)

                if return_multi:
                    Y = [trainY[:, :, :, i:i + 1] for i in range(CHANNEL)]
                else:
                    Y = trainY

                X = [trainX]
                if return_meta:
                    X.append(temporal)
                if return_gcn:
                    X.append(adj)
                    X.append(desc)
                yield X, Y

                batch_num = 0
                trainX, temporal, desc, trainY = [], [], [], []


def test_generator(data, temporal_data, batch, return_meta=True,
                   return_gcn=True, desc_type='Local', adj_type='Semantic'):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP

    testX, temporal, desc = [], [], []
    batch_num = 0

    if adj_type == 'Dis':
        adj = np.load(dis_adjFile)
        adj = 1 / adj
        adj[adj == np.inf] = 0
        adj[range(adj.shape[0]), range(adj.shape[1])] = 1
        adj = normalize(adj, norm='l1')
        adj = np.array([adj for i in range(TIMESTEP)])
        adj = np.array([adj for i in range(BATCHSIZE)])
    elif adj_type == 'Semantic':
        adj = np.load(hisflow_adjFile)
        adj = normalize(adj, norm='l1')
        adj = np.array([adj for i in range(TIMESTEP)])
        adj = np.array([adj for i in range(BATCHSIZE)])

    for t in range(num_time):
        x = data[t:t + TIMESTEP]
        x = x.transpose((1, 0, 2, 3))

        # Build Grid Descriptor
        if return_gcn:
            if desc_type == 'Local':
                x_temp = x.reshape((x.shape[0], x.shape[1], HEIGHT, WIDTH, x.shape[-1]))
                window = np.zeros(shape=(x_temp.shape[0], x_temp.shape[1],
                                         local_window * local_window, x_temp.shape[-1]))
                pad_x = np.pad(x_temp, ((0, 0), (0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
                for n in range(x.shape[0]):
                    loc_x = n // 20
                    loc_y = n % 20
                    cut = pad_x[n, :, loc_x:loc_x + local_window, loc_y:loc_y + local_window, :]
                    window[n] = cut.reshape((cut.shape[0], local_window * local_window, cut.shape[-1]))
            elif desc_type == 'Global':
                window = x
            else:
                raise NameError('Wrong Descriptor Name!')
        else:
            window = None

        temp_fea = temporal_data[t + TIMESTEP]
        testX.append(x), temporal.append(temp_fea), desc.append(window)
        batch_num += 1

        if batch_num == batch:
            testX, temporal, desc = np.array(testX), np.array(temporal), np.array(desc)

            X = [testX]
            if return_meta:
                X.append(temporal)
            if return_gcn:
                X.append(adj)
                X.append(desc)
            yield X

            batch_num = 0
            testX, temporal, desc = [], [], []


def get_true(data):
    return data[TIMESTEP:]
