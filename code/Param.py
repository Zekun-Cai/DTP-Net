# -*- coding: utf-8 -*-
'''
@Time    : 2020/06/24 20:26
@Author  : Zekun Cai
@File    : preddiffusion_DTM.py
@E-mail  : caizekun@csis.u-tokyo.ac.jp
'''

dataSet = 'bousai_tokyo'
dataPath = '../data/{}/'.format(dataSet)
diffusionFile = dataPath + 'all_60min_6.npz'
dayinfoFile = dataPath + 'day_info_1h.csv'
dis_adjFile = dataPath + 'dis_adjacent.npy'
hisflow_adjFile = dataPath + 'hisflow_adjacent.npy'

HEIGHT = 20
WIDTH = 20
MAX_DIFFUSION = ****
CHANNEL = 6
INTERVAL = 60
DAYSTEP = int(24 * 60 / INTERVAL)
WEEKTIMESTEP = int(7 * 24 * 60 / INTERVAL)
DAY_FEA = 33

shared_layer = 1
gcn_layer = 1
convlstm_filter = 32
convlstm_ks = 3
local_window = 5
padding = local_window // 2
mse_weight = 1e8
mape_weight = 1

Metadata = True
MultiTask = True
GCN = True
Descriptor = 'Local'  # Global or Local
AdjType = 'Semantic'  # Dis or Semantic

TIMESTEP = 6
EPSILON = 1e-5
trainRatio = 0.8
SPLIT = 0.2
LOSS = 'mse'
OPTIMIZER = 'adam'
LEARN = 0.001
BATCHSIZE = 1
EPOCH = 200
