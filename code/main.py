# -*- coding: utf-8 -*-
'''
@Time    : 2020/06/24 20:26
@Author  : Zekun Cai
@File    : preddiffusion_DTM.py
@E-mail  : caizekun@csis.u-tokyo.ac.jp
'''

import sys
import shutil
import os
import time
from datetime import datetime
import scipy.sparse as ss
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from DTP_Net import *
from load_data_DTM import *
from metric import *
from Param import *
###########################Reproducible#############################
import random

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.set_random_seed(100)


###################################################################

def testModel(name, testData, dayinfo):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = getModel()
    model.load_weights(PATH + '/' + name + '.h5')

    test_gene = test_generator(testData, dayinfo, batch=BATCHSIZE, return_meta=Metadata,
                               return_gcn=GCN, desc_type=Descriptor, adj_type=AdjType)
    test_step = (testData.shape[0] - TIMESTEP) // BATCHSIZE
    testY = get_true(testData)

    pred = model.predict_generator(test_gene, steps=test_step, verbose=1)
    pred = np.concatenate(pred, axis=-1)
    print('pred shape: {}'.format(pred.shape))
    pred_sparse = ss.csr_matrix(pred.reshape(pred.shape[0], -1))
    re_pred_sparse, re_testY = pred_sparse * MAX_DIFFUSION, testY * MAX_DIFFUSION
    mse_score = mse_func(re_testY, re_pred_sparse)
    rmse_score = rmse_func(re_testY, re_pred_sparse)
    mae_score = mae_func(re_testY, re_pred_sparse)
    mape_score = mape_func(re_testY, re_pred_sparse)

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Rescaled MSE on testData, {}\n".format(mse_score))
    f.write("Rescaled RMSE on testData, {}\n".format(rmse_score))
    f.write("Rescaled MAE on testData, {}\n".format(mae_score))
    f.write("Rescaled MAPE on testData, {:.3f}%\n".format(100 * mape_score))
    f.close()

    print('*' * 40)
    print('MSE', mse_score)
    print('RMSE', rmse_score)
    print('MAE', mae_score)
    print('MAPE {:.3f}%'.format(100 * mape_score))
    print('Model Evaluation Ended ...', time.ctime())

    predictionDiffu = re_pred_sparse
    groundtruthDiffu = re_testY
    ss.save_npz(PATH + '/' + MODELNAME + '_prediction.npz', predictionDiffu)
    ss.save_npz(PATH + '/' + MODELNAME + '_groundtruth.npz', groundtruthDiffu)


def trainModel(name, trainData, dayinfo):
    print('Model Training Started ...', time.ctime())

    train_num = int(trainData.shape[0] * (1 - SPLIT))
    print('train num: {}, val num: {}'.format(train_num, trainData.shape[0] - train_num))

    train_gene = data_generator(trainData[:train_num], dayinfo[:train_num], BATCHSIZE,
                                return_meta=Metadata, return_multi=MultiTask,
                                return_gcn=GCN, desc_type=Descriptor, adj_type=AdjType)
    val_gene = data_generator(trainData[train_num:], dayinfo[train_num:], BATCHSIZE,
                              return_meta=Metadata, return_multi=MultiTask,
                              return_gcn=GCN, desc_type=Descriptor, adj_type=AdjType)
    train_step = (train_num - TIMESTEP) // BATCHSIZE
    val_step = (trainData.shape[0] - train_num - TIMESTEP) // BATCHSIZE

    model = getModel()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit_generator(train_gene, steps_per_epoch=train_step, epochs=EPOCH,
                        validation_data=val_gene, validation_steps=val_step,
                        callbacks=[csv_logger, checkpointer, LR, early_stopping])
    pred = model.predict_generator(test_generator(trainData[train_num:], dayinfo[train_num:],
                                                  BATCHSIZE, return_meta=Metadata, return_gcn=GCN,
                                                  desc_type=Descriptor, adj_type=AdjType), steps=val_step)
    pred = np.concatenate(pred, axis=-1)
    print('pred shape: {}'.format(pred.shape))
    pred_sparse = ss.csr_matrix(pred.reshape(pred.shape[0], -1))
    valY = get_true(trainData[train_num:])

    re_pred_sparse, re_valY = pred_sparse * MAX_DIFFUSION, valY * MAX_DIFFUSION
    mse_score = mse_func(re_valY, re_pred_sparse)
    rmse_score = rmse_func(re_valY, re_pred_sparse)
    mae_score = mae_func(re_valY, re_pred_sparse)
    mape_score = mape_func(re_valY, re_pred_sparse)

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Rescaled MSE on valData, {}\n".format(mse_score))
    f.write("Rescaled RMSE on valData, {}\n".format(rmse_score))
    f.write("Rescaled MAE on valData, {}\n".format(mae_score))
    f.write("Rescaled MAPE on valData, {:.3f}%\n".format(100 * mape_score))
    f.close()

    print('*' * 40)
    print('MSE', mse_score)
    print('RMSE', rmse_score)
    print('MAE', mae_score)
    print('MAPE {:.3f}%'.format(100 * mape_score))
    print('Model Train Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'Shared_{}{}{}'.format('Metadata_' if Metadata else None,
                                   'Multitask_' if MultiTask else None,
                                   Descriptor + AdjType if GCN else None)
KEYWORD = 'preddiffusion_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M%S")
PATH = '../' + KEYWORD


##########################################################


def main():
    param = sys.argv
    if len(param) == 2:
        GPU = param[-1]
    else:
        GPU = '0'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
    set_session(tf.Session(graph=tf.get_default_graph(), config=config))

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('load_data_DTM.py', PATH)
    shutil.copy2('metric.py', PATH)

    diffusion_data = ss.load_npz(diffusionFile)
    diffusion_data = diffusion_data / MAX_DIFFUSION
    dayinfo = np.genfromtxt(dayinfoFile, delimiter=',', skip_header=1)
    print('data.shape, dayinfo.shape', diffusion_data.shape, dayinfo.shape)
    train_Num = int(diffusion_data.shape[0] * trainRatio)

    print(KEYWORD, 'training started', time.ctime())
    trainvalidateData = diffusion_data[:train_Num]
    trainvalidateDay = dayinfo[:train_Num, ]
    print('trainvalidateData.shape', trainvalidateData.shape)
    trainModel(MODELNAME, trainvalidateData, trainvalidateDay)

    print(KEYWORD, 'testing started', time.ctime())
    testData = diffusion_data[train_Num:]
    testDay = dayinfo[train_Num:]
    print('testData.shape', testData.shape)
    testModel(MODELNAME, testData, testDay)


if __name__ == '__main__':
    main()
