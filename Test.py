# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:25:21 2017

@author: Quentin
"""

## ======= Test NeuralNet Library
from NNLib import NNet
import scipy.io
import numpy as np
import time
start=time.time()

mat=scipy.io.loadmat('data.mat')

X=mat['data']['training'][0][0]['inputs'][0][0]
y=mat['data']['training'][0][0]['targets'][0][0]
print('X',X.shape,'y',y.shape)

n_output=[10]
n_hid=[7]
n_input=[256]
sizes=np.concatenate([n_input,n_hid,n_output])
type_layer=['logistic','logistic','softmax']
threshold=1e-4
max_iter=1000

min_training_loss=np.Inf
min_val_loss=np.Inf
min_test_loss=np.Inf

i=0
for alpha in [0.35]:#[0.002,0.01,0.05,0.2,1,5]:
    for mom in [0.9]:
        for wd in [0]:#[0.001,0,1,0.1,10,0.0001]:
            for n_hid in [[30,20],[15,15],[40,30],[10,15],[20,20]]:
                i=i+1
                print('==================== Test n°',i,' alpha=',alpha,' mom=',mom,' wd=',wd,' n_hid=',n_hid)
                sizes=np.concatenate([n_input,n_hid,n_output])
                [model,losses]=NNet.build_model(X,y,wd,sizes,type_layer,'classification', max_iter, alpha, mom, True,100,threshold,True)
                if losses[0]<min_training_loss:
                    min_training_loss=losses[0]
                    min_alpha=alpha
                    momentum=mom
                    wd_t=wd
                    n_hid_t=n_hid
                if losses[1]<min_val_loss:
                    min_val_loss=losses[1]
                    min_alpha_val=alpha
                    momentum_val=mom
                    wd_v=wd
                    n_hid_v=n_hid
                if losses[2]<min_test_loss:
                    min_test_loss=losses[2]
                    min_alpha_test=alpha
                    momentum_test=mom
                    wd_test=wd
                    n_hid_test=n_hid
                    # best_class_perf=class_perf[2]

print(i, ' configurations have been tested' )

print('==== Training test')
print('Min alpha: ',min_alpha)
print('Momentum: ',momentum)
print('Min val loss: ',min_training_loss)
print('WD: ',wd_t)
print('n_hid: ',n_hid_t)

print('==== Validation test')
print('Min alpha: ',min_alpha_val)
print('Momentum: ',momentum_val)
print('Min val loss: ',min_val_loss)
print('WD: ',wd_v)
print('n_hid: ',n_hid_v)

print('==== Generalisation test')
print('Min alpha: ',min_alpha_test)
print('Momentum: ',momentum_test)
print('Min val loss: ',min_test_loss)
print('WD: ',wd_test)
print('n_hid: ',n_hid_test)
# print('Classification performance: ',best_class_perf)

print('\nElapsed time:',time.time()-start)