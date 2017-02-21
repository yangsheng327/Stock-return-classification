# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:07:49 2017
##feature selection: 1. distribution: skew, kurtosis
                     2. distribution square: mean, std, skew, kurtosis
                     3. pac: first several parameters
                     4. GARCH: parameters, distribution of conditional volatility
                     5. HMM: transition matrix, latent states ranked by mean
@author: Sunrise
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import scipy.stats as scs
from statsmodels.tsa.stattools import pacf
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

import matplotlib.pyplot as plt
plt.close('all')
p = print

#data input: cat: us-0, jp-1, kr-2
us = pd.read_csv("us.csv")
jp = pd.read_csv("jp.csv")
kr = pd.read_csv("kr.csv")

#data_all has all the data
data_all = pd.concat((us, jp, kr))
data_all.index = range(data_all.shape[0])# data[sample, feature/days]
#data_sample has same amount of each category
sample_n = 400#min(us.shape[0],jp.shape[0],kr.shape[0])
data_sample = pd.concat((us.sample(sample_n), jp.sample(sample_n), 
                         kr.sample(sample_n)))
data_sample.index = range(data_sample.shape[0])

def distr_feat_sk(ts):
    return np.array([scs.skew(ts),scs.kurtosis(ts)]) #skew, kurt

def distr_feat(ts):
    return np.array([np.mean(ts), np.std(ts), scs.skew(ts), 
                     scs.kurtosis(ts)]) #mean, std, skew, kurt   
                    
def sqr_dist_feat(ts):
    return distr_feat(ts**2)

def pacf_sqr_feat(ts):#first 5 pacf, except 0
    return pacf(ts**2)[1:6]
    
def garch_feat(ts):
    n = 1
    p_, o_, q_=n, 0, n
    am = arch_model(ts, p=p_, o=o_, q=q_, dist='StudentsT')
    res = am.fit(update_freq=5, disp='off')
    pars = res.params.as_matrix()
    return np.append(pars, distr_feat(res.conditional_volatility))
    
def HMM_feat(ts, hmm_n=4):
    X = np.column_stack([ts])
    model = GaussianHMM(n_components=hmm_n, covariance_type="diag", 
                        n_iter=1000)
    model.fit(X)
    id = np.argsort(model.means_, axis=0).T[0]
    return np.concatenate((np.reshape(model.transmat_[id,:][:,id], -1), 
                                     np.reshape(model.covars_[id], -1), 
                                                np.reshape(model.means_[id], -1)))
    
def feat_collect(ts):
    return np.concatenate((distr_feat_sk(ts), sqr_dist_feat(ts), 
                           pacf_sqr_feat(ts), HMM_feat(ts), garch_feat(ts)))    

def feat(data):
    N, P = data.shape[0], feat_collect(data.loc[0,'0':]).shape[0]
    train_X = np.zeros([N, P])
    train_Y = np.chararray([N])
    for n in range(N):
        if n%10 == 0:
            p('processing {0}th sample'.format(n))
        train_X[n,:] =  feat_collect(data.loc[n,'0':])
        train_Y[n] = data.loc[n,'cat']
    return train_X, train_Y   

    
#train_X, train_Y = feat(data_all)
train_X_allfeat, train_Y = feat(data_sample)
#features: statistical - [0:6], pac - [6:11], HMM - [11:35], garch - [35:44]        
train_X = train_X_allfeat[:,:]