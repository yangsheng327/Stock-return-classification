# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:55:23 2017
#artificial time series: 1. IID normal
                         2. GARCH
                         3. HMM
                         rule of generation: randomly sample from real
                         data, then fit with model, finally generate time
                         series with corresponding model, and normalize to
                         mean, std = 0, 1
#feature selection: 1. distribution: skew, kurtosis
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
T = 252

#data input: cat: us-0, jp-1, kr-2
us = pd.read_csv("us.csv")
jp = pd.read_csv("jp.csv")
kr = pd.read_csv("kr.csv")

#data_all has all the data
data_all = pd.concat((us, jp, kr))
data_all.index = range(data_all.shape[0])# data[sample, feature/days]

#data_real from real signal
real_n = 2#min(us.shape[0],jp.shape[0],kr.shape[0])
data_real = pd.concat((us.sample(real_n), jp.sample(real_n), 
                         kr.sample(real_n)))
data_real.index = range(data_real.shape[0])
data_real['cat'] = 0 #0 for real, 1 for artificial

#sample generation based on IID normal(0,1)  
def norm_gen_TS(N):
    return np.random.normal(0,1,[N,T])

#sample generation based on GARCH(1,1)     
def garch_gen_TS(N):    
    #generate one time series with GARCH(1,1)
    def garch_gen_ts(omega, alpha, beta):
        w = np.random.normal(size=T)
        eps = np.zeros_like(w)
        sigsq = np.zeros_like(w)
        eps[0] = w[0]
        for i in range(1,T):
            sigsq[i] = omega + alpha*(eps[i-1]**2) + beta*sigsq[i-1]
            eps[i] = w[i] * np.sqrt(sigsq[i])
        return eps  
        
    TS = np.zeros([N,T])
    for n in range(N):
        if n%10 == 0:
            p('generating {0}th sample by GARCH(1,1)'.format(n))
        r = np.random.randint(0, data_all.shape[0])
        ts_real = np.array(data_all.loc[r, '0':])
        am = arch_model(ts_real, p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(update_freq=5, disp='off')
        omega, alpha, beta = res.params[1:4]
        ts = garch_gen_ts(omega, alpha, beta)
        TS[n,:] = (ts-np.mean(ts))/np.std(ts)
    return TS 


#sample generation based on HMM 
def hmm_gen_TS(N, hmm_n=4):
    TS = np.zeros([N,T])
    model = GaussianHMM(n_components=hmm_n, covariance_type="diag", n_iter=1000)
    for n in range(N):
        if n%10 == 0:
            p('generating {0}th sample by HMM'.format(n))
        r = np.random.randint(0, data_all.shape[0])
        ts_real = np.array(data_all.loc[r, '0':])
        X = np.column_stack([ts_real])
        model.fit(X)
        ts, Z = model.sample(T)
        ts = ts[:,0]
        TS[n,:] = (ts-np.mean(ts))/np.std(ts)
    return TS

#select sampling method        
#data_hum = norm_gen_TS(3*real_n)
#data_hum = garch_gen_TS(3*real_n)
data_hum = hmm_gen_TS(3*real_n)
data_hum = pd.DataFrame(np.concatenate([np.ones([3*real_n,1]),data_hum],axis=1))
data_hum.columns =  data_real.columns
data =  pd.concat((data_real, data_hum))
data.index = range(data.shape[0])


def distr_feat_sk(ts):
    return np.array([scs.skew(ts),scs.kurtosis(ts)]) #skew, kurt

def distr_feat(ts):
    return np.array([np.mean(ts), np.std(ts), scs.skew(ts), 
                     scs.kurtosis(ts)]) #mean, std, skew, kurt   
                    
def sqr_dist_feat(ts):
    return distr_feat(ts**2)

def pacf_sqr_feat(ts):#first 5 pacf
    return pacf(ts**2)[1:6]
    
def garch_feat(ts):
    n = 1
    p_, o_, q_=n, 0, n
    am = arch_model(ts, p=p_, o=o_, q=q_, dist='StudentsT')
    res = am.fit(update_freq=5, disp='off')
    pars = res.params.as_matrix()
#    return pars[1:4] #this is the return for omega, alpha, beta
    return np.append(pars, distr_feat(res.conditional_volatility))
#information hidden in conditional volatility
    
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
#    return garch_feat(ts)

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
train_X_allfeat, train_Y = feat(data)
#features: statistical - [0:6], pac - [6:11], HMM - [11:35], garch - [35:44]        
train_X = train_X_allfeat[:,35:44]
