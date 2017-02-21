# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:01:44 2017
##Time series analysis for artificial detection
##Given time series, plot: 1. signal, 2.autocorrelation, 
                           3. particial autocorrelation, 
                           4. Quantile vs. Quantile
@author: Sunrise
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

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
real_n = 100#min(us.shape[0],jp.shape[0],kr.shape[0])
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
        TS[n,:] = garch_gen_ts(omega, alpha, beta)
    return TS 


#sample generation based on HMM 
def hmm_gen_TS(N, hmm_n=4):
    TS = np.zeros([N,T])
    model = GaussianHMM(n_components=hmm_n, covariance_type="diag", n_iter=1000)
    for n in range(N):
        if n%10 == 0:
            p('generating {0}th sample by HMM({0})'.format(n, hmm_n))
        r = np.random.randint(0, data_all.shape[0])
        ts_real = np.array(data_all.loc[r, '0':])
        X = np.column_stack([ts_real])
        model.fit(X)
        ts, Z = model.sample(T)
        TS[n,:] = ts[:,0]
    return TS

#select sampling method        
data_hum = norm_gen_TS(3*real_n)
#data_hum = garch_gen_TS(3*real_n)
#data_hum = hmm_gen_TS(3*real_n)
data_hum = pd.DataFrame(np.concatenate([np.ones([3*real_n,1]),data_hum],axis=1))
data_hum.columns =  data_real.columns

#plotting
def tsplot(y, lags=30, figsize=(10, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        ts_ax.set_ylim([-6,6])
        acf_ax = plt.subplot2grid(layout, (1, 0))
        acf_ax.set_ylim([-0.3,1.1])
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        pacf_ax.set_ylim([-0.3,1.1])        
        qq_ax = plt.subplot2grid(layout, (2, 0))
        qq_ax.set_ylim([-6,6])
        pp_ax = plt.subplot2grid(layout, (2, 1))
        pp_ax.set_ylim([-6,6])
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('signal')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return

    
#random sample from US, JP and Kr
CAT = [data_real, data_hum]
for cat in CAT:
    for i in range(2):
        s = np.random.randint(0,cat.shape[0])
        p(s)
        dat = cat.loc[s,'0':]
        tsplot(dat)
        tsplot(dat**2)