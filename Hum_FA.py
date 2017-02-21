# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:15:16 2017
#Fourier analysis
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
real_n = 10#min(us.shape[0],jp.shape[0],kr.shape[0])
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
data_hum = pd.DataFrame(np.concatenate([np.ones([30,1]),data_hum],axis=1))
data_hum.columns =  data_real.columns

#Fourier Analysis
def fft(x, T):
    x = x
    N = len(x)
    xf = np.fft.fft(x)
#    tf = np.linspace(0,1./T, N)
    tf = np.linspace(-1./2./T,1./2./T, N)
    return np.append(xf[N/2:],xf[0:N/2]), tf
                     
def ifft(xf, T):
    N = len(xf)
    xf = np.append(xf[N/2:N],xf[:N/2])
    x = np.fft.ifft(xf)
    t = np.linspace(0.,N*T,N)
    return x, t

def filter_normal(xf, tf, mu, sigma):
    fil = 1/(sigma * np.sqrt(2 * np.pi))* \
        np.exp( - (np.abs(tf) - mu)**2 / (2 * sigma**2))
    return xf*fil

    
def plot_filter(X, title):
    N = X.shape[0]
    f, axarr = plt.subplots(2)
    for n in range(N):
        x = X[n,:]
        xf, tf = fft(x, 1)
        filtered_xf = filter_normal(xf,tf, 0, 5)
        xif, tif = ifft( filtered_xf,T)
        axarr[0].plot(x)
        axarr[0].plot(np.real(xif),'*')
        axarr[0].set_title(title + ' signal')
        axarr[0].set_xlim([0,252])
        axarr[0].set_ylim([-6, 6])
        axarr[0].grid()
        axarr[1].plot(abs(xf[tf>0]))
        axarr[1].set_title('spectrum')
        axarr[1].set_xlim([0,125])
        axarr[1].set_ylim([0,100])
        axarr[1].grid()
    
    
def plot_moment_fa(X, title):
    N = X.shape[0]
    M = [1,2]
    f, axarr = plt.subplots(len(M)+1, figsize=[8,20])
    for n in range(N):
        x = X[n,:]
        xf = np.zeros([len(M), 252])
        axarr[0].plot(x) 
        axarr[0].set_ylim([-10,10])
        for m in range(len(M)):
            xf[m], tf = fft(x**M[m], 1)
            axarr[m+1].plot(abs(xf[m][tf>0]), 
                label='mean =' + str(int(100*np.mean(abs(xf[m][tf>0])))/100.))
            axarr[m+1].set_title('spectrum for x^' +str(M[m]))
            axarr[m+1].set_xlim([0,125])
            axarr[m+1].grid()
            axarr[m+1].legend(loc='right')
        axarr[0].set_title(title + ' signal')
        axarr[0].set_xlim([0,255])
        axarr[0].grid()
        axarr[1].set_ylim([0,50])
        axarr[2].set_ylim([0,200])
          
    
def fa(n_fa=10):
    
    p('processing artificial signal')
    S = np.random.randint(0,data_hum.shape[0],n_fa)
    plot_filter(np.array(data_hum.loc[S,'0':]),'artificial')
    plot_moment_fa(np.array(data_hum.loc[S,'0':]),'artificial')

    #US
    p('processing real signal')
    S = np.random.randint(0,data_real.shape[0],n_fa)
    plot_filter(np.array(data_real.loc[S,'0':]), 'real')
    plot_moment_fa(np.array(data_real.loc[S,'0':]), 'real')
 
    
fa(2)
plt.show()