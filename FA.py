# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 23:39:26 2017
#Fourier analysis
@author: Sunrise
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
p = print

#data input: cat: us-0, jp-1, kr-2
us = pd.read_csv("us.csv")
jp = pd.read_csv("jp.csv")
kr = pd.read_csv("kr.csv")
data_all = pd.concat((us, jp, kr))
data_all.index = range(data_all.shape[0])# data[sample, feature/days]

                       
data = data_all.sample(frac=0.01).reset_index(drop=True)#part of data for testing
# shift [0, 1/T] to [-1/2/T,1/2/T]
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
        filtered_xf = filter_normal(xf,tf, 0, 10)
        xif, tif = ifft( filtered_xf,T)
        axarr[0].plot(x)
        axarr[0].plot(np.real(xif),'*')
        axarr[0].set_title('signal'+title)
        axarr[0].set_xlim([0,252])
        axarr[0].grid()
        axarr[1].plot(abs(xf[tf>0]))
        axarr[1].set_title('spectrum')
        axarr[1].set_xlim([0,125])
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
        axarr[0].set_title('signal'+title)
        axarr[0].set_xlim([0,255])
        axarr[0].grid()
        axarr[1].set_ylim([0,50])
        axarr[2].set_ylim([0,200])
          
    
def fa(n_fa=10):
    
    p('processing IID normal')
    #IID normal(0,1)
    X_iid_norm = np.random.normal(0,1,[n_fa, 252])
#    plot_filter(X_iid_norm, 'IID normal')
    plot_moment_fa(X_iid_norm,'IID normal')

    #US
    p('processing US')
    S = np.random.randint(0,us.shape[0],n_fa)
    X_us = us.loc[S,'0':]
#    plot_filter(np.array(X_us), 'US')
    plot_moment_fa(np.array(X_us),'US')

    #JP
    p('processing JP')
    S = np.random.randint(0,jp.shape[0],n_fa)
    X_jp = us.loc[S,'0':]
#    plot_filter(np.array(X_jp), 'JP')    
    plot_moment_fa(np.array(X_jp),'JP')    
    #KR
    p('processing KR')
    S = np.random.randint(0,kr.shape[0],n_fa)
    X_kr = us.loc[S,'0':]
#    plot_filter(np.array(X_kr), 'KR')
    plot_moment_fa(np.array(X_kr),'KR')    
    
fa(3)
plt.show()