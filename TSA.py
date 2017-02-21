# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:58:35 2017
##Time series analysis
##Given time series, plot: 1. signal, 2.autocorrelation, 
                           3. particial autocorrelation, 
                           4. Quantile vs. Quantile
@author: Sunrise
"""
import os
import sys

import pandas as pd
import numpy as np

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import matplotlib.pyplot as plt
plt.close('all')
p = print

#data input: cat: us-0, jp-1, kr-2
us = pd.read_csv("us.csv")
jp = pd.read_csv("jp.csv")
kr = pd.read_csv("kr.csv")

#plotting
def tsplot(y, lags=30, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        acf_ax.set_ylim([-0.3,1.1])
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        pacf_ax.set_ylim([-0.3,1.1])        
        qq_ax = plt.subplot2grid(layout, (2, 0))
        qq_ax.set_ylim([-6,6])
        pp_ax = plt.subplot2grid(layout, (2, 1))
        pp_ax.set_ylim([-6,6])
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return

    
#random IID normal(0,1)
rants = np.random.normal(size=2520)
tsplot(rants)
tsplot(rants**2)
#random sample from US, JP and Kr
CAT = [us, jp, kr]
for cat in CAT:
    for i in range(1):
        s =np.random.randint(0,cat.shape[0])
        p(s)
        dat = cat.loc[s,'0':]
        tsplot(dat)
        tsplot(dat**2)

