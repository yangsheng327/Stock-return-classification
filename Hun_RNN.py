# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:25:16 2017
#artificial time series: 1. IID normal
                         2. GARCH
                         3. HMM
                         rule of generation: randomly sample from real
                         data, then fit with model, finally generate time
                         series with corresponding model
#recurrent neural network: 1. directly fit the time series
                           2. chop the time series and get the varience
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

from sklearn.model_selection import StratifiedKFold
import pyrenn as prn
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
plt.close('all')
p = print
T = 252

#setting
real_n = 100#min(us.shape[0],jp.shape[0],kr.shape[0])
gr = 100

#data input: cat: us-0, jp-1, kr-2
us = pd.read_csv("us.csv")
jp = pd.read_csv("jp.csv")
kr = pd.read_csv("kr.csv")

#data_all has all the data
data_all = pd.concat((us, jp, kr))
data_all.index = range(data_all.shape[0])# data[sample, feature/days]

#data_real from real signal
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
data_hum = norm_gen_TS(3*real_n)
#data_hum = garch_gen_TS(3*real_n)
#data_hum = hmm_gen_TS(3*real_n)
data_hum = pd.DataFrame(np.concatenate([np.ones([3*real_n,1]),data_hum],axis=1))
data_hum.columns =  data_real.columns
data =  pd.concat((data_real, data_hum))
data.index = range(data.shape[0])

#group the time series
def grouping(TS, gr=25):
    N, T = TS.shape
    TS_RNN = np.zeros([N, gr])
    gr_size = int(T/gr)
    for i in range(gr):
#        p(np.mean(TS[:,i*gr_size:(i+1)*gr_size]))#mean of group close to 0
        TS_RNN[:,i] = np.var(TS[:,i*gr_size:(i+1)*gr_size], axis=1)
    return TS_RNN

    
data_X = grouping(np.array(np.array(data.loc[:,'0':])), gr)
data_Y = data.loc[:,'cat']
data_Y_cat = \
    np.array(pd.get_dummies(data.loc[:,'cat'].astype('category')))
    
#StratifiedKFold
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(data_X, data_Y)

#creat ANN: 252 inputs, 3 output (prob for each category)
net = prn.CreateNN([gr,20,20,2],dIn=[0],dIntern=[],dOut=[])

#StratifiedKFold
tr = np.transpose
test_Y_cat = np.array([])
prod_Y_cat = np.array([])
for train_index, test_index in skf.split(data_X, data_Y):   
   train_X, test_X = tr(data_X[train_index,:]), tr(data_X[test_index,:])
   train_Y, test_Y = tr(data_Y_cat[train_index,:]), \
                        tr(data_Y_cat[test_index,:])
   net = prn.train_LM(train_X,train_Y,net,verbose=True,k_max=500,E_stop=1e-5)
   prob_y = prn.NNOut(test_X,net)
   test_Y_cat = np.append(test_Y_cat,np.argmax(test_Y,axis=0))
   prod_Y_cat = np.append(prod_Y_cat,np.argmax(prob_y, axis=0))
   
#confusion matrix, return accuray score
#Following function taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', 
                          cmap=plt.cm.Blues):
    cat = [0,1,2]
    cat_num = cm.shape[0]
    corm = np.zeros([cat_num, cat_num])
    for i in range(cat_num):
        for j in range(cat_num):
            corm[i,j] = \
                cm[i,j]/np.sqrt(sum(data_Y == cat[i])*sum(data_Y == cat[j]))
    plt.subplots()
    plt.imshow(corm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return corm.diagonal()

class_names = ['real', 'artificial']  
_cm = confusion_matrix(test_Y_cat,prod_Y_cat)
cat_score = plot_confusion_matrix(_cm,class_names,'RNN')

#plotting
f, ax = plt.subplots()
w = 0.2
s = np.arange(2)
_rect = ax.bar(s+1.5*w,cat_score,w,color='g')

ax.set_title('Scores by Categories')
ax.set_ylabel('Scores')
ax.set_xticks(s+2*w)
ax.set_xticklabels(class_names)

plt.show()