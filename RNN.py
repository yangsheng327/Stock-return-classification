# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:58:28 2017
#recurrent neural network: 1. directly fit the time series
                           2. chop the time series and get the varience
@author: Sunrise
"""
import os
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
import pyrenn as prn
from sklearn.metrics import confusion_matrix

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
sample_n = 100#min(us.shape[0],jp.shape[0],kr.shape[0])
data_sample = pd.concat((us.sample(sample_n), jp.sample(sample_n), 
                         kr.sample(sample_n)))
data_sample.index = range(data_sample.shape[0])

#group the time series
def grouping(TS, gr=50):
    N, T = TS.shape
    TS_RNN = np.zeros([N, gr])
    gr_size = int(T/gr)
    for i in range(gr):
#        p(np.mean(TS[:,i*gr_size:(i+1)*gr_size]))#mean of group close to 0
        TS_RNN[:,i] = np.var(TS[:,i*gr_size:(i+1)*gr_size], axis=1)
    return TS_RNN

gr = 50    
data_X = grouping(np.array(np.array(data_sample.loc[:,'0':])), gr)
data_Y = data_sample.loc[:,'cat']
data_Y_cat = \
    np.array(pd.get_dummies(data_sample.loc[:,'cat'].astype('category')))
    
#StratifiedKFold
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(data_X, data_Y)

#creat ANN: 252 inputs, 3 output (prob for each category)
net = prn.CreateNN([gr,20,20,20,20,3],dIn=[0],dIntern=[1],dOut=[])

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

class_names = ['US', 'JP', 'KR']  
_cm = confusion_matrix(test_Y_cat,prod_Y_cat)
cat_score = plot_confusion_matrix(_cm,class_names,'ANN')

#plotting
f, ax = plt.subplots()
w = 0.2
s = np.arange(3)
_rect = ax.bar(s+1.5*w,cat_score,w,color='g')

ax.set_title('Scores by Categories')
ax.set_ylabel('Scores')
ax.set_xticks(s+2*w)
ax.set_xticklabels(class_names)

plt.show()
