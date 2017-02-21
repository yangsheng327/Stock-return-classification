# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 00:45:48 2017
dtw_kmean
@author: Sunrise
"""
import os
import sys

import pandas as pd
import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from sklearn.cross_validation import (cross_val_predict, StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier

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

train_X = data_sample.loc[:,'0':]
train_Y = data_sample.loc[:,'cat']

#fast Dynamic time warping
def dtw_metric(ts1, ts2):
    distance, path = fastdtw(ts1, ts2, dist=euclidean)
    return distance
    
class_names = ['0', '1', '2']
cv = StratifiedKFold(train_Y, 3, shuffle=True)

#SVC classifier
p('running k-Neighbors')
kn_clf = KNeighborsClassifier(n_neighbors=5, metric=dtw_metric)
kn_predictions = cross_val_predict(kn_clf,train_X,train_Y,cv=cv)

def index2cat_num(id):
    if id == 0:
        return us.shape[0]
    elif id ==1:
        return jp.shape[0]
    else:
        return kr.shape[0]
kn_cat_score = np.zeros(3,)
for i in range(3):
    if i == 0: #us
        begin = 0
        end = us.shape[0]
    elif i == 1: #jp
        begin =us.shape[0]
        end = us.shape[0] + jp.shape[0]
    else:
        begin = us.shape[0] + jp.shape[0]
        end = data.shape[0]
    kn_cat_score[i] = kn_cat_score[i] + \
        accuracy_score(train_Y[begin:end],kn_predictions[begin:end])
        
#confusion matrix
#Following function taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', 
                          cmap=plt.cm.Blues):
    cat_num = cm.shape[0]
    corm = np.zeros([cat_num, cat_num])
    for i in range(cat_num):
        for j in range(cat_num):
            corm[i,j] = cm[i,j]/np.sqrt(index2cat_num(i)*index2cat_num(j))
#    p(corm)
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
    

kn_cm = confusion_matrix(train_Y,svc_predictions)
plot_confusion_matrix(kn_cm,class_names,'kn')


#plotting
f, ax = plt.subplots()
w = 0.2
s = np.arange(3)
svc_rect = ax.bar(s+w*1,kn_cat_score,w,color='b')