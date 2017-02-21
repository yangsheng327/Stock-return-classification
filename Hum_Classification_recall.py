# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 23:12:26 2017
#classification: 1. k-Neighbors classifier
                 2. SVC classifier
                 3. Logistic classifier
                 4. Random Forest classifier
                 5. Multi-layer Perceptron
#Crossvalidation: StratifiedKFold
#Run after Feature.py need train_X, train_Y
#Return recall
@author: Sunrise
"""
from sklearn.cross_validation import (cross_val_predict, StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.close('all')

class_names = ['real','artificial']
cv = StratifiedKFold(train_Y, 3, shuffle=True)

#k-Neighbors classifier
p('running k-Neighbors')
kn_clf = KNeighborsClassifier(n_neighbors=20)
kn_predictions = cross_val_predict(kn_clf,train_X,train_Y,cv=cv)

#SVC classifier
p('running SVC')
svc_clf = SVC(kernel='linear')
#svc_clf = SVC()
svc_predictions = cross_val_predict(svc_clf,train_X,train_Y,cv=cv)

#Logistic classifier
p('running Logistic classifier')
lg_clf = LogisticRegression()
lg_predictions = cross_val_predict(lg_clf,train_X,train_Y,cv=cv)

#Random Forest classifier
p('running Random Forest')
rf_clf = RandomForestClassifier(n_estimators=500)
rf_predictions = cross_val_predict(rf_clf,train_X,train_Y,cv=cv)

#Multi-layer Perceptron
p('running Multi-layer Perceptron')
mlp_clf = MLPClassifier(hidden_layer_sizes=(20, 20))
mlp_predictions = cross_val_predict(mlp_clf,train_X,train_Y,cv=cv)

      
      
#confusion matrix, return accuray score
#Following function taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, predict, target_names, title='Confusion matrix', 
                          cmap=plt.cm.Blues):
    cat = [b'0',b'1',b'2']
    cat_num = cm.shape[0]
    corm = np.zeros([cat_num, cat_num])
    for i in range(cat_num):
        for j in range(cat_num):
            corm[i,j] = \
                cm[i,j]/sum(train_Y == cat[j])            
    plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return corm.diagonal()
    
kn_cm = confusion_matrix(train_Y,kn_predictions)
svc_cm = confusion_matrix(train_Y,svc_predictions)
lg_cm = confusion_matrix(train_Y,lg_predictions)
rf_cm = confusion_matrix(train_Y,rf_predictions)
mlp_cm = confusion_matrix(train_Y,mlp_predictions)

kn_cat_score = plot_confusion_matrix(kn_cm, 
                                     kn_predictions,class_names,'k-neigh')
svc_cat_score = plot_confusion_matrix(svc_cm,
                                      svc_predictions, class_names,'svc')
lg_cat_score = plot_confusion_matrix(lg_cm,
                                     lg_predictions, class_names,'lg')
rf_cat_score = plot_confusion_matrix(rf_cm,
                                     rf_predictions, class_names,'rf')
mlp_cat_score = plot_confusion_matrix(mlp_cm,
                                      mlp_predictions, class_names,'mlp')

#plotting
f, ax = plt.subplots()
w = 0.15
s = np.arange(2)
kn_rect = ax.bar(s+w*0,kn_cat_score,w,color='r')
svc_rect = ax.bar(s+w*1,svc_cat_score,w,color='b')
lg_rect = ax.bar(s+w*2,lg_cat_score,w,color='g')
rf_rect = ax.bar(s+w*3,rf_cat_score,w,color='y')
mlp_rect = ax.bar(s+w*4,mlp_cat_score,w,color='m')


ax.set_title('Recall scores by categories')
ax.set_ylabel('Scores')
ax.set_xticks(s+2.5*w)
ax.set_xticklabels(class_names)
ax.legend((kn_rect[0],svc_rect[0],lg_rect[0],rf_rect[0], mlp_rect[0]), 
           ('k-neigh','svc','lg','rforest','mlp'), loc=3)

plt.show()
p(np.mean(kn_cat_score))
p(np.mean(svc_cat_score))
p(np.mean(lg_cat_score))
p(np.mean(rf_cat_score))
p(np.mean(mlp_cat_score))