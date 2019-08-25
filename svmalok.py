# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:03:17 2019

@author: alok
"""
import scikitplot as skplt
import numpy as np
import pandas as pd
#from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
#from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc,roc_auc_score,f1_score,matthews_corrcoef,classification_report
from sklearn.model_selection import ShuffleSplit
from mlxtend.plotting import plot_confusion_matrix
from scipy import interp
from itertools import cycle
from pycm import *
dataset=pd.read_csv('C:/Users/alok/ribfuzzy.csv', low_memory=False)

inc= dataset.columns
incol= inc[3:]
otcol =inc[:1]
X=dataset[incol]
Y=dataset[otcol]
#Y = label_binarize(Y, classes=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
train_X, test_X, train_Y, test_Y = train_test_split(X,Y, random_state=1,test_size=0.25)
#crossvalidation_split
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
model=OneVsRestClassifier(svm.SVC(kernel='rbf',C=45,verbose=True,probability=True,gamma='scale'))
#crossvalidation
scores = cross_val_score(model, X, Y, cv=cv)
model.fit(train_X,np.ravel(train_Y))
val_predict= model.predict(test_X)
#scores for roc_auc
Y_scores= model.fit(train_X,train_Y).predict_proba(test_X)
model.score(train_X,train_Y)
#model accuracy
print(" " , 100 *accuracy_score(test_Y, val_predict, normalize=True))
#accurate no. of outputs
print(" " , accuracy_score(test_Y, val_predict, normalize=False))
#crossvalidation acuracy
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores.max()*100)
preds=[ np.argmax(t) for t in val_predict ]
test=[ np.argmax(t) for t in test_Y ]
#confusionmatrix
cm = ConfusionMatrix(actual_vector=np.ravel(test_Y), predict_vector=np.ravel(val_predict))
cmdf= pd.DataFrame(cm.table)
cmdf.to_csv("cmdf.csv")
cm.AUC
cm.FPR
cm.TPR
fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_absolute=True, show_normed=False, figsize=(50,50))
fig.savefig('con_max_svmsimple.png')
#roc_curve
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr=cm.FPR
tpr=cm.TPR
roc_auc=cm.AUC
# Compute micro-average ROC curve and ROC area

# Compute macro-average ROC curve and ROC area
pl_roc=skplt.metrics.plot_roc(test_Y,Y_scores)
pl_prec=skplt.metrics.plot_precision_recall_curve(test_Y, Y_scores, figsize=(100,100))
pl_cm= skplt.metrics.plot_confusion_matrix(test_Y, val_predict, normalize=True, figsize=(10,10))
#roc_auc_scores
roc_auc_score(test_Y, Y_scores)
#matthews_corrcoef
matthews_corrcoef(test, preds)
#classification report
print(classification_report(test_Y, val_predict))