# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:25:17 2019

@author: alok
"""
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import scikitplot as skplt
import numpy as np
import pandas as pd
from pandas import *
from Bio.Seq import Seq

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
from itertools import *
from pycm import *
from sklearn.ensemble import RandomForestClassifier

dataset=read_csv('C:/Users/alok/rfam_riboswitches.csv', low_memory=False)
dataset['words'] = dataset.apply(lambda x: getKmers(x['Sequences']), axis=1)
dataset = dataset.drop('Sequences', axis=1)
#dataset.to_csv("ribfuzzy.csv")
data_text = list(dataset['words'])
for item in range(len(data_text)):
    data_text[item] = ' '.join(data_text[item])
Y = dataset.iloc[:, 1].values
CV = CountVectorizer(input=data_text,dtype=int,analyzer='word')
x = CV.fit(data_text)
X=CV.transform(data_text)
X=X.toarray()
Z= pd.DataFrame(X, columns=CV.get_feature_names())
X = data.iloc[:,0:].values
Z.to_csv("ribfuzzy1.csv")
#336 features
format(x.get_feature_names())
#format(CV.get_feature_names())
data=pd.read_csv('C:/Users/alok/ribfuzzy2.csv', low_memory=False)
Y = data.iloc[:, 0].values
X = data.iloc[:,4:].values
X = X.astype(np.float32())
train_X, test_X, train_Y, test_Y = train_test_split(X,Y, random_state=0,test_size=0.30)
model= OneVsRestClassifier(svm.SVC(kernel='rbf',C=45,verbose=True,probability=True,gamma='scale'))
val_predict=model.fit(train_X,np.ravel(train_Y)).predict(test_X)
#val_predict= model.predict(test_X)
#model accuracy
model.score(test_X,test_Y)
print(" " , 100 *accuracy_score(test_Y, val_predict, normalize=True))
print(" " , 100 *accuracy_score(train_Y, model.predict(train_X), normalize=True))
  
#accurate no. of outputs
print(" " , accuracy_score(test_Y, val_predict, normalize=False))
cm = ConfusionMatrix(actual_vector=np.ravel(test_Y), predict_vector=np.ravel(val_predict))
cmdf= pd.DataFrame(cm.table)
cmdf.to_csv("Confusion matrix.csv")

print(cm)
cm.PPV
CM= confusion_matrix(test_Y, val_predict)
matthews_corrcoef(test_Y, val_predict)
matthews_corrcoef(test_Y, val_predict)
#crossvalidation acuracy
cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
scores = cross_val_score(model, X, Y, cv=12,scoring='accuracy')
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
print(scores.max()*100)
print(classification_report(test_Y, val_predict))

fig, ax = plot_confusion_matrix(conf_mat=CM, colorbar=True, show_absolute=False, show_normed=True, figsize=(50,50))
fig.savefig('con_max_OvRSVM.png')
see=Seq("ACAACUCAGGUCUGUGGUUGCAAGUCGAUGCCAGUUGCAGGCAAAACGAUCCACGUAAGCAGGGAAACCCCUGUGAGCACGGUGCAGCUUAGAAGUAAGUCCUGCCGCAAAAUGCGAGAGAGGCAGUAGUGGGGAGCACGAAGCUUAGGAGCGAACCCUCCAGCAGGCGAGUGUGGGGGCGAAAACCAGGUCAGCUGAGUUGU")
#see=see.transcribe()
see=getKmers(see.tostring())
see=' '.join(see)
seq=[]
seq.append(see)
see=x.transform(seq)
see=see.toarray()
model.fit(X,Y)
model.predict(see)



