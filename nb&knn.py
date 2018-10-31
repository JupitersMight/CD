#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:52:17 2018

@author: claudia
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def plot_confusion_matrix(cnf_matrix, classesNames, normalize=False,
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if normalize:
        soma = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / soma
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix, without normalization'

    print(cm)

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classesNames))
    plt.xticks(tick_marks, classesNames, rotation=45)
    plt.yticks(tick_marks, classesNames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :4]
y = iris['class']
labels = pd.unique(y)

trX, tsX, trY, tsY = train_test_split(X, y, train_size=0.7, stratify=y)

knn = KNeighborsClassifier(n_neighbors=3)
gnb = GaussianNB()
model1 = knn.fit(trX, trY)
model2 = gnb.fit(trX, trY)
predY1 = model1.predict(tsX)
predY2 = model2.predict(tsX)

cnf_matrix1 = confusion_matrix(tsY, predY1, labels)
cnf_matrix2 = confusion_matrix(tsY, predY2, labels)

plot_confusion_matrix(cnf_matrix1, labels)
plot_confusion_matrix(cnf_matrix2, labels)

acknn = accuracy_score(tsY, predY1)
acbayes = accuracy_score(tsY, predY2)

print("accuracy knn :", acknn)
print("accuracy naive bayes:", acbayes)









