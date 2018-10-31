# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn import make_classification
import pandas as pd

dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, :4]
y = dataset['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y)
clf = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0)
clf.fit(X, y)

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=2, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#            oob_score=False, random_state=0, verbose=0, warm_start=False)

#X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

predict = clf.predict(X_test)
print (accuracy_score(y_test, predict))