# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#from sklearn import make_classification
import pandas as pd
# Change to fit the index and name of the class attribute
INDEX_OF_CLASS_ATTRIBUTE = 4
NAME_OF_CLASS_ATTRIBUTE = 'class'

# If the dataset is not divided into training and testing datasets

NAME_OF_DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\iris.csv'
df = pd.read_csv(NAME_OF_DATASET_FILE, delimiter=',', na_values=['na'])

X = df.iloc[:, df.columns != NAME_OF_CLASS_ATTRIBUTE].values
y = df.iloc[:, INDEX_OF_CLASS_ATTRIBUTE]

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, stratify=Y)

# If dataset is already divided into training and testing datasets

# NAME_OF_TRAINING_DATASET = ''
# NAME_OF_TESTING_DATASET= ''
# df_training = pd.read_csv(NAME_OF_TRAINING_DATASET, delimiter=',', na_values=['na'])
# df_testing = pd.read_csv(NAME_OF_TESTING_DATASET, delimiter=',', na_values=['na'])

# X_train = df_training.iloc[:, df_training.columns != NAME_OF_CLASS_ATTRIBUTE].values
# y_train = df_training.iloc[:, INDEX_OF_CLASS_ATTRIBUTE].values
# X_test = df_testing.iloc[:, df_training.columns != NAME_OF_CLASS_ATTRIBUTE].values
# y_test = df_testing.iloc[:, INDEX_OF_CLASS_ATTRIBUTE].values


clf = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0)
clf.fit(X, y)

predict = clf.predict(X_test)
print (accuracy_score(y_test, predict))
conf_matrix = confusion_matrix(y_test, predict, pd.unique(y))




########  Outra implementação das random forests (a primeira funcionou, por isso pus este em comentario)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=2, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#            oob_score=False, random_state=0, verbose=0, warm_start=False)

#X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)