# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 00:35:26 2018

@author: joao-
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import pandas as pd
from dummy_var import preprocessData as ppd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

TRAIN_PATH = 'C:\\Users\\joao-\\OneDrive\\Documentos\\Ciencia de Dados\\Project\\CD\\datasets\\under_sampling_aps_training.csv'
TEST_PATH = 'C:\\Users\\joao-\\OneDrive\\Documentos\\Ciencia de Dados\\Project\\CD\\datasets\\aps_test_average.csv'

"""Train Dataset Import and dummyfication"""
df_train = pd.read_csv(TRAIN_PATH, delimiter=',')
df_train= ppd(df_train)

"""Test Dataset Import and dummyfication"""
df_test = pd.read_csv(TEST_PATH, delimiter=',')
df_test = ppd(df_test)

"""Train Dataset Normalization"""
df_train['ab_000']=df_train['ab_000'].astype(float)
X_train =  df_train.iloc[:, 2:(len(df_train.columns))].values
X_train = normalize(X_train)
"""Making the class attribution of the Train dataset"""
y_train = df_train.iloc[:, 0:2].values

"""Test Dataset Normalization"""
df_test['ab_000']=df_test['ab_000'].astype(float)
X_test =df_test.iloc[:, 2:(len(df_test.columns))].values
X_test = normalize(X_test)
"""Making the class attribution of the Test dataset"""
y_test = df_test.iloc[:, 0:2].values

def randForest(crit, n, leafs, splits,feats, depth):
    test_text = ""
    train_text = ""
    d_train=open(crit+"_optimal_train.txt",'w')
    d_test=open(crit+"_optimal_test.txt",'w')
    clf = RandomForestClassifier(criterion=crit, n_estimators=n, min_samples_leaf=leafs, min_samples_split=splits, max_features=feats, max_depth=depth)
        
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    acc = accuracy_score(y_test, predict)*100
    test_text += ("Accuracy test for "+crit +" (with optimal values): " + str(acc)+'\n')
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), predict.argmax(axis=1))
    test_text += (str(conf_matrix)+'\n')
    test_text += "Cost function value of " + crit + "(with optimal values): " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"

    
    pred1= clf.predict(X_train)
    acc = accuracy_score(y_train, pred1)*100
    train_text += ("Accuracy test for "+crit +" (with optimal values): " + str(acc)+'\n')
    conf_matrix = confusion_matrix(y_train.argmax(axis=1), pred1.argmax(axis=1))
    train_text += (str(conf_matrix)+'\n')
    train_text += ("Cost function value of " + crit + "(with optimal values): " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n")
        

    d_test.write(test_text)
    d_train.write(train_text)
    d_train.close()
    d_test.close()
    print( 'Ended optimal forest of ' + crit)
    
randForest('entropy', 50,0.05,0.2,14,4 )
randForest('gini', 60,0.05,0.4,1,3 )