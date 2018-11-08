# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import forestci as fci
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn import make_classification
import pandas as pd
from dummy_var import preprocessData as ppd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold as skf
#from sklearn.feature_extraction

"""<<<<<<<<<<<<<<<<<<<<<<<<<<Datasets' Filepaths>>>>>>>>>>>>>>>>>>>>>>>>>"""
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
#print(df.columns)
#X = df.iloc[:, 0:2].values
##print (X)
##print(df.columns)
#y = df.iloc[:, 2:(len(df.columns))]

K_Fold=skf(n_splits=10)

"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Callables >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""

"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<< minDifference >>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""Returns the index whose accuracy difference between """
def minDifference(lists, test_accuracy, train_accuracy):
    n=0
    curIndex=0
    curMin=10
    for accDiff in lists:
        if accDiff<curMin and test_accuracy[n] > 0.8 and train_accuracy[n] > 0.8:
            curIndex=n
            curMin=accDiff
        n+=1
    return curIndex



"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<< Vary Estimators >>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""calulates the accuracies for each value of estimators and depths"""
def forestEstimatorDepthAccuracies(crit, XTrain, YTrain):
    depths = range(1,11)
    estimators = [10, 20, 30]#, 40, 50, 60, 70, 80, 90, 100]
    #for each value of estimators
    for estimator in estimators:
        test_accuracy=[]
        train_accuracy = []
        #for each value of depth
        for depth in depths:
        
            clf = RandomForestClassifier( n_estimators = estimator, criterion=crit, max_depth=depth, max_features=12)
            
            clf.fit(XTrain, YTrain)
            predict = clf.predict(X_test)
            acc = accuracy_score(y_test, predict)*100
            print("Accuracy test: " + str(acc))
            test_accuracy.append(acc)
            conf_matrix = confusion_matrix(y_test.argmax(axis=1), predict.argmax(axis=1))
            print(conf_matrix)
            
            
            pred1= clf.predict(XTrain)
            acc1 = accuracy_score(YTrain, pred1)*100
            print("Accuracy train: " + str(acc1))
            train_accuracy.append(acc1)
            conf_matrix = confusion_matrix(YTrain.argmax(axis=1), pred1.argmax(axis=1))
            print(conf_matrix)
        
#            confidence = fci.random_forest_error(clf, X_train, X_test )
        #confidence=0
#            print ('confidence = ' + str(confidence))
        
        
        Labels = range(1,12)
        
        plt.plot( minSplit, test_accuracy, color = 'r', label = 'Test')
        plt.plot( minSplit, train_accuracy, color='b', label = 'Train)
        plt.xticks(depths, Labels)
        plt.ylabel('Accuracy')
        plt.xlabel('Tree Depth')
        plt.savefig(('Estimators&Depth\\graph'+str(estimator)+'est.png'), dpi=100)
        plt.show()

"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<< Vary Features >>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
def maxFeaturesForest(crit, XTrain, YTrain):
    test_accuracy=[]
    train_accuracy = []
    differenceBetweenAccuracies=[]
    max_feat=range(1,171,1)
    #for each value of depth
    for feat in max_feat:
        
        d_train=open("Matrix\\"+str(feat)+"features_"+str(crit)+"_train.txt",'w')
        d_test=open("Matrix\\"+str(feat)+"features_"+str(crit)+"_test.txt",'w')

        clf = RandomForestClassifier(criterion=crit, max_features=feat)
            
        clf.fit(XTrain, YTrain)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)*100
        print("Accuracy test: " + str(acc))
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test.argmax(axis=1), predict.argmax(axis=1))
#        print(conf_matrix)
        d_test.write(str(conf_matrix))

        
        pred1= clf.predict(XTrain)
        acc1 = accuracy_score(YTrain, pred1)*100
        print("Accuracy train: " + str(acc1))
        train_accuracy.append(acc1)
        conf_matrix = confusion_matrix(YTrain.argmax(axis=1), pred1.argmax(axis=1))
#        print(conf_matrix)
        d_train.write(str(conf_matrix))


        differenceBetweenAccuracies.append((acc1-acc))
        
    
#            confidence = fci.random_forest_error(clf, X_train, X_test )
    #confidence=0
#            print ('confidence = ' + str(confidence))
    
    
#    Labels = depths 
    
    plt.plot( minSplit, test_accuracy, color = 'r', label = 'Test')
    plt.plot( minSplit, train_accuracy, color='b', label = 'Train)
#    plt.xticks(depths, Labels)
    plt.ylabel('Accuracy')
    plt.xlabel('N of Features')
    plt.savefig(('Features\\graph'+str(feat)+'feat'+str(crit)), dpi=100)
    plt.show()
    print(differenceBetweenAccuracies)
    n = (minDifference(differenceBetweenAccuracies, test_accuracy, train_accuracy))
    print('index: '+str(n)+'\nvalue: '+str(differenceBetweenAccuracies[n])+'\ntrain_acc: ' + str(train_accuracy[n]) + '\ntest_acc' + str(test_accuracy[n]))

#    print(str(n)+diffe)
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<< Vary Sample Splits >>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
def minSampleSplitForest(crit, XTrain, YTrain):
    test_accuracy=[]
    train_accuracy = []
    differenceBetweenAccuracies=[]
    minSplit=[0.05,0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4,0.45, 0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8,0.85,0.9]
    #for each value of depth
    for split in minSplit:
        
        d_train=open("Matrix\\"+str(split)+"splits_"+str(crit)+"_train.txt",'w')
        d_test=open("Matrix\\"+str(split)+"splits_"+str(crit)+"_test.txt",'w')
        clf = RandomForestClassifier(criterion=crit, min_samples_split=split)
        
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)
#        print("Accuracy test: ")# + str(acc))
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test.argmax(axis=1), predict.argmax(axis=1))
        d_test.write(str(conf_matrix))
        
        
        pred1= clf.predict(X_train)
        acc1 = accuracy_score(y_train, pred1)
#        print("Accuracy train: " )#+ str(acc1))
        train_accuracy.append(acc1)
        conf_matrix = confusion_matrix(y_train.argmax(axis=1), pred1.argmax(axis=1))
        d_train.write(str(conf_matrix))
        
        
        differenceBetweenAccuracies.append(abs(acc1-acc))
        
    
#            confidence = fci.random_forest_error(clf, X_train, X_test )
    #confidence=0
#            print ('confidence = ' + str(confidence))
    
    
    Labels =  [0.05,0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4,0.45, 0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8,0.85,0.9]
    #[0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    plt.plot( minSplit, test_accuracy, color = 'r', label = 'Test')
    plt.plot( minSplit, train_accuracy, color='b', label = 'Train)
    plt.xticks(minSplit, Labels)
    plt.ylabel('Accuracy')
    plt.xlabel('Sample Splits')
    plt.savefig(('SampleSplits\\graph'+str(split)+'split'+str(crit)+'.png'), dpi=100)
    plt.show()
#    print(differenceBetweenAccuracies)
    n = (minDifference(differenceBetweenAccuracies, test_accuracy, train_accuracy))
    print('index: '+str(n)+'\nvalue: '+str(differenceBetweenAccuracies[n])+'\ntrain_acc: ' + str(train_accuracy[n]) + '\ntest_acc' + str(test_accuracy[n]))

"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<< Calls to functions >>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
def minSampleLeaf(crit, XTrain, YTrain):
    test_accuracy=[]
    train_accuracy = []
    differenceBetweenAccuracies=[]
    minLeaf=[0.1, 0.15,0.2,0.25,0.3,0.35, 0.4, 0.45, 0.5]#range(1,1000,50)
    #for each value of depth
    for leafs in minLeaf:
        
        d_train=open("Matrix\\"+str(leafs)+"leafs"+str(crit)+"_train.txt",'w')
        d_test=open("Matrix\\"+str(leafs)+"leafs"+str(crit)+"_test.txt",'w')
        clf = RandomForestClassifier(criterion=crit, min_samples_leaf=leafs)
        
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)
#        print("Accuracy test: ")# + str(acc))
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test.argmax(axis=1), predict.argmax(axis=1))
        d_test.write(str(conf_matrix))
        
        
        pred1= clf.predict(X_train)
        acc1 = accuracy_score(y_train, pred1)
#        print("Accuracy train: " )#+ str(acc1))
        train_accuracy.append(acc1)
        conf_matrix = confusion_matrix(y_train.argmax(axis=1), pred1.argmax(axis=1))
        d_train.write(str(conf_matrix))
        differenceBetweenAccuracies.append(abs(acc1-acc))
        
    
#        confidence = fci.random_forest_error(clf, X_train, X_test )
#        confidence=0
#        print ('confidence = ' + str(confidence))
    
    plt.plot( minSplit, test_accuracy, color = 'r', label = 'Test')
    plt.plot( minSplit, train_accuracy, color='b', label = 'Train)
#    plt.xticks(depths, Labels)
    plt.ylabel('Accuracy')
    plt.xlabel('Min Sample Leafs')
    plt.savefig('SampleLeaf\\graph'+str(leafs)+'leafs'+str(crit)+'.png', dpi=100) #(
    plt.show()
#    print(differenceBetweenAccuracies)
    n = (minDifference(differenceBetweenAccuracies, test_accuracy,train_accuracy))
    print('index: '+str(minLeaf[n])+'\nvalue: '+str(differenceBetweenAccuracies[n])+'\ntrain_acc: ' + str(train_accuracy[n]) + '\ntest_acc' + str(test_accuracy[n]))
    




"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< RUN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""

def runAll():
    for train_index, test_index in K_Fold.split(X_train, y_train.argmax(axis=1)):
        forestEstimatorDepthAccuracies('entropy',X_train[train_index], y_train[train_index])
        forestEstimatorDepthAccuracies('gini', X_train[train_index], y_train[train_index])
        minSampleLeaf('entropy', X_train[train_index], y_train[train_index])
        minSampleLeaf('gini',X_train[train_index], y_train[train_index])
        maxFeaturesForest('entropy', X_train[train_index], y_train[train_index])
        maxFeaturesForest('gini', X_train[train_index], y_train[train_index])
        minSampleSplitForest('entropy', X_train[train_index], y_train[train_index])
        minSampleSplitForest('gini', X_train[train_index], y_train[train_index])
        
runAll()



"""<<<<<<<<<<<<<<<<<<<<<< IGNORE FROM HERE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
########  Outra implementação das random forests (a primeira funcionou, por isso pus este em comentario)

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

#            max_depth=2, max_features='auto', max_leaf_nodes=None,

#            min_impurity_decrease=0.0, min_impurity_split=None,

#            min_samples_leaf=1, min_samples_split=2,

#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,

#            oob_score=False, random_state=0, verbose=0, warm_start=False)



#X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)