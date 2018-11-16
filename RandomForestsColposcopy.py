# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:26:22 2018

@author: joao-
"""

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
from matplotlib.legend_handler import HandlerLine2D
#from sklearn.feature_extraction
import numpy as np

"""<<<<<<<<<<<<<<<<<<<<<<<<<<Datasets' Filepaths>>>>>>>>>>>>>>>>>>>>>>>>>"""
DATA_PATH = 'C:\\Users\\joao-\\OneDrive\\Documentos\\Ciencia de Dados\\Project\\CD\\datasets\\col\\smote_over_sampling_col.csv'


"""Train Dataset Import and dummyfication"""
df = pd.read_csv(DATA_PATH, delimiter=',')
df = ppd(df)

X = normalize(df.iloc[:,1:len(df.columns)])
y = df.iloc[:,0]

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


def draw_graphs(criterion, test_range, varying_parameter, test_accuracy, train_accuracy):
    Labels = test_range
#    print (train_accuracy)
    
    tests, = plt.plot( test_range, test_accuracy, color = 'r')
    trains, = plt.plot( test_range, train_accuracy, color='b')
    if(varying_parameter!='max_features'):
        plt.plot( test_range, test_accuracy, 'ro', color = 'r')
        plt.plot( test_range, train_accuracy, 'ro', color='b')
    plt.legend([trains, tests], ['Train','Test'])
    if(varying_parameter!='max_features'):
        plt.xticks(test_range, Labels)
    plt.ylabel('Accuracy')
    plt.xlabel(varying_parameter)
    plt.yticks([50,55,60,65,70,75,80,85,90,95,100,105])
    plt.savefig(('Colposcopy\\Graphs\\'+varying_parameter+'\\graph_'+varying_parameter+'_'+str(criterion)+'.png'), dpi=100)
    plt.show()
#    +str(lentest_range)+
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<< Vary Estimators >>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""calulates the accuracies for each value of estimators and depths"""
def num_estimators(crit, XTrain, YTrain,X_test, y_test, num):
    estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    test_accuracy=[]
    train_accuracy = [] 
    costs_tr=[]
    costs_ts=[]
    
    c_test = open("Colposcopy\\Matrix\\num_estimators\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_test.txt",'w')
    c_train = open("Colposcopy\\Matrix\\num_estimators\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_train.txt",'w')
    
    d_train=open("Colposcopy\\Matrix\\num_estimators\\"+str(crit)+"estimators"+"_run_num_"+str(num)+"_train.txt",'w')
    d_test=open("Colposcopy\\Matrix\\num_estimators\\"+str(crit)+"estimators"+"_run_num_"+str(num)+"_test.txt",'w')
    train_text = ""
    test_text = ""
    #for each value of estimators
    for estimator in estimators:

#            d_train=open("Matrix\\depth\\"+str(depth)+"depth"+str(crit)+"estima"+ str(estimator)+"_train.txt",'w')
#            d_test=open("Matrix\\depth\\"+str(depth)+"depth"+str(crit)+"estima"+ str(estimator) +"_test.txt",'w')
        clf = RandomForestClassifier( n_estimators = estimator, criterion=crit, max_features=12)
        
        clf.fit(XTrain, YTrain)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)*100
        test_text += ("Accuracy test for "+str(estimator) + " estimators: " + str(acc) +'\n')
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test, predict)
        test_text += (str(conf_matrix)+'\n')
        test_text += "Cost function value of " + str(estimator) + " estimators is: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+" False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_ts.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

        pred1= clf.predict(XTrain)
        acc1 = accuracy_score(YTrain, pred1)*100
        train_text += ("Accuracy Train for "+str(estimator) + " estimators: " + str(acc) +'\n')
        train_accuracy.append(acc1)
        conf_matrix = confusion_matrix(YTrain, pred1)
        train_text += (str(conf_matrix)+'\n')
        train_text += "Cost function value of " + str(estimator) + " estimators is: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+" False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_tr.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

#            confidence = fci.random_forest_error(clf, X, X_test )
#            confidence=0
#            print ('confidence = ' + str(confidence))

        
#    Labels = estimators
#    
#    tests, = plt.plot( estimators, test_accuracy, color = 'r', label = 'Test')
#    trains, = plt.plot( estimators, train_accuracy, color='b', label = 'Train')
#    plt.plot( estimators, test_accuracy, 'ro', color = 'r')
#    plt.plot( estimators, train_accuracy, 'ro', color='b')
#    plt.legend([trains, tests], ['Train','Test'])
#    plt.xticks(estimators, Labels)
#    plt.ylabel('Accuracy')
#    plt.xlabel('Number of Estimators')
#    plt.yticks([75,80,85,90,95,100, 105])
#    plt.savefig(('Colposcopy\\Graphs\\num_estimators\\graph'+'crit'+ str(crit)+str(estimator)+'estimators'+"_run_num_"+str(num)+".png"), dpi=100)
#    plt.show()

    c_test.write(str(costs_ts))
    c_test.close()
    c_train.write(str(costs_tr))
    c_train.close()
    d_test.write(test_text)
    d_train.write(train_text)
    d_train.close()
    d_test.close()
    print( 'Ended estimators_tree of ' + crit)

    return train_accuracy, test_accuracy


"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<< Vary Features >>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
def max_features(crit, XTrain, YTrain,X_test, y_test, num):
    test_accuracy=[]
    train_accuracy = []
    differenceBetweenAccuracies=[]
    max_feat=range(1,63,1)
    train_text = ""
    test_text = ""
    
    costs_tr=[]
    costs_ts=[]
    
    c_test = open("Colposcopy\\Matrix\\max_features\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_test.txt",'w')
    c_train = open("Colposcopy\\Matrix\\max_features\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_train.txt",'w')
   
    
    d_train=open("Colposcopy\\Matrix\\max_features\\"+str(crit)+"_run_num_"+str(num)+"_train.txt",'w')
    d_test=open("Colposcopy\\Matrix\\max_features\\"+str(crit)+"_run_num_"+str(num)+"_test.txt",'w')
    for feat in max_feat:
        
#        d_train=open("Matrix\\feature\\"+str(feat)+"features_"+str(crit)+"_train.txt",'w')
#        d_test=open("Matrix\\feature\\"+str(feat)+"features_"+str(crit)+"_test.txt",'w')

        clf = RandomForestClassifier(criterion=crit, max_features=feat)
            
        clf.fit(XTrain, YTrain)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)*100
        test_text += ("Accuracy test for " + str(feat) + " features: " + str(acc)+'\n')
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test, predict)
        test_text += (str(conf_matrix)+'\n')
        test_text += "Cost function value of " + str(feat) + " features: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_ts.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

        pred1= clf.predict(XTrain)
        acc1 = accuracy_score(YTrain, pred1)*100
        train_text += ("Accuracy train for " + str(feat) + " features: " + str(acc1) + '\n')
        train_accuracy.append(acc1)
        conf_matrix = confusion_matrix(YTrain, pred1)
        train_text += (str(conf_matrix)+'\n')
        train_text += "Cost function value of " + str(feat) + " features: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_tr.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

        differenceBetweenAccuracies.append((acc1-acc))
    
#    tests, = plt.plot( max_feat, test_accuracy, color = 'r', label = 'Test')
#    trains, = plt.plot( max_feat, train_accuracy, color='b', label = 'Train')
#    plt.legend([trains, tests], ['Train','Test'])
##    plt.xticks(depths, Labels)
#    plt.ylabel('Accuracy')
#    plt.xlabel('Max Features')
#    plt.yticks([75,80,85,90,95,100, 105])
#    plt.savefig(('Colposcopy\\Graphs\\max_features\\graph'+str(feat)+'feat'+str(crit)+"_run_num_"+str(num)+".png"), dpi=100)
#    plt.show()
# 

#   print(differenceBetweenAccuracies)
#    n = (minDifference(differenceBetweenAccuracies, test_accuracy, train_accuracy))
#    print('index: '+str(n)+'\nvalue: '+str(differenceBetweenAccuracies[n])+'\ntrain_acc: ' + str(train_accuracy[n]) + '\ntest_acc' + str(test_accuracy[n]))
#    
    c_test.write(str(costs_ts))
    c_test.close()
    c_train.write(str(costs_tr))
    c_train.close()
    d_test.write(test_text)
    d_train.write(train_text)
    d_train.close()
    d_test.close()
    print( 'Ended max_features of ' + crit)
    
    
    return train_accuracy, test_accuracy

    
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<< Vary Sample Splits >>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
def min_sample_splits(crit, XTrain, YTrain,X_test, y_test, num):
    test_accuracy=[]
    train_accuracy = []
    differenceBetweenAccuracies=[]
    d_train=open("Colposcopy\\Matrix\\min_sample_splits\\"+str(crit)+"_run_num_"+str(num)+"_train.txt",'w')
    d_test=open("Colposcopy\\Matrix\\min_sample_splits\\"+str(crit)+"_run_num_"+str(num)+"_test.txt",'w')
    train_text = ""
    test_text = ""
    costs_tr=[]
    costs_ts=[]
    
    c_test = open("Colposcopy\\Matrix\\min_sample_splits\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_test.txt",'w')
    c_train = open("Colposcopy\\Matrix\\min_sample_splits\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_train.txt",'w')
   
    minSplit=[0.05,0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4,0.45, 0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8,0.85,0.9]
    for split in minSplit:
        
#        d_train=open("Matrix\\minSplit\\"+str(split)+"splits_"+str(crit)+"_train.txt",'w')
#        d_test=open("Matrix\\minSplit\\"+str(split)+"splits_"+str(crit)+"_test.txt",'w')
        clf = RandomForestClassifier(criterion=crit, min_samples_split=split)
        
        clf.fit(XTrain, YTrain)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)*100
        test_text += ("Accuracy test for "+str(split) +" min_splits: " + str(acc)+'\n')
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test, predict)
        test_text += (str(conf_matrix)+'\n')
        test_text += "Cost function value of " + str(split) + " min_splits: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_ts.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

        
        pred1= clf.predict(XTrain)
        acc1 = accuracy_score(YTrain, pred1)*100
        train_text += ("Accuracy train for "+str(split) +" min_splits: " + str(acc)+'\n')
        train_accuracy.append(acc1)
        conf_matrix1 = confusion_matrix(YTrain, pred1)
        train_text += (str(conf_matrix1)+'\n')
        train_text += "Cost function value of " + str(split) + " min_splits: " + str(10*conf_matrix1[0][1]+500*conf_matrix1[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+" False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_tr.append((10*conf_matrix1[0][1]+500*conf_matrix1[1][0]))

        differenceBetweenAccuracies.append(abs(acc1-acc))
        
    
    Labels =  [0.05,0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4,0.45, 0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8,0.85,0.9]
    #[0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
#    tests, = plt.plot( minSplit, test_accuracy, color = 'r', label = 'Test')
#    trains, = plt.plot( minSplit, train_accuracy, color='b', label = 'Train')
#    plt.plot( minSplit, test_accuracy, 'ro', color = 'r', label = 'Test')
#    plt.plot( minSplit, train_accuracy, 'ro', color='b', label = 'Train')
#    plt.legend([trains, tests], ['Train','Test'])
#    plt.xticks(minSplit, Labels)
#    plt.ylabel('Accuracy')
#    plt.xlabel('Min Sample Splits')
#    plt.yticks([0, 5, 10,15, 20, 25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105])
#    plt.savefig(('Colposcopy\\Graphs\\min_sample_splits\\graph'+str(split)+'split'+str(crit)+"_run_num_"+str(num)+".png"), dpi=100)
#    plt.show()
#    print(differenceBetweenAccuracies)
#    n = (minDifference(differenceBetweenAccuracies, test_accuracy, train_accuracy))
#    print('index: '+str(n)+'\nvalue: '+str(differenceBetweenAccuracies[n])+'\ntrain_acc: ' + str(train_accuracy[n]) + '\ntest_acc' + str(test_accuracy[n]))
    
    
    c_test.write(str(costs_ts))
    c_test.close()
    c_train.write(str(costs_tr))
    c_train.close()
    
    
    d_train.write(train_text)
    d_test.write(test_text)
    
    d_train.close()
    d_test.close()
    
    print( 'Ended min_sample_split of ' + crit)
    
    return train_accuracy, test_accuracy

"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<< Calls to functions >>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""

def min_sample_leafs(crit, XTrain, YTrain,X_test, y_test, num):
    test_accuracy=[]
    train_accuracy = []
    train_text = ""
    test_text = ""
    costs_tr=[]
    costs_ts=[]
    
    c_test = open("Colposcopy\\Matrix\\min_sample_leafs\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_test.txt",'w')
    c_train = open("Colposcopy\\Matrix\\min_sample_leafs\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_train.txt",'w')
   
#    differenceBetweenAccuracies=[]
    minLeaf=[0.05,0.1, 0.15,0.2,0.25,0.3,0.35, 0.4, 0.45, 0.5]#range(1,1000,50)
    d_train=open("Colposcopy\\Matrix\\min_sample_leafs\\min_leafs"+str(crit)+"_run_num_"+str(num)+"_train.txt",'w')
    d_test=open("Colposcopy\\Matrix\\min_sample_leafs\\min_leafs"+str(crit)+"_run_num_"+str(num)+"_test.txt",'w')
    for leafs in minLeaf:
        clf = RandomForestClassifier(criterion=crit, min_samples_leaf=leafs)
        
        clf.fit(XTrain, YTrain)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)*100
        test_text += ("Accuracy test for "+str(leafs) +" min_leafs: " + str(acc)+'\n')
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test, predict)
        test_text += (str(conf_matrix)+"\n")
        test_text += "Cost function value of " + str(leafs) + " min_leafs: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_ts.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

        pred1= clf.predict(XTrain)
        acc1 = accuracy_score(YTrain, pred1)*100
        train_text += ("Accuracy train for "+str(leafs) +" min_leafs: " + str(acc)+'\n')
        train_accuracy.append(acc1)
        conf_matrix = confusion_matrix(YTrain, pred1)
        train_text += (str(conf_matrix)+"\n")
        train_text += "Cost function value of " + str(leafs) + " min_leafs: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_tr.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

#        differenceBetweenAccuracies.append(abs(acc1-acc))
    Labels=minLeaf
    
#    tests = plt.plot( minLeaf, test_accuracy, color = 'r', label = 'Test')
#    trains = plt.plot( minLeaf, train_accuracy, color='b', label = 'Train')
#    plt.plot( minLeaf, test_accuracy, 'ro', color = 'r', label = 'Test')
#    plt.plot( minLeaf, train_accuracy, 'ro', color='b', label = 'Train')
#    plt.legend([trains, tests], ['Train','Test'])
#    plt.xticks(minLeaf, Labels)
#    plt.ylabel('Accuracy')
#    plt.xlabel('Min Sample Leafs')
#    plt.yticks([0, 5, 10,15, 20, 25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105])
#    plt.savefig('Colposcopy\\Graphs\\min_sample_leafs\\graph'+str(leafs)+'leafs'+str(crit)+"_run_num_"+str(num)+".png", dpi=100) #(
#    plt.show()
#    print(differenceBetweenAccuracies)
#    n = (minDifference(differenceBetweenAccuracies, test_accuracy,train_accuracy))
#    print('index: '+str(minLeaf[n])+'\nvalue: '+str(differenceBetweenAccuracies[n])+'\ntrain_acc: ' + str(train_accuracy[n]) + '\ntest_acc' + str(test_accuracy[n]))
    
    c_test.write(str(costs_ts))
    c_test.close()
    c_train.write(str(costs_tr))
    c_train.close()
    
    d_test.write(test_text)
    d_train.write(train_text)
    
    d_train.close()
    d_test.close()

    print('Ended min_sample_leaf of ' + crit)
    
    return train_accuracy, test_accuracy


"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Max Depths >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
def max_depth(crit, XTrain, YTrain,X_test, y_test, num):
    test_accuracy=[]
    train_accuracy = []
    differenceBetweenAccuracies=[]
    depths = range(1,11)

    train_text = ""
    test_text = ""
    costs_tr=[]
    costs_ts=[]
    
    c_test = open("Colposcopy\\Matrix\\max_depth\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_test.txt",'w')
    c_train = open("Colposcopy\\Matrix\\max_depth\\cost_"+str(crit)+"estimators"+"_run_num_"+str(num)+"_train.txt",'w')
   
    d_train=open("Colposcopy\\Matrix\\max_depth\\"+str(crit)+"_run_num_"+str(num)+"_train.txt",'w')
    d_test=open("Colposcopy\\Matrix\\max_depth\\"+str(crit)+"_run_num_"+str(num)+"_test.txt",'w')
    for depth in depths:
        
#        d_train=open("Matrix\\feature\\"+str(feat)+"features_"+str(crit)+"_train.txt",'w')
#        d_test=open("Matrix\\feature\\"+str(feat)+"features_"+str(crit)+"_test.txt",'w')

        clf = RandomForestClassifier(criterion=crit, max_depth=depth)
            
        clf.fit(XTrain, YTrain)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)*100
        test_text += ("Accuracy test for " + str(depth) + " depth: " + str(acc)+'\n')
        test_accuracy.append(acc)
        conf_matrix = confusion_matrix(y_test, predict)
        test_text += (str(conf_matrix)+'\n')
        test_text += "Cost function value fpr a depth of " + str(depth) + " is: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_ts.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

        pred1= clf.predict(XTrain)
        acc1 = accuracy_score(YTrain, pred1)*100
        train_text += ("Accuracy train for " + str(depth) + " depth: " + str(acc1) + '\n')
        train_accuracy.append(acc1)
        conf_matrix = confusion_matrix(YTrain, pred1)
        train_text += (str(conf_matrix)+'\n')
        train_text += "Cost function value for a depth of " + str(depth) + " is: " + str(10*conf_matrix[0][1]+500*conf_matrix[1][0])+ "\nFalse Positives: "+str(conf_matrix[0][1])+"False Negatives: " + str(conf_matrix[1][0])+"\n\n"
        costs_tr.append((10*conf_matrix[0][1]+500*conf_matrix[1][0]))

        differenceBetweenAccuracies.append((acc1-acc))
#    print (test_accuracy)
#    print (train_accuracy)
    

#    Labels = range(1,11)
##    
#    tests, = plt.plot( depths, test_accuracy, color = 'r', label = 'Test')
#    trains, = plt.plot( depths, train_accuracy, color='b', label = 'Train')
#    plt.plot( depths, test_accuracy, 'ro', color = 'r')
#    plt.plot( depths, train_accuracy, 'ro', color='b')
#    plt.legend([trains, tests], ['Train','Test'])
#    plt.xticks(depths, Labels)
#    plt.ylabel('Accuracy')
#    plt.xlabel('Max Depth')
##    plt.yticks([75,80,85,90,95,100,105])
#    plt.savefig(('Colposcopy\\Graphs\\max_depth\\graph'+str(depth)+'depth'+str(crit)+"_run_num_"+str(num)+".png"), dpi=100)
#    plt.show()
    
#    print(differenceBetweenAccuracies)
#    n = (minDifference(differenceBetweenAccuracies, test_accuracy, train_accuracy))
#    print('index: '+str(n)+'\nvalue: '+str(differenceBetweenAccuracies[n])+'\ntrain_acc: ' + str(train_accuracy[n]) + '\ntest_acc' + str(test_accuracy[n]))
    


    c_test.write(str(costs_ts))
    c_test.close()
    c_train.write(str(costs_tr))
    c_train.close()
    
    d_test.write(test_text)
    d_train.write(train_text)
    d_train.close()
    d_test.close()
    print( 'Ended max_depth of ' + crit)
    
    return train_accuracy, test_accuracy


#def calc_avg_vecs!(ListOfListAccuracies):
#    listForAvg = np.empty(len(ListOfListAccuracies[1]))
#    for listOfAcc in ListOfListAccuracies:
#        n=0
#        while n<len(listOfAcc):
#            listForAvg[n]+=listOfAcc[n]
#            n+=1
#        print("List of Acc " + str(n) + ": "+str(listOfAcc))
#    n=0
#    while (n<len(listForAvg)):
#        listForAvg[n] = listForAvg[n]/len(ListOfListAccuracies)
#        n+=1
#    print (listForAvg)
#    print (len(ListOfListAccuracies))
#    return listForAvg
    
def calc_avg_vecs(ListOfList):
    averages = []
    i = 0
    while i < len(ListOfList[0]):
        sum_values = 0
        for listof in ListOfList:
            sum_values+= listof[i]
        averages.append(sum_values/len(ListOfList))
        i+=1
    return averages
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< RUN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""

def runAll():
    test_num_estimators_entropy = []
    train_num_estimators_entropy = []
    test_num_estimators_gini = []
    train_num_estimators_gini = []

    test_max_features_entropy = []
    train_max_features_entropy = []
    test_max_features_gini = []
    train_max_features_gini = []
    
    test_min_sample_splits_entropy = []
    train_min_sample_splits_entropy = []
    test_min_sample_splits_gini = []
    train_min_sample_splits_gini = []
    
    test_min_sample_leafs_entropy = []
    train_min_sample_leafs_entropy = []
    test_min_sample_leafs_gini = []
    train_min_sample_leafs_gini = []
    
    test_max_depth_entropy = []
    train_max_depth_entropy = []
    test_max_depth_gini = []
    train_max_depth_gini = []
    n=1
    for train_index, test_index in K_Fold.split(X, y):
        
        estims_tr_e, estims_ts_e = num_estimators('entropy',X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_num_estimators_entropy.append(estims_ts_e)
        train_num_estimators_entropy.append(estims_tr_e)

        
        
        estims_tr_g, estims_ts_g = num_estimators('gini', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_num_estimators_gini.append(estims_ts_g)
        train_num_estimators_gini.append(estims_tr_g)
        
        leafs_tr_e, leafs_ts_e = min_sample_leafs('entropy', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_min_sample_leafs_entropy.append(leafs_ts_e)
        train_min_sample_leafs_entropy.append(leafs_tr_e)
        
        leafs_tr_g, leafs_ts_g = min_sample_leafs('gini',X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_min_sample_leafs_gini.append(leafs_ts_g)
        train_min_sample_leafs_gini.append(leafs_tr_g)
        
        feats_tr_e, feats_ts_e = max_features('entropy', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_max_features_entropy.append(feats_ts_e)
        train_max_features_entropy.append(feats_tr_e)
        
        feats_tr_g, feats_ts_g = max_features('gini', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_max_features_gini.append(feats_ts_g)
        train_max_features_gini.append(feats_tr_g)
        
        splits_tr_e, splits_ts_e = min_sample_splits('entropy', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_min_sample_splits_entropy.append(splits_ts_e)
        train_min_sample_splits_entropy.append(splits_tr_e)
        
        splits_tr_g, splits_ts_g = min_sample_splits('gini', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_min_sample_splits_gini.append(splits_ts_g)
        train_min_sample_splits_gini.append(splits_tr_g)
        
        depths_tr_e, depths_ts_e = max_depth('entropy', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_max_depth_entropy.append(depths_ts_e)
        train_max_depth_entropy.append(depths_tr_e)
        
        depths_tr_g, depths_ts_g = max_depth('gini', X[train_index], y[train_index], X[test_index], y[test_index],n)
        test_max_depth_gini.append(depths_ts_g)
        train_max_depth_gini.append(depths_tr_g)

        n+=1
    
    test_min_sample_leafs_entropy_avg = calc_avg_vecs(test_min_sample_leafs_entropy)
    train_min_sample_leafs_entropy_avg = calc_avg_vecs(train_min_sample_leafs_entropy)
    test_min_sample_leafs_gini_avg = calc_avg_vecs(test_min_sample_leafs_gini)
    train_min_sample_leafs_gini_avg = calc_avg_vecs(train_min_sample_leafs_gini)
    
    draw_graphs('entropy', [0.05,0.1, 0.15,0.2,0.25,0.3,0.35, 0.4, 0.45, 0.5], 'min_sample_leafs', test_min_sample_leafs_entropy_avg,train_min_sample_leafs_entropy_avg,)
    draw_graphs('gini', [0.05,0.1, 0.15,0.2,0.25,0.3,0.35, 0.4, 0.45, 0.5], 'min_sample_leafs', test_min_sample_leafs_gini_avg,train_min_sample_leafs_gini_avg)    
    
    
    test_min_sample_splits_entropy_avg = calc_avg_vecs(test_min_sample_splits_entropy)
    train_min_sample_splits_entropy_avg = calc_avg_vecs(train_min_sample_splits_entropy)
    test_min_sample_splits_gini_avg = calc_avg_vecs(test_min_sample_splits_gini)
    train_min_sample_splits_gini_avg = calc_avg_vecs(train_min_sample_splits_gini)
    
    
    draw_graphs('entropy',[0.05,0.1, 0.15,0.2,0.25,0.3,0.35, 0.4, 0.45, 0.5, 0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9], 'min_sample_splits',test_min_sample_splits_entropy_avg, train_min_sample_splits_entropy_avg )
    draw_graphs('gini',[0.05,0.1, 0.15,0.2,0.25,0.3,0.35, 0.4, 0.45, 0.5, 0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9], 'min_sample_splits',test_min_sample_splits_gini_avg, train_min_sample_splits_gini_avg )
    
    
    test_max_depth_entropy_avg = calc_avg_vecs(test_max_depth_entropy)
    train_max_depth_entropy_avg = calc_avg_vecs(train_max_depth_entropy)
    test_max_depth_gini_avg = calc_avg_vecs(test_max_depth_gini)
    train_max_depth_gini_avg = calc_avg_vecs(train_max_depth_gini)
    
    draw_graphs('entropy', range(1,11), 'max_depth', test_max_depth_entropy_avg, train_max_depth_entropy_avg)
    draw_graphs('gini', range(1,11), 'max_depth', test_max_depth_gini_avg, train_max_depth_gini_avg)
    
    
    test_max_features_entropy_avg = calc_avg_vecs(test_max_features_entropy)
    train_max_features_entropy_avg = calc_avg_vecs(train_max_features_entropy)
    test_max_features_gini_avg = calc_avg_vecs(test_max_features_gini)
    train_max_features_gini_avg = calc_avg_vecs(train_max_features_gini)
    
    draw_graphs('entropy', range(1,63), 'max_features', test_max_features_entropy_avg, train_max_features_entropy_avg)
    draw_graphs('gini',  range(1,63), 'max_features', test_max_features_gini_avg, train_max_features_gini_avg)

    
    test_num_estimators_entropy_avg = calc_avg_vecs(test_num_estimators_entropy)
    train_num_estimators_entropy_avg = calc_avg_vecs(train_num_estimators_entropy)
    test_num_estimators_gini_avg = calc_avg_vecs(test_num_estimators_gini)
    train_num_estimators_gini_avg = calc_avg_vecs(train_num_estimators_gini)
    
    draw_graphs('entropy', [10,20,30,40,50,60,70,80,90,100], 'num_estimators', test_num_estimators_entropy_avg, train_num_estimators_entropy_avg)
    draw_graphs('gini', [10,20,30,40,50,60,70,80,90,100], 'num_estimators', test_num_estimators_gini_avg, train_num_estimators_gini_avg)
    
    doc=open('optimal_colposcopy.txt', 'w')
    doc.write('Max_depth:\n Entropy test, train; Gini test, train\n'+ str(test_max_depth_entropy_avg) + '\n' + str(train_max_depth_entropy_avg) + '\n\n' +
              str(test_max_depth_gini_avg) + '\n' + str(train_max_depth_gini_avg) + '\n\n' + 
              'num_estimators:\n Entropy test, train; Gini test, train\n'+ str(test_num_estimators_entropy_avg) + '\n' + str(train_num_estimators_entropy_avg) + '\n\n' +
              str(test_num_estimators_gini_avg) + '\n' + str(train_num_estimators_gini_avg) + '\n\n' + 
              'max_features:\n Entropy test, train; Gini test, train\n'+ str(test_max_features_entropy_avg) + '\n' + str(train_max_features_entropy_avg) + '\n\n' +
              str(test_max_features_gini_avg) + '\n' + str(train_max_features_gini_avg) + '\n\n' + 
              'min_leafs:\n Entropy test, train; Gini test, train\n'+ str(test_min_sample_leafs_entropy_avg) + '\n' + str(train_max_depth_entropy_avg) + '\n\n' +
              str(test_min_sample_leafs_gini_avg) + '\n' + str(train_min_sample_leafs_gini_avg) + '\n\n' + 
              'min_splits:\n Entropy test, train; Gini test, train\n'+ str(test_min_sample_splits_entropy_avg) + '\n' + str(train_min_sample_splits_entropy_avg) + '\n\n' +
              str(test_min_sample_splits_gini_avg) + '\n' + str(train_min_sample_splits_gini_avg) + '\n\n')
    doc.close()
    
runAll()