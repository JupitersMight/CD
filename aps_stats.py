import numpy as np
import scipy as sp
import matplotlib as plt
import pandas as pd
import seaborn as sea
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from statistics import mean, median
from copy import copy


def calculate_mean(dataset):
    mean_values = []
    for column_name in dataset.columns:
        if column_name == "class":
            continue
        sum_values = 0
        i = 0
        for value in dataset[column_name]:
            if value == 'na':
                continue
            sum_values += float(value)
            i += 1
        mean_values.append(str(sum_values/i))
    return mean_values

def number_of_missing_values(dataset):
    missing_values = []
    for column_name in dataset.columns:
        if column_name == "class":
            continue
        counter = 0
        for value in dataset[column_name]:
            if value != 'na':
                continue
            counter += 1
        missing_values.append(counter)
    return missing_values

def calculate_meadian(dataset):
    median_values = []
    for column_name in dataset.columns:
        if column_name == "class":
            continue
        values = []
        for value in dataset[column_name]:
            if value == 'na':
                continue
            values.append(float(value))
        median_values.append(np.median(values))
    return median_values


df_training_types = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv', delimiter=',',na_values=['na'])
df_training = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv')

df_class_neg = df_training[df_training["class"] == "neg"]
df_class_pos = df_training[df_training["class"] == "pos"]

means_neg = calculate_mean(df_class_neg)
means_pos = calculate_mean(df_class_pos)

missings_neg = number_of_missing_values(df_class_neg)
missings_pos = number_of_missing_values(df_class_neg)

medians_neg = calculate_meadian(df_class_neg)
medians_pos = calculate_meadian(df_class_pos)

file = open('statistics.txt', 'a')
file.write('Number of attributes : '+str(len(df_training.columns))+' | Number of rows : '+str(len(df_training['class']))+'\n')
file.write('Unique classification : '+str(pd.unique(df_training['class']))+'\n')
file.write('Number of neg rows : '+str(len(df_class_neg['class']))+'\n')
file.write('Number of pos rows : '+str(len(df_class_pos['class']))+'\n')
file.write('Column types : '+str(df_training_types.dtypes)+'\n')

file.write('class neg : \n')
i = 0
for column in df_training.columns:
    if column == 'class':
        continue
    file.write('Attribute '+str(column)+' mean : '+means_neg[i]+' | meadian : '+str(medians_neg[i])+' | missings : '+str(missings_neg[i])+'\n')
    i += 1

file.write('class pos : \n')
i = 0
for column in df_training.columns:
    if column == 'class':
        continue
    file.write('Attribute '+str(column)+' mean : '+means_pos[i]+' | meadian : '+str(medians_pos[i])+' | missings : '+str(missings_pos[i])+'\n')
    i += 1
file.close()
print('done')
