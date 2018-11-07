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
    for idx, column_name in enumerate(dataset.columns):
        if idx >= len(dataset.columns)-7:
            break
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
    for idx, column_name in enumerate(dataset.columns):
        if idx >= len(dataset.columns)-7:
            break
        counter = 0
        for value in dataset[column_name]:
            if value != 'na':
                continue
            counter += 1
        missing_values.append(counter)
    return missing_values

def calculate_meadian(dataset):
    median_values = []
    for idx, column_name in enumerate(dataset.columns):
        if idx >= len(dataset.columns)-7:
            break
        values = []
        for value in dataset[column_name]:
            if value == 'na':
                continue
            values.append(float(value))
        median_values.append(np.median(values))
    return median_values

def number_of_positives(column):
    counter = 0
    for value in column:
        if value == 1:
            counter += 1
    return counter

df_green = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\green.csv')
df_hinselmann = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\hinselmann.csv')
df_schiller = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\schiller.csv')

combined = pd.concat([df_green, df_hinselmann, df_schiller])

experts_0 = combined.iloc[:,len(combined.columns)-7]
experts_1 = combined.iloc[:,len(combined.columns)-6]
experts_2 = combined.iloc[:,len(combined.columns)-5]
experts_3 = combined.iloc[:,len(combined.columns)-4]
experts_4 = combined.iloc[:,len(combined.columns)-3]
experts_5 = combined.iloc[:,len(combined.columns)-2]
consensus = combined.iloc[:,len(combined.columns)-1]

file = open('statistics_col.txt', 'a')
file.write('Number of attributes : '+str(len(combined.columns))+' | Number of rows : '+str(len(combined.iloc[:, [1]]))+'\n')
file.write('Unique classification : '+str(combined.columns[len(combined.columns)-7:len(combined.columns)])+'\n')
file.write('Number of experts_0 positive rows : '+str(number_of_positives(experts_0))+'\n')
file.write('Number of experts_1 positive rows : '+str(number_of_positives(experts_1))+'\n')
file.write('Number of experts_2 positive rows : '+str(number_of_positives(experts_2))+'\n')
file.write('Number of experts_3 positive rows : '+str(number_of_positives(experts_3))+'\n')
file.write('Number of experts_3 positive rows : '+str(number_of_positives(experts_3))+'\n')
file.write('Number of experts_4 positive rows : '+str(number_of_positives(experts_4))+'\n')
file.write('Number of consensus positive rows : '+str(number_of_positives(consensus))+'\n')
file.write('Column types : '+str(combined.dtypes)+'\n')

means = calculate_mean(combined)

missings = number_of_missing_values(combined)

medians = calculate_meadian(combined)

file.write('Attributes : \n')
i = 0
for idx, column in enumerate(combined.columns):
    if idx >= len(combined.columns) - 7:
        break
    file.write('Attribute '+str(column)+' mean : '+means[i]+' | meadian : '+str(medians[i])+' | missings : '+str(missings[i])+'\n')
    i += 1
file.close()
print('done')
