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

def outliers_interval(vector):
    Q1 = vector.quantile(0.25)
    Q3 = vector.quantile(0.75)
    IQR = Q3 - Q1
    low_interval = Q1 - 1.5 * IQR
    high_interval = Q3 + 1.5 * IQR
    return [low_interval, high_interval]

df_green = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\green.csv')
df_hinselmann = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\hinselmann.csv')
df_schiller = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\schiller.csv')
combined = pd.concat([df_green, df_hinselmann, df_schiller]).reset_index(drop=True).drop(columns=['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5'])

result = []
for column in combined.columns:
    if column == 'consensus':
        continue
    result.append(outliers_interval(combined[column]))


column_intervals = []
for column in combined.columns:
    if column == 'consensus':
        continue
    column_intervals.append(outliers_interval(combined[column]))

i=0
for column in combined.columns:
    if column == 'consensus':
        continue
    combined = combined[combined[column] >= result[i][0]]
    combined = combined[combined[column] <= result[i][1]]
    i+=1

combined = combined.reset_index(drop=True)

print(len(combined['consensus']))
print(len(combined[combined['consensus'] == 1]))

combined.to_csv('col_wihtout_outliers.csv', encoding='utf-8', index=False)

df_train = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_training_average.csv')

result = []
for column in df_train.columns:
    if column == 'class':
        continue
    result.append(outliers_interval(df_train[column]))


column_intervals = []
for column in df_train.columns:
    if column == 'class':
        continue
    column_intervals.append(outliers_interval(df_train[column]))

i=0
df_neg = df_train[df_train['class'] == 'neg']
df_pos = df_train[df_train['class'] == 'pos']

for column in df_neg.columns:
    if column == 'class':
        continue
    df_neg = df_neg[df_neg[column] >= result[i][0]]
    df_neg = df_neg[df_neg[column] <= result[i][1]]
    i+=1

df_neg = df_neg.reset_index(drop=True)

df_train = pd.concat([df_pos, df_neg]).reset_index(drop=True)

print(len(df_train['class']))
print(len(df_train[df_train['class'] == 'pos']))

df_train.to_csv('aps_training_without_outliers.csv', encoding='utf-8', index=False)



