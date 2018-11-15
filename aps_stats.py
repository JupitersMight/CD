import numpy as np
import pandas as pd

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

def number_of_outliers(vector):
    vector = pd.Series(vector)
    Q1 = vector.quantile(0.25)
    Q3 = vector.quantile(0.75)
    IQR = Q3 - Q1
    low_interval = Q1 - 1.5 * IQR
    high_interval = Q3 + 1.5 * IQR
    counter = 0
    for value in vector:
        if value < low_interval:
            counter += 1
        if value > high_interval:
            counter += 1
    return counter

df_training_types = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv', delimiter=',',na_values=['na'])
df_training = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv')
df_test = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstest.csv')
final_columns = []
i=0
for column in df_training.columns:
    if i == 0:
        i+=1
        continue
    final_columns.append(column)
    i+=1

count_outliers = 0
for column in final_columns:
    vector = []
    for value in df_training[column]:
        if value == 'na':
            continue
        vector.append(float(value))
    count_outliers += number_of_outliers(vector)

print(count_outliers)

count_outliers = 0
for column in final_columns:
    vector = []
    for value in df_test[column]:
        if value == 'na':
            continue
        vector.append(float(value))
    count_outliers += number_of_outliers(vector)

print(count_outliers)

df_class_neg = df_test[df_test["class"] == "neg"]
df_class_pos = df_test[df_test["class"] == "pos"]

print(len(df_class_neg))
print(len(df_class_pos))

means_neg = calculate_mean(df_class_neg)
means_pos = calculate_mean(df_class_pos)

missings_neg = number_of_missing_values(df_class_neg)
missings_pos = number_of_missing_values(df_class_pos)

medians_neg = calculate_meadian(df_class_neg)
medians_pos = calculate_meadian(df_class_pos)

file = open('statistics.txt', 'a')
file.write('Number of attributes : '+str(len(df_training.columns))+' | Number of rows : '+str(len(df_training['class']))+'\n')
file.write('Unique classification : '+str(pd.unique(df_training['class']))+'\n')
file.write('Number of neg rows : '+str(len(df_class_neg['class']))+'\n')
file.write('Number of pos rows : '+str(len(df_class_pos['class']))+'\n')
file.write('Column types : '+str(df_training_types.dtypes)+'\n')
print(sum(missings_neg))
print(sum(missings_pos))

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