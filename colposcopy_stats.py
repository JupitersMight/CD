import numpy as np
import pandas as pd


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

def number_of_outliers(vector):
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

df_green = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\green.csv')
df_hinselmann = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\hinselmann.csv')
df_schiller = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\schiller.csv')
combined = pd.concat([df_green, df_hinselmann, df_schiller])

final_columns = []
i=0
for column in df_green.columns:
    if i == len(df_green.columns)-7:
        break
    i+=1
    final_columns.append(column)


counter = 0
for column in final_columns:
    counter += number_of_outliers(df_green[column])

print(counter)


counter = 0
for column in final_columns:
    counter += number_of_outliers(df_hinselmann[column])

print(counter)


counter = 0
for column in final_columns:
    counter += number_of_outliers(df_schiller[column])

print(counter)


counter = 0
for column in final_columns:
    counter += number_of_outliers(combined[column])

print(counter)



consensus = combined.iloc[:,len(combined.columns)-1]

df_g = df_green[df_green['consensus'] == 1]
df_h = df_hinselmann[df_hinselmann['consensus'] == 1]
df_s = df_schiller[df_schiller['consensus'] == 1]

print(len(df_green.iloc[:,0]))
print(len(df_g.iloc[:,0]))
print(len(df_hinselmann.iloc[:,0]))
print(len(df_h.iloc[:,0]))
print(len(df_schiller.iloc[:,0]))
print(len(df_s.iloc[:,0]))


file = open('statistics_col.txt', 'a')
file.write('Number of attributes : '+str(len(combined.columns))+' | Number of rows : '+str(len(combined.iloc[:, [1]]))+'\n')
file.write('Unique classification : '+str(combined.columns[len(combined.columns)-7:len(combined.columns)])+'\n')
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
