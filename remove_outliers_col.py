import pandas as pd

def outliers_interval(vector):
    Q1 = vector.quantile(0.25)
    Q3 = vector.quantile(0.75)
    IQR = Q3 - Q1
    low_interval = Q1 - 1.5 * IQR
    high_interval = Q3 + 1.5 * IQR
    return [low_interval, high_interval]

df_green = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\green.csv').drop(columns=['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5'])
df_hinselmann = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\hinselmann.csv').drop(columns=['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5'])
df_schiller = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\schiller.csv').drop(columns=['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5'])

result = []
for column in df_green.columns:
    if column == 'consensus':
        continue
    result.append(outliers_interval(df_green[column]))


column_intervals = []
for column in df_green.columns:
    if column == 'consensus':
        continue
    column_intervals.append(outliers_interval(df_green[column]))

i=0
for column in df_green.columns:
    if column == 'consensus':
        continue
    df_green = df_green[df_green[column] >= result[i][0]]
    df_green = df_green[df_green[column] <= result[i][1]]
    i+=1

result = []
for column in df_hinselmann.columns:
    if column == 'consensus':
        continue
    result.append(outliers_interval(df_hinselmann[column]))


column_intervals = []
for column in df_hinselmann.columns:
    if column == 'consensus':
        continue
    column_intervals.append(outliers_interval(df_hinselmann[column]))

i=0
for column in df_hinselmann.columns:
    if column == 'consensus':
        continue
    df_hinselmann = df_hinselmann[df_hinselmann[column] >= result[i][0]]
    df_hinselmann = df_hinselmann[df_hinselmann[column] <= result[i][1]]
    i+=1

result = []
for column in df_schiller.columns:
    if column == 'consensus':
        continue
    result.append(outliers_interval(df_schiller[column]))


column_intervals = []
for column in df_schiller.columns:
    if column == 'consensus':
        continue
    column_intervals.append(outliers_interval(df_schiller[column]))

i=0
for column in df_schiller.columns:
    if column == 'consensus':
        continue
    df_schiller = df_schiller[df_schiller[column] >= result[i][0]]
    df_schiller = df_schiller[df_schiller[column] <= result[i][1]]
    i+=1


combined = pd.concat([df_green, df_hinselmann, df_schiller]).reset_index(drop=True)

print(len(combined['consensus']))
print(len(combined[combined['consensus'] == 1]))

combined.to_csv('col_wihtout_outliers_or_balancing.csv', encoding='utf-8', index=False)

df_train = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_test_average.csv')

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

df_train.to_csv('aps_testing_without_outliers.csv', encoding='utf-8', index=False)



