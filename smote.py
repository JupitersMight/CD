import pandas as pd
from imblearn.over_sampling import SMOTE

#          WARNING !!! In order to use SMOTE the dataset must not contain missing values          #

# # FILES
# NAME_OF_DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_training_average.csv'
# NAME_OF_SMOTE_OVER_SAMPLING_FILE = 'smote_over_sampling_aps_training.csv'
#
# # CLASS ATTRIBUTE
# INDEX_OF_CLASS_ATTRIBUTE = 0
# NAME_OF_CLASS_ATTRIBUTE = 'class'
#
# # Read file
# df_train = pd.read_csv(NAME_OF_DATASET_FILE, delimiter=',')
#
# # Smote over sampling
# X = df_train.iloc[:, df_train.columns != NAME_OF_CLASS_ATTRIBUTE].values
# Y = df_train.iloc[:, INDEX_OF_CLASS_ATTRIBUTE]
# smote = SMOTE(ratio='minority')
# X_res, Y_res = smote.fit_sample(X, Y)
# result = pd.concat([pd.Series(Y_res), pd.DataFrame(data=X_res)], axis=1)
# result.columns = df_train.columns
#
# # Write to files
# result.to_csv(NAME_OF_SMOTE_OVER_SAMPLING_FILE, encoding='utf-8', index=False)
# print('done')


# COL

# Read file
green = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\green.csv', delimiter=',')
hinselmann = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\hinselmann.csv', delimiter=',')
schiller = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\schiller.csv', delimiter=',')
df_train = pd.concat([green, hinselmann, schiller])
# Smote over sampling
X = df_train.iloc[:, 0:len(df_train.columns)-7].values
Y = df_train.iloc[:, len(df_train.columns)-1]
smote = SMOTE(ratio='minority')
X_res, Y_res = smote.fit_sample(X, Y)
result = pd.concat([pd.Series(Y_res), pd.DataFrame(data=X_res)], axis=1)
final_columns = []
final_columns.append('consensus')
i=0
for column in df_train.columns:
    if i == len(df_train.columns)-7:
        break
    i+=1
    final_columns.append(column)
result.columns = final_columns

# Write to files
result.to_csv('smote_over_sampling_col.csv', encoding='utf-8', index=False)
print('done')
