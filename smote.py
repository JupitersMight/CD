import pandas as pd
from imblearn.over_sampling import SMOTE

#          WARNING !!! In order to use SMOTE the dataset must not contain missing values          #

# FILES
NAME_OF_DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv'
NAME_OF_SMOTE_OVER_SAMPLING_FILE = 'smote_over_sampling_aps_training.csv'

# CLASS ATTRIBUTE
INDEX_OF_CLASS_ATTRIBUTE = 0
NAME_OF_CLASS_ATTRIBUTE = 'class'

# Read file
df_train = pd.read_csv(NAME_OF_DATASET_FILE, delimiter=',')

# Smote over sampling
X = df_train.iloc[:, df_train.columns != NAME_OF_CLASS_ATTRIBUTE].values
Y = df_train.iloc[:, INDEX_OF_CLASS_ATTRIBUTE]
smote = SMOTE(ratio='minority')
X_res, Y_res = smote.fit_sample(X, Y)
result = Y_res.append(X_res)

# Write to files
result.to_csv(NAME_OF_SMOTE_OVER_SAMPLING_FILE, encoding='utf-8', index=False)
