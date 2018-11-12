import pandas as pd


# FILES
NAME_OF_DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_training_average.csv'
NAME_OF_UNDERSAMPLING_FILE = 'under_sampling_aps_training.csv'
NAME_OF_OVERSAMPLE_FILE = 'over_sampling_aps_training.csv'

# CLASS ATTRIBUTE
INDEX_OF_CLASS_ATTRIBUTE = 0
NAME_OF_CLASS_ATTRIBUTE = 'class'

# Read file
df_train = pd.read_csv(NAME_OF_DATASET_FILE, delimiter=',')
# Take out white spaces from column names
df_train.columns = df_train.columns.str.strip()

# How many rows each class attribute has
count_class_neg, count_class_pos = df_train[NAME_OF_CLASS_ATTRIBUTE].value_counts()

# Separate rows for each diferent class attribute
df_neg = df_train[df_train[NAME_OF_CLASS_ATTRIBUTE] == 'neg']
df_pos = df_train[df_train[NAME_OF_CLASS_ATTRIBUTE] == 'pos']

# Under sampling
#under = getNegBestRows(df_pos,count_class_pos)
under = df_neg.sample(count_class_pos)
df_train_balanced_under = pd.concat([under, df_pos], axis=0)

# Over sampling
over = df_pos.sample(count_class_neg, replace=True)
df_train_balanced_over = pd.concat([df_neg, over], axis=0)

# Write to files
df_train_balanced_under.to_csv(NAME_OF_UNDERSAMPLING_FILE, encoding='utf-8', index=False)
df_train_balanced_over.to_csv(NAME_OF_OVERSAMPLE_FILE, encoding='utf-8', index=False)
print('done')


# FILES
NAME_OF_DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_training_without_outliers.csv'
NAME_OF_UNDERSAMPLING_FILE = 'under_sampling_aps_training_without_outliers.csv'
NAME_OF_OVERSAMPLE_FILE = 'over_sampling_aps_training_without_outliers.csv'

# CLASS ATTRIBUTE
INDEX_OF_CLASS_ATTRIBUTE = 0
NAME_OF_CLASS_ATTRIBUTE = 'class'

# Read file
df_train = pd.read_csv(NAME_OF_DATASET_FILE, delimiter=',')
# Take out white spaces from column names
df_train.columns = df_train.columns.str.strip()

# How many rows each class attribute has
count_class_neg, count_class_pos = df_train[NAME_OF_CLASS_ATTRIBUTE].value_counts()

# Separate rows for each diferent class attribute
df_neg = df_train[df_train[NAME_OF_CLASS_ATTRIBUTE] == 'neg']
df_pos = df_train[df_train[NAME_OF_CLASS_ATTRIBUTE] == 'pos']

# Under sampling
#under = getNegBestRows(df_pos,count_class_pos)
under = df_neg.sample(count_class_pos)
df_train_balanced_under = pd.concat([under, df_pos], axis=0)

# Over sampling
over = df_pos.sample(count_class_neg, replace=True)
df_train_balanced_over = pd.concat([df_neg, over], axis=0)

# Write to files
df_train_balanced_under.to_csv(NAME_OF_UNDERSAMPLING_FILE, encoding='utf-8', index=False)
df_train_balanced_over.to_csv(NAME_OF_OVERSAMPLE_FILE, encoding='utf-8', index=False)
print('done')

