import pandas as pd

# Files names
FILE_NAME_FOR_AVERAGE_APS_TRAINING = 'aps_training_average.csv'
FILE_NAME_FOR_AVERAGE_APS_TEST = 'aps_test_average.csv'

# File path
APS_TRAINING = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv'
APS_TEST = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstest.csv'

def remove_na_training_data(dataset,neg_means,pos_means,consult):
    idx_mean = 0
    for column_name in dataset.columns:
        if column_name == 'class':
            continue
        idx_row = 0
        for value in dataset[column_name]:
            if value == 'na':
                if consult[idx_row] == 0:
                    dataset.at[idx_row, column_name] = neg_means[idx_mean]
                else:
                    dataset.at[idx_row, column_name] = pos_means[idx_mean]
            idx_row += 1
        idx_mean += 1
    return dataset


def remove_na_test_data(dataset, mean_values):
    i = 0
    for column_name in dataset.columns:
        if column_name == 'class':
            continue
        for idx, value in enumerate(dataset[column_name]):
            if value == 'na':
                dataset.at[idx, column_name] = mean_values[i]
        i += 1
    return dataset


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


# Load files
df_train = pd.read_csv(APS_TRAINING, delimiter=',')
df_test = pd.read_csv(APS_TEST, delimiter=',')

# Take out the white spaces from column names
df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

# For the training data separate neg and pos to calculate the mean and replace the NA with the mean
df_class_neg = df_train[df_train["class"] == "neg"]
df_class_pos = df_train[df_train["class"] == "pos"]
# Calculate the mean for the training and test dataset
means_neg = calculate_mean(df_class_neg)
means_pos = calculate_mean(df_class_pos)
mean_values = calculate_mean(df_test)

# Create a index array for where the neg and pos values are
consult = []
for value in df_train['class']:
    if value == 'neg':
        consult.append(0)
    else:
        consult.append(1)

# Replace NA with mean
df_train = remove_na_training_data(df_train, means_neg, means_pos, consult)
df_test = remove_na_test_data(df_test, mean_values)

# Write to files
df_train.to_csv(FILE_NAME_FOR_AVERAGE_APS_TRAINING, encoding='utf-8', index=False)
df_test.to_csv(FILE_NAME_FOR_AVERAGE_APS_TEST, encoding='utf-8', index=False)

# Just a print
print('done')
