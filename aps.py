import pandas as pd

def preprocess_substitute_na_with_average(dataset):

    list_of_columns = dataset.columns
    array_of_mean_values = []
    for column_name in list_of_columns:
        if column_name == "class":
            continue
        list_of_items = dataset[column_name]

        array_of_mean_values.append(calculate_mean_with_na(list_of_items))

    i = 0
    for column_name in list_of_columns:
        if column_name == "class":
            continue
        dataset[column_name].fillna(str(array_of_mean_values[i]))
        i += 1

def calculate_mean_with_na(list):
    sum_value = 0
    counter = 0
    for value in list:
        if value != "na":
            if isinstance(value, str):
                value = float(value)
            sum_value = sum_value + value
            counter += 1
    return sum_value/counter


df_train = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv', delimiter=',',na_values=['na'])
df_train.columns = df_train.columns.str.strip()

count_class_neg, count_class_pos = df_train["class"].value_counts()
df_class_neg = df_train[df_train["class"] == "neg"]
df_class_pos = df_train[df_train["class"] == "pos"]

df_class_neg_under = df_class_neg.sample(count_class_pos)
df_train_balanced = pd.concat([df_class_neg_under, df_class_pos], axis=0)

preprocess_substitute_na_with_average(df_train_balanced)

df_train.to_csv("aps_train_removed_na.csv", encoding='utf-8', index=False)
