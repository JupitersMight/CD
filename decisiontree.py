import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statistics


def preprocessData(df):
    label_encoder = LabelEncoder()
    dummy_encoder = OneHotEncoder()
    pdf = pd.DataFrame()
    for att in df.columns:
        if df[att].dtype == np.float64 or df[att].dtype == np.int64:
            pdf = pd.concat([pdf, df[att]], axis=1)
        else:
            df[att] = label_encoder.fit_transform(df[att])
            # Fitting One Hot Encoding on train data
            temp = dummy_encoder.fit_transform(df[att].values.reshape(-1,1)).toarray()
            # Changing encoded features into a dataframe with new column names
            temp = pd.DataFrame(temp,
                                columns=[(att + "_" + str(i)) for i in df[att].value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            pdf = pd.concat([pdf, temp], axis=1)
    return pdf


def print_statistics(welcome, accuracy_testing, accuracy_training, deviation, conf_matrix):
    print(welcome)
    print('decision tree was written to file')
    print()
    print('Accuracy for testing : ' + str(accuracy_testing))
    print()
    print('Accuracy for training : ' + str(accuracy_training))
    print()
    print('Confusion matrix : ')
    print(conf_matrix)
    print()
    print('Standard deviation : ' + str(deviation))
    print()
    interval = 1.96 * np.sqrt(1 - accuracy_testing * accuracy_testing / 16000)
    error = 1 - accuracy_testing
    low_interval = error - interval
    if low_interval < 0:
        low_interval = 0
    upper_interval = error + interval
    print('Confidence interval : With an error of ' + str(
        error) + ' the confidence interval on the classification error is [' + str(low_interval) + ',' + str(
        upper_interval) + ']')
    print()


def decision_tree(X_train, y_train, welcome, file, criterion, splitter, max_depth):
    clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
    clf.fit(X_train, y_train)
    predict_testing = clf.predict(X_test)
    predict_training = clf.predict(X_train)
    accuracy_testing = accuracy_score(y_test, predict_testing)
    accuracy_training = accuracy_score(y_train, predict_training)
    conf_m = confusion_matrix(y_test.argmax(axis=1), predict_testing.argmax(axis=1), labels=None, sample_weight=None)
    dotfile = open(file, 'w')
    tree.export_graphviz(clf, out_file=dotfile, feature_names=df_training_under.columns[2:])
    dotfile.close()
    print_statistics(welcome, accuracy_testing, accuracy_training, statistics.stdev(predict_testing.ravel()), conf_m)


# APS
# Load files

df_training = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_training_average.csv')
df_training = preprocessData(df_training)
df_training_under = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\under_sampling_aps_training.csv')
df_training_under = preprocessData(df_training_under)
df_training_over = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\over_sampling_aps_training.csv')
df_training_over = preprocessData(df_training_over)
df_training_smote = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\under_sampling_aps_training.csv')
df_training_smote = preprocessData(df_training_smote)
df_test = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_test_average.csv')
df_test = preprocessData(df_test)

X_train_normal = df_training_under.iloc[:, 2:].values
y_train_normal = df_training_under.iloc[:, 0:2].values
X_train_under = df_training_under.iloc[:, 2:].values
y_train_under = df_training_under.iloc[:, 0:2].values
X_train_over = df_training_over.iloc[:, 2:].values
y_train_over = df_training_over.iloc[:, 0:2].values
X_train_smote = df_training_smote.iloc[:, 2:].values
y_train_smote = df_training_smote.iloc[:, 0:2].values
X_test = df_test.iloc[:, 2:].values
y_test = df_test.iloc[:, 0:2].values

# Decision tree with imbalanced dataset
decision_tree(X_train_normal, y_train_normal, 'NORMAL', 'normal.dot', 'gini', 'best', None)
# Decision tree with under_sampling
decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'gini', 'best', None)
# Decision tree with over_sampling
decision_tree(X_train_over, y_train_over, 'OVER', 'over.dot', 'gini', 'best', None)
# Decision tree with Smote
decision_tree(X_train_smote, y_train_smote, 'SMOTE', 'smote.dot', 'gini', 'best', None)

