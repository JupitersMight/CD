import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statistics
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt

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


def print_statistics(welcome, accuracy_testing, accuracy_training, deviation, conf_matrix, roc_accuracy_testing, roc_accuracy_training):
    print(welcome)
    print('decision tree was written to file')
    print()
    print('Accuracy for testing : ' + str(accuracy_testing))
    print()
    print('ROC_Accuracy for testing : ' + str(roc_accuracy_testing))
    print()
    print('Accuracy for training : ' + str(accuracy_training))
    print()
    print('ROC_Accuracy for training : ' + str(roc_accuracy_training))
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


def decision_tree(X_train, y_train, welcome, file, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, testing):
    # Create tree
    clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)

    # Fit tree
    clf.fit(X_train, y_train)

    # Prediction for the training and the testing data
    predict_testing = clf.predict(X_test)
    predict_training = clf.predict(X_train)

    # Accuracy measures using ROC
    false_positive_rate_test, true_positive_rate_test, thresholds_test = roc_curve(y_test.argmax(axis=1), predict_testing.argmax(axis=1))
    roc_accuracy_testing = auc(false_positive_rate_test, true_positive_rate_test)
    false_positive_rate_train, true_positive_rate_train, thresholds_train = roc_curve(y_train.argmax(axis=1), predict_training.argmax(axis=1))
    roc_accuracy_training = auc(false_positive_rate_train, true_positive_rate_train)
    if testing == 1:
        train_results.append(roc_accuracy_training*100)
        test_results.append(roc_accuracy_testing*100)

    # Normal Accuracy
    accuracy_testing = accuracy_score(y_test, predict_testing)
    accuracy_training = accuracy_score(y_train, predict_training)
    # if testing == 1:
    #     train_results.append(accuracy_training)
    #     test_results.append(accuracy_testing)

    # Confusion matrix
    conf_m = confusion_matrix(y_test.argmax(axis=1), predict_testing.argmax(axis=1), labels=None, sample_weight=None)

    # Tree built
    dotfile = open(file, 'w')
    tree.export_graphviz(clf, out_file=dotfile, feature_names=df_training_under.columns[2:])
    dotfile.close()

    # Print to console the statistics
    if testing == 0:
        print_statistics(welcome, accuracy_testing, accuracy_training, statistics.stdev(predict_testing.ravel()), conf_m, roc_accuracy_testing, roc_accuracy_training)


def find_max(train, test):
    idx_of_max = -1
    for idx, value in enumerate(train):
        if value == 100:
            continue
        if idx_of_max == -1:
            idx_of_max = idx
        else:
            if abs(train[idx_of_max] - test[idx_of_max]) > abs(value-test[idx]):
                idx_of_max = idx
    if idx_of_max == -1:
        for idx, value in enumerate(train):
            if abs(train[idx_of_max] - test[idx_of_max]) > abs(value-test[idx]):
                idx_of_max = idx
    return idx_of_max

# APS
# Load files

# df_training = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_training_average.csv')
# df_training = preprocessData(df_training)
df_training_under = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\under_sampling_aps_training.csv')
df_training_under = preprocessData(df_training_under)
# df_training_over = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\over_sampling_aps_training.csv')
# df_training_over = preprocessData(df_training_over)
# df_training_smote = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\under_sampling_aps_training.csv')
# df_training_smote = preprocessData(df_training_smote)
df_test = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_test_average.csv')
df_test = preprocessData(df_test)

# X_train_normal = df_training_under.iloc[:, 2:].values
# y_train_normal = df_training_under.iloc[:, 0:2].values
X_train_under = df_training_under.iloc[:, 2:].values
y_train_under = df_training_under.iloc[:, 0:2].values
# X_train_over = df_training_over.iloc[:, 2:].values
# y_train_over = df_training_over.iloc[:, 0:2].values
# X_train_smote = df_training_smote.iloc[:, 2:].values
# y_train_smote = df_training_smote.iloc[:, 0:2].values
X_test = df_test.iloc[:, 2:].values
y_test = df_test.iloc[:, 0:2].values

# Decision tree with imbalanced dataset
# decision_tree(X_train_normal, y_train_normal, 'NORMAL', 'normal.dot', 'gini', 'best', None)
# Decision tree with under_sampling
# decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'gini', 'best', None, 2, 0)
# Decision tree with over_sampling
# decision_tree(X_train_over, y_train_over, 'OVER', 'over.dot', 'gini', 'best', None)
# Decision tree with Smote
# decision_tree(X_train_smote, y_train_smote, 'SMOTE', 'smote.dot', 'gini', 'best', None)

GINI = []
ENTROPY = []

# Tunning max_depth

max_depths = np.linspace(1, 32, 32, endpoint=True)

# GINI
train_results = []
test_results = []
for max_depth in max_depths:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'gini', max_depth, 2, 1, None, 1)

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.savefig('max_depth_gini.png', dpi=100)
plt.show()
GINI.append(find_max(train_results, test_results))


# Entropy
train_results = []
test_results = []
for max_depth in max_depths:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'entropy', max_depth, 2, 1, None, 1)

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.savefig('max_depth_entropy.png', dpi=100)
plt.show()
ENTROPY.append(find_max(train_results, test_results))

# Tunning min_samples

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

# GINI
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'gini', None, min_samples_split, 1, None, 1)

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.savefig('min_samples_split_gini.png', dpi=100)
plt.show()
GINI.append(find_max(train_results, test_results))

# Entropy
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'entropy', None, min_samples_split, 1, None, 1)

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.savefig('min_samples_split_entropy.png', dpi=100)
plt.show()
ENTROPY.append(find_max(train_results, test_results))

# Tunning min_samples_leaf

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

# GINI
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'gini', None, 2, min_samples_leaf, None, 1)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.savefig('min_samples_leaf_gini.png', dpi=100)
plt.show()
GINI.append(find_max(train_results, test_results))

# Entropy
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'entropy', None, 2, min_samples_leaf, None, 1)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.savefig('min_samples_leaf_entropy.png', dpi=100)
plt.show()
ENTROPY.append(find_max(train_results, test_results))

# Tunning max_features

max_features = list(range(1, df_training_under.shape[1]-2))

# GINI
train_results = []
test_results = []
for max_feature in max_features:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'gini', None, 2, 1, max_feature, 1)

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.savefig('max_features_gini.png', dpi=100)
plt.show()
GINI.append(find_max(train_results, test_results))

# Entropy
train_results = []
test_results = []
for max_feature in max_features:
    decision_tree(X_train_under, y_train_under, 'UNDER', 'under.dot', 'entropy', None, 2, 1, max_feature, 1)

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.savefig('max_features_entropy.png', dpi=100)
plt.show()
ENTROPY.append(find_max(train_results, test_results))

print('GINI : '+str(GINI))
print('ENTROPY : '+str(ENTROPY))


decision_tree(X_train_under, y_train_under, 'GINI', 'gini.dot', 'gini', 1, 0.6, 0.5, 12, 0)
print()
decision_tree(X_train_under, y_train_under, 'ENTROPY', 'entropy.dot', 'entropy', 1, 0.6, 0.5, 107, 0)

