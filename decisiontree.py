import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
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
        #final_results.append(roc_accuracy_testing*100)
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
    print('Confusion matrix : ')
    print(conf_m)
    print()
    # Tree built
    if testing == 0:
        dotfile = open(file, 'w')
        tree.export_graphviz(clf, out_file=dotfile, feature_names=df_training_under.columns[2:])
        dotfile.close()
    # Print to console the statistics
    if testing == 0:
        print_statistics(welcome, accuracy_testing, accuracy_training, statistics.stdev(predict_testing.ravel()), conf_m, roc_accuracy_testing, roc_accuracy_training)


def draw_graphic(function, train_test, filename, parameter_description):
    line1, = plt.plot(function, train_test[0], 'b', label='Train AUC')
    line2, = plt.plot(function, train_test[1], 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel(parameter_description)
    plt.savefig(filename, dpi=100)
    plt.show()


def find_max(train, test):
    idx_of_max = [-1, 0]
    for idx, value in enumerate(train):
        if value == 100:
            continue
        if idx_of_max[0] == -1:
            idx_of_max[0] = idx
            idx_of_max[1] = test[idx]
        else:
            if value < test[idx]:
                if test[idx] > idx_of_max[1]:
                    idx_of_max[0] = idx
                    idx_of_max[1] = test[idx]
            if abs(train[idx_of_max[1]]-test[idx_of_max[1]]) > abs(value-test[idx]):
                idx_of_max[0] = idx
                idx_of_max[1] = value

    if idx_of_max == -1:
        for idx, value in enumerate(train):
            if abs(train[idx_of_max] - test[idx_of_max]) > abs(value-test[idx]):
                idx_of_max = idx
    return idx_of_max


def data_for_graphs(parameter_array, size_of_array):
    max_depth_sums_train = [0] * size_of_array
    max_depth_sums_test = [0] * size_of_array
    for array in parameter_array:
        for idx, train_value in enumerate(array[0]):
            max_depth_sums_train[idx] = max_depth_sums_train[idx] + train_value
        for idx, test_value in enumerate(array[1]):
            max_depth_sums_test[idx] = max_depth_sums_test[idx] + test_value

    for idx, value in enumerate(max_depth_sums_train):
        max_depth_sums_train[idx] = max_depth_sums_train[idx] / 30
    for idx, value in enumerate(max_depth_sums_test):
        max_depth_sums_test[idx] = max_depth_sums_test[idx] / 30

    return [max_depth_sums_train, max_depth_sums_test]


# APS
# Load files

# df_training = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_training_average.csv')
# df_training = preprocessData(df_training)
df_training_under = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\under_sampling_aps_training.csv')
df_training_under = preprocessData(df_training_under)
df_training_under['ab_000'] = df_training_under['ab_000'].astype(float)
# df_training_over = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\over_sampling_aps_training.csv')
# df_training_over = preprocessData(df_training_over)
# df_training_smote = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\under_sampling_aps_training.csv')
# df_training_smote = preprocessData(df_training_smote)
df_test = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_test_average.csv')
df_test = preprocessData(df_test)
df_test['ab_000'] = df_test['ab_000'].astype(float)

# X_train_normal = df_training_under.iloc[:, 2:].values
# y_train_normal = df_training_under.iloc[:, 0:2].values
X_train_under = df_training_under.iloc[:, 2:].values
X_train_under = normalize(X_train_under)
y_train_under = df_training_under.iloc[:, 0:2].values
# X_train_over = df_training_over.iloc[:, 2:].values
# y_train_over = df_training_over.iloc[:, 0:2].values
# X_train_smote = df_training_smote.iloc[:, 2:].values
# y_train_smote = df_training_smote.iloc[:, 0:2].values
X_test = df_test.iloc[:, 2:].values
X_test = normalize(X_test)
y_test = df_test.iloc[:, 0:2].values

GINI = []
ENTROPY = []

skf = StratifiedKFold(n_splits=30)

max_depth_splits_gini = []
samples_split_splits_gini = []
samples_leaf_splits_gini = []
features_splits_gini = []

max_depth_splits_entropy = []
samples_split_splits_entropy = []
samples_leaf_splits_entropy = []
features_splits_entropy = []

final_results = []

i = 1
for train_index, test_index in skf.split(X_train_under, y_train_under.argmax(axis=1)):

    X_train = X_train_under[train_index]
    y_train = y_train_under[train_index]

    # decision_tree(X_train, y_train, 'FINAL RESULT', 'final_result.dot', 'entropy', 5, 0.8, 0.3, 16, 0)

    # Tunning max_depth
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    # GINI
    train_results = []
    test_results = []
    for max_depth in max_depths:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'gini', max_depth, 2, 1, None, 1)
    max_depth_splits_gini.append([train_results, test_results])
    # Entropy
    train_results = []
    test_results = []
    for max_depth in max_depths:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'entropy', max_depth, 2, 1, None, 1)
    max_depth_splits_entropy.append([train_results, test_results])

    # Tunning min_samples
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    # GINI
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'gini', None, min_samples_split, 1, None, 1)
    samples_split_splits_gini.append([train_results, test_results])
    # Entropy
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'entropy', None, min_samples_split, 1, None, 1)
    samples_split_splits_entropy.append([train_results, test_results])

    # Tunning min_samples_leaf
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    # GINI
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'gini', None, 2, min_samples_leaf, None, 1)
    samples_leaf_splits_gini.append([train_results, test_results])
    # Entropy
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'entropy', None, 2, min_samples_leaf, None, 1)
    samples_leaf_splits_entropy.append([train_results, test_results])

    # Tunning max_features
    max_features = list(range(1, 85))
    # GINI
    train_results = []
    test_results = []
    for max_feature in max_features:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'gini', None, 2, 1, max_feature, 1)
    features_splits_gini.append([train_results, test_results])
    # Entropy
    train_results = []
    test_results = []
    for max_feature in max_features:
        decision_tree(X_train, y_train, 'UNDER', 'under.dot', 'entropy', None, 2, 1, max_feature, 1)
    features_splits_entropy.append([train_results, test_results])
    print(i)
    i += 1

# print('Average accuracy for AUC : '+str(np.mean(final_results)))
# print('Standard deviation : '+str(np.std(final_results)))

max_depth_gini = data_for_graphs(max_depth_splits_gini, 32)
samples_split_gini = data_for_graphs(samples_split_splits_gini, 10)
samples_leaf_gini = data_for_graphs(samples_leaf_splits_gini, 5)
features_gini = data_for_graphs(features_splits_gini, 85)

max_depth_entropy = data_for_graphs(max_depth_splits_entropy, 32)
samples_split_entropy = data_for_graphs(samples_split_splits_entropy, 10)
samples_leaf_entropy = data_for_graphs(samples_leaf_splits_entropy, 5)
features_entropy = data_for_graphs(features_splits_entropy, 85)

draw_graphic(np.linspace(1, 32, 32, endpoint=True), max_depth_gini, 'max_depth_gini.png', 'Tree depth')
draw_graphic(np.linspace(0.1, 1.0, 10, endpoint=True), samples_split_gini, 'samples_split_gini.png', 'Samples for split')
draw_graphic(np.linspace(0.1, 0.5, 5, endpoint=True), samples_leaf_gini, 'samples_leaf_gini.png', 'Samples for leaf')
draw_graphic(list(range(1, 86)), features_gini, 'features_gini.png', 'Number of Features')

draw_graphic(np.linspace(1, 32, 32, endpoint=True), max_depth_entropy, 'max_depth_entropy.png', 'Tree depth')
draw_graphic(np.linspace(0.1, 1.0, 10, endpoint=True), samples_split_entropy, 'samples_split_entropy.png', 'Samples for split')
draw_graphic(np.linspace(0.1, 0.5, 5, endpoint=True), samples_leaf_entropy, 'samples_leaf_entropy.png', 'Samples for leaf')
draw_graphic(list(range(1, 86)), features_entropy, 'features_entropy.png', 'Number of Features')
