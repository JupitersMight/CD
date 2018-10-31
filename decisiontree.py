import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split

# Change to fit the index and name of the class attribute
INDEX_OF_CLASS_ATTRIBUTE = 4
NAME_OF_CLASS_ATTRIBUTE = 'class'

# If the dataset is not divided into training and testing datasets

NAME_OF_DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\iris.csv'
df = pd.read_csv(NAME_OF_DATASET_FILE, delimiter=',', na_values=['na'])

X = df.iloc[:, df.columns != NAME_OF_CLASS_ATTRIBUTE].values
Y = df.iloc[:, INDEX_OF_CLASS_ATTRIBUTE]

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, stratify=Y)

# If dataset is already divided into training and testing datasets

# NAME_OF_TRAINING_DATASET = ''
# NAME_OF_TESTING_DATASET= ''
# df_training = pd.read_csv(NAME_OF_TRAINING_DATASET, delimiter=',', na_values=['na'])
# df_testing = pd.read_csv(NAME_OF_TESTING_DATASET, delimiter=',', na_values=['na'])

# X_train = df_training.iloc[:, df_training.columns != NAME_OF_CLASS_ATTRIBUTE].values
# y_train = df_training.iloc[:, INDEX_OF_CLASS_ATTRIBUTE].values
# X_test = df_testing.iloc[:, df_training.columns != NAME_OF_CLASS_ATTRIBUTE].values
# y_test = df_testing.iloc[:, INDEX_OF_CLASS_ATTRIBUTE].values


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(y_test, predict)
print(accuracy)
conf_m = confusion_matrix(y_test, predict, pd.unique(Y))
print(conf_m)