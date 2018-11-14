import numpy as np
import pandas as pd
import itertools
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

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
            temp = pd.DataFrame(temp, columns=[(att + "_" + str(i)) for i in df[att].value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            pdf = pd.concat([pdf, temp], axis=1)
    return pdf

def plot_confusion_matrix(cnf_matrix, classesNames, normalize=False, cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if normalize:
        soma = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / soma
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix, without normalization'

    print(cm)

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classesNames))
    plt.xticks(tick_marks, classesNames, rotation=45)
    plt.yticks(tick_marks, classesNames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def Knn_k (data_training, data_test, n_neighbors):
    if data_training == data_test:
        test = pd.read_csv(data_test)
        test = preprocessData(test)
        tsX = test.iloc[:, 0:len(test.columns)-1]
        tsY = test.iloc[:, len(test.columns)-1]
    else:
        test = pd.read_csv(data_test)
        test = preprocessData(test)
        tsX = test.iloc[:, 2:(len(test.columns)-1)].values
        tsY = test.iloc[:, 0].values
    neighbors = list(range(1, n_neighbors+1, 25))
    possib_neighbors = []
    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        scores = cross_val_score(knn, tsX, tsY, cv=10, scoring='accuracy')
        possib_neighbors.append(scores.mean())
    # determining best k
    MSE = [1 - x for x in possib_neighbors]
    optimal_k = neighbors[MSE.index(min(MSE))]
    print("The optimal number of neighbors is %d" % optimal_k)
    # plot misclassification error vs k
    plt.plot(neighbors, MSE)
    plt.title('The optimal number of neighbors', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()
    return optimal_k









def Knn(data_training, data_test, optimal_k):
    if data_training == data_test:
        data = pd.read_csv(data_training)
        X = data.iloc[:, 0:len(data.columns)-1]
        X = preprocessData(X)
        y = data.iloc[:, len(data.columns)-1]
        trX, tsX, trY, tsY = train_test_split(X, y, train_size=0.6, stratify=y)
    else:
        training = pd.read_csv(data_training)
        training = preprocessData(training)
        trX = training.iloc[:, 2:(len(training.columns)-1)].values
        trY = training.iloc[:, 0].values
        test = pd.read_csv(data_test)
        test = preprocessData(test)
        tsX= test.iloc[:, 2:(len(training.columns)-1)].values
        tsY = test.iloc[:, 0].values
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    model1 = knn.fit(trX, trY)
    predY1 = model1.predict(tsX)
    cnf_matrixknn = confusion_matrix(tsY, predY1)
    labels = pd.unique(tsY)
    plot_confusion_matrix(cnf_matrixknn, labels)
    acknn = (accuracy_score(tsY, predY1))*100
    print("accuracy knn :", acknn)



def NaiveBayes (data_training, data_test):
    if data_training == data_test:
        data = pd.read_csv(data_training)
        X = data.iloc[:, 0:len(data.columns)-1]
        X = preprocessData(X)
        y = data.iloc[:, len(data.columns)-1]
        trX, tsX, trY, tsY = train_test_split(X, y, train_size=0.6, stratify=y)
    else:
        training_bayes = pd.read_csv(data_training)
        training_bayes = preprocessData(training_bayes)
        test_bayes = pd.read_csv(data_test)
        test_bayes = preprocessData(test_bayes)
        trX=training_bayes.iloc[:, 2:(len(training_bayes.columns)-1)].values
        trY=training_bayes.iloc[:, 0].values
        tsX = test_bayes.iloc[:, 2:(len(test_bayes.columns)-1)].values
        tsY = test_bayes.iloc[:, 0].values
    #training_bayes['ab_000'] = training_bayes['ab_000'].astype(float)
    #test_bayes['ab_000'] = test_bayes['ab_000'].astype(float)

    gnb = GaussianNB()
    bnb= BernoulliNB()
    #trX = normalize(trX)

    #Gaussian Distribution
    model_gnb = gnb.fit(trX, trY)
    predY_gnb = model_gnb.predict(tsX)
    acbayes_gnb = accuracy_score(tsY, predY_gnb)
    print("accuracy naive bayes Gaussian:", acbayes_gnb)
    cnf_matrix_gnb = confusion_matrix(tsY, predY_gnb)
    labels = pd.unique(tsY)
    plot_confusion_matrix(cnf_matrix_gnb, labels)

    # Bernoulli Distribution
    model_bnb = bnb.fit(trX, trY)
    predY_bnb = model_bnb.predict(tsX)
    acbayes_bnb = accuracy_score(tsY, predY_bnb)
    print("accuracy naive bayes Bernoulli:", acbayes_bnb)
    cnf_matrix_bnb = confusion_matrix(tsY, predY_gnb)
    labels = pd.unique(tsY)
    plot_confusion_matrix(cnf_matrix_bnb, labels)

    #cross validation
    scores_cv = list(range(2, 10))
    scores_cv_bnb =[]
    scores_cv_gnb =[]
    for i in scores_cv:
        scores_bnb = cross_val_score(bnb, tsX, tsY, cv=i, scoring='accuracy') #training or test
        scores_gnb = cross_val_score(gnb, tsX, tsY, cv=i, scoring='accuracy')
        scores_cv_bnb.append(scores_bnb.mean()*100)
        scores_cv_gnb.append(scores_gnb.mean()*100)

    plt.plot(scores_cv, scores_cv_bnb)
    plt.title('Naive Bayes Bernoulli', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Cross validation value')
    plt.show()
    plt.plot(scores_cv, scores_cv_gnb)
    plt.title('Naive Bayes Gaussian', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Cross validation value')
    plt.show()
    print( "accuracy Bernoulli cross validation (cv=",scores_cv_bnb.index(max(scores_cv_bnb))+2,")=",(max(scores_cv_bnb)))
    print( "accuracy Gausssin cross validation (cv=",scores_cv_gnb.index(max(scores_cv_gnb))+2,")=",(max(scores_cv_gnb)))





source1 ="aps_training_without_outliers.csv"
source2 ="aps_test_average.csv"
source3 = "smote_over_sampling_col_without_outliers.csv"




neighbors = 100
optimal_k = Knn_k(source3, source3, neighbors)
print(optimal_k)
Knn (source3, source3, optimal_k)
NaiveBayes(source3, source3)






