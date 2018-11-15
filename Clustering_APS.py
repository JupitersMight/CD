import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

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


# FILES
DATASET_FILE_TEST = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\aps_testing_without_outliers.csv'

# NAME OF CLASS ATTRIBUTE
NAME_OF_CLASS_ATTRIBUTE = 'class'

# Load dataset
df_test = pd.read_csv(DATASET_FILE_TEST, delimiter=',')
df_test = preprocessData(df_test)


# Take out class attribute from table
pca = PCA(n_components=2)
df_without_class_test = df_test.iloc[:, 2:len(df_test.columns)-1].values
df_without_class_test = StandardScaler().fit_transform(df_without_class_test)
df_without_class_test = pca.fit_transform(df_without_class_test)
df_class_testing = df_test.iloc[:, 0].values

# Hierarchical and agglomerative clustering

cluster_complete = AgglomerativeClustering(n_clusters=5, linkage='complete').fit_predict(df_without_class_test)

plt.scatter(df_without_class_test[:, 0], df_without_class_test[:, 1], c=cluster_complete, cmap='rainbow')
plt.savefig('cluster_complete.png', dpi=100)
plt.show()

cluster_single = AgglomerativeClustering(n_clusters=5, linkage='single').fit_predict(df_without_class_test)

plt.scatter(df_without_class_test[:, 0], df_without_class_test[:, 1], c=cluster_single, cmap='rainbow')
plt.savefig('cluster_single.png', dpi=100)
plt.show()

cluster_ward = AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(df_without_class_test)

plt.scatter(df_without_class_test[:, 0], df_without_class_test[:, 1], c=cluster_ward, cmap='rainbow')
plt.savefig('cluster_ward.png', dpi=100)
plt.show()

cluster_average = AgglomerativeClustering(n_clusters=5, linkage='average').fit_predict(df_without_class_test)

plt.scatter(df_without_class_test[:, 0], df_without_class_test[:, 1], c=cluster_average, cmap='rainbow')
plt.savefig('cluster_average.png', dpi=100)
plt.show()

# k-Means

Sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_without_class_test)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('clusters')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal number of clusters for K-means')
plt.savefig('elbow.png', dpi=100)
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(df_without_class_test)
y_kmeans = kmeans.predict(df_without_class_test)

plt.scatter(df_without_class_test[:, 0], df_without_class_test[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.savefig('kMeans.png', dpi=100)
plt.show()


# DBSCAN

dbs = DBSCAN(eps=0.3, min_samples=5).fit(df_without_class_test)
y_dbs = dbs.fit_predict(df_without_class_test)

labels = dbs.labels_
core_samples_mask = np.zeros_like(dbs.labels_, dtype=bool)
core_samples_mask[dbs.core_sample_indices_] = True
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = df_without_class_test[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = df_without_class_test[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('DBSCAN.png', dpi=100)
plt.show()

# Evaluation using silhouette coefficient

# # Aglomerative data
silhoute_coefficient_cluster_complete = metrics.silhouette_score(df_without_class_test, cluster_complete, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_ward = metrics.silhouette_score(df_without_class_test, cluster_ward, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_average = metrics.silhouette_score(df_without_class_test, cluster_average, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_single = metrics.silhouette_score(df_without_class_test, cluster_single, metric='euclidean', sample_size=50)

# K-means
silhoute_coefficient_kmeans = metrics.silhouette_score(df_without_class_test, y_kmeans, metric='euclidean', sample_size=50)

#DBSCAN
silhoute_coefficient_dbs = metrics.silhouette_score(df_without_class_test, y_dbs, metric='euclidean', sample_size=50)

print('Silhouette coefficient - Higher scores mean relates to better defined clusters and has 2 scores')
print('A) The mean distance between a sample and all other points in the same class')
print('B) The mean distance between a sample and all other points in the next nearest cluster')
print()
print('Aglomerative:')
print()
print('cluster_complete : ' + str(silhoute_coefficient_cluster_complete))
print('cluster_ward : ' + str(silhoute_coefficient_cluster_ward))
print('cluster_average : ' + str(silhoute_coefficient_cluster_average))
print('cluster_single : ' + str(silhoute_coefficient_cluster_single))
print()
print('K-Means : ' + str(silhoute_coefficient_kmeans))
print()

