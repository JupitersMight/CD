import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn import cluster, datasets, mixture, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_swiss_roll
import time, warnings
from itertools import cycle, islice
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# FILES
DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\iris.csv'

# NAME OF CLASS ATTRIBUTE
NAME_OF_CLASS_ATTRIBUTE = 'class'

# Load dataset
df_iris = pd.read_csv(DATASET_FILE, delimiter=',')

# Take out class attribute from table
df_iris_without_class = df_iris.iloc[:, df_iris.columns != NAME_OF_CLASS_ATTRIBUTE].values
df_iris_class = df_iris.iloc[:, df_iris.columns == NAME_OF_CLASS_ATTRIBUTE].values

# Create and fit scaler
scaler = StandardScaler()

# Fit the scaler
scaler.fit(df_iris_without_class)

# Tranform the data
df_iris_scaled = scaler.transform(df_iris_without_class)
file = open('df_iris_scaled', 'w')
file.write(str(df_iris_scaled))
file.close()

# Plot heatmap with dendongrams

# Normalized
sns.clustermap(df_iris_scaled)
# Save figure
plt.savefig('Normalized.png', dpi=100)
# Standarized
sns.clustermap(df_iris_without_class, standard_scale=1)
# Save figure
plt.savefig('Standarized.png', dpi=100)

# Display it
plt.show()

# Pairwise similarity matrices using euclidean distance
euclidean = pairwise_distances(df_iris_scaled, metric='euclidean')
file = open('pairwise_euclidean', 'w')
file.write(str(euclidean))
file.close()

# Hierarchical and agglomerative clustering

# Create and fit all the clusters with all the parameters
cluster_2_min_max = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(df_iris_scaled)
cluster_2_complete = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(df_iris_scaled)
cluster_2_average = AgglomerativeClustering(n_clusters=2, linkage='average').fit_predict(df_iris_scaled)
cluster_2_ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(df_iris_scaled)

cluster_3_min_max = AgglomerativeClustering(n_clusters=3, linkage='single').fit_predict(df_iris_scaled)
cluster_3_complete = AgglomerativeClustering(n_clusters=3, linkage='complete').fit_predict(df_iris_scaled)
cluster_3_average = AgglomerativeClustering(n_clusters=3, linkage='average').fit_predict(df_iris_scaled)
cluster_3_ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(df_iris_scaled)

# Draw the clusters
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_2_min_max, cmap='rainbow')
plt.savefig('cluster_2_min_max.png', dpi=100)
plt.show()
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_2_complete, cmap='rainbow')
plt.savefig('cluster_2_complete.png', dpi=100)
plt.show()
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_2_average, cmap='rainbow')
plt.savefig('cluster_2_average.png', dpi=100)
plt.show()
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_2_ward, cmap='rainbow')
plt.savefig('cluster_2_ward.png', dpi=100)
plt.show()

plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_3_min_max, cmap='rainbow')
plt.savefig('cluster_3_min_max.png', dpi=100)
plt.show()
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_3_complete, cmap='rainbow')
plt.savefig('cluster_3_complete.png', dpi=100)
plt.show()
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_3_average, cmap='rainbow')
plt.savefig('cluster_3_average.png', dpi=100)
plt.show()
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=cluster_3_ward, cmap='rainbow')
plt.savefig('cluster_3_ward.png', dpi=100)
plt.show()

# k-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_iris_scaled)
y_kmeans = kmeans.predict(df_iris_scaled)

plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.savefig('kMeans.png', dpi=100)
plt.show()

# DBSCAN

dbs = DBSCAN(eps=3, min_samples=2).fit(df_iris_scaled)
y_dbs = dbs.fit_predict(df_iris_scaled)

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

    xy = df_iris_scaled[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = df_iris_scaled[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('DBSCAN.png', dpi=100)
plt.show()

# Gaussian Mixture

gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit_predict(df_iris_scaled)
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=gmm, s=40, cmap='viridis')
plt.savefig('GaussianMixture.png', dpi=100)
plt.show()

# Birch

birch = cluster.Birch(n_clusters=3).fit_predict(df_iris_scaled)
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=birch, s=40, cmap='viridis')
plt.savefig('birch.png', dpi=100)
plt.show()

# Spectral

spectral = cluster.SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors").fit_predict(df_iris_scaled)
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=spectral, s=40, cmap='viridis')
plt.savefig('spectral.png', dpi=100)
plt.show()

# Affinaty

affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200).fit_predict(df_iris_scaled)
plt.scatter(df_iris_scaled[:, 0], df_iris_scaled[:, 1], c=affinity_propagation, s=40, cmap='viridis')
plt.savefig('affinity.png', dpi=100)
plt.show()

# Evaluation with silhouette coefficient

# Aglomerative data
silhoute_coefficient_cluster_2_min_max = metrics.silhouette_score(df_iris_scaled, cluster_2_min_max, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_2_complete = metrics.silhouette_score(df_iris_scaled, cluster_2_complete, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_2_average = metrics.silhouette_score(df_iris_scaled, cluster_2_average, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_2_ward = metrics.silhouette_score(df_iris_scaled, cluster_2_ward, metric='euclidean', sample_size=50)

silhoute_coefficient_cluster_3_min_max= metrics.silhouette_score(df_iris_scaled, cluster_3_min_max, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_3_complete = metrics.silhouette_score(df_iris_scaled, cluster_3_complete, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_3_average = metrics.silhouette_score(df_iris_scaled, cluster_3_average, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_3_ward = metrics.silhouette_score(df_iris_scaled, cluster_3_ward, metric='euclidean', sample_size=50)

# Affinaty
silhoute_coefficient_affinity_propagation = metrics.silhouette_score(df_iris_scaled, affinity_propagation, metric='euclidean', sample_size=50)

# Spectral
silhoute_coefficient_spectral = metrics.silhouette_score(df_iris_scaled, spectral, metric='euclidean', sample_size=50)

# Birch
silhoute_coefficient_birch = metrics.silhouette_score(df_iris_scaled, birch, metric='euclidean', sample_size=50)

# Gaussian
silhoute_coefficient_gmm = metrics.silhouette_score(df_iris_scaled, gmm, metric='euclidean', sample_size=50)

# K-means
silhoute_coefficient_kmeans = metrics.silhouette_score(df_iris_scaled, y_kmeans, metric='euclidean', sample_size=50)

print('silhouette coefficient')
print()
print('Aglomerative:')
print()
print('cluster_2_min_max : ' + str(silhoute_coefficient_cluster_2_min_max))
print('cluster_2_complete : ' + str(silhoute_coefficient_cluster_2_complete))
print('cluster_2_average : ' + str(silhoute_coefficient_cluster_2_average))
print('cluster_2_ward : ' + str(silhoute_coefficient_cluster_2_ward))
print()
print('cluster_3_min_max : ' + str(silhoute_coefficient_cluster_3_min_max))
print('cluster_3_complete : ' + str(silhoute_coefficient_cluster_3_complete))
print('cluster_3_average : ' + str(silhoute_coefficient_cluster_3_average))
print('cluster_3_ward : ' + str(silhoute_coefficient_cluster_3_ward))
print()
print('K-Means : ' + str(silhoute_coefficient_kmeans))
print('Gaussian : ' + str(silhoute_coefficient_gmm))
print('Birch : ' + str(silhoute_coefficient_birch))
print('Spectral : ' + str(silhoute_coefficient_spectral))
print('Affinaty : ' + str(silhoute_coefficient_affinity_propagation))


# Evaluation with adjusted Rand index

