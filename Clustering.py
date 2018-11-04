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
from sklearn.preprocessing import LabelEncoder


# FILES
DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\iris.csv'

# NAME OF CLASS ATTRIBUTE
NAME_OF_CLASS_ATTRIBUTE = 'class'

# Load dataset
df = pd.read_csv(DATASET_FILE, delimiter=',')

# Take out class attribute from table
df_without_class = df.iloc[:, df.columns != NAME_OF_CLASS_ATTRIBUTE].values
df_class = df.iloc[:, df.columns == NAME_OF_CLASS_ATTRIBUTE].values

# Create and fit scaler and labeler
scaler = StandardScaler()
label_enconder = LabelEncoder()

# Fit and tranform the data
df_without_class_scaled = scaler.fit_transform(df_without_class)
file = open('df_iris_scaled', 'w')
file.write(str(df_without_class_scaled))
file.close()

df_class_scaled = label_enconder.fit_transform(df_class.ravel())
file = open('df_iris_class_scaled', 'w')
file.write(str(df_class_scaled))
file.close()

# Plot heatmap with dendongrams

# Normalized
sns.clustermap(df_without_class_scaled)
# Save figure
plt.savefig('Normalized.png', dpi=100)
# Standarized
sns.clustermap(df_without_class, standard_scale=1)
# Save figure
plt.savefig('Standarized.png', dpi=100)

# Display it
plt.show()

# Pairwise similarity matrices using euclidean distance
euclidean = pairwise_distances(df_without_class_scaled, metric='euclidean')
file = open('pairwise_euclidean', 'w')
file.write(str(euclidean))
file.close()

# Hierarchical and agglomerative clustering

# Create and fit all the clusters with all the parameters
cluster_2_min_max = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(df_without_class_scaled)
cluster_2_complete = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(df_without_class_scaled)
cluster_2_average = AgglomerativeClustering(n_clusters=2, linkage='average').fit_predict(df_without_class_scaled)
cluster_2_ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(df_without_class_scaled)

cluster_3_min_max = AgglomerativeClustering(n_clusters=3, linkage='single').fit_predict(df_without_class_scaled)
cluster_3_complete = AgglomerativeClustering(n_clusters=3, linkage='complete').fit_predict(df_without_class_scaled)
cluster_3_average = AgglomerativeClustering(n_clusters=3, linkage='average').fit_predict(df_without_class_scaled)
cluster_3_ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(df_without_class_scaled)

# Draw the clusters
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_2_min_max, cmap='rainbow')
plt.savefig('cluster_2_min_max.png', dpi=100)
plt.show()
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_2_complete, cmap='rainbow')
plt.savefig('cluster_2_complete.png', dpi=100)
plt.show()
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_2_average, cmap='rainbow')
plt.savefig('cluster_2_average.png', dpi=100)
plt.show()
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_2_ward, cmap='rainbow')
plt.savefig('cluster_2_ward.png', dpi=100)
plt.show()

plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_3_min_max, cmap='rainbow')
plt.savefig('cluster_3_min_max.png', dpi=100)
plt.show()
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_3_complete, cmap='rainbow')
plt.savefig('cluster_3_complete.png', dpi=100)
plt.show()
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_3_average, cmap='rainbow')
plt.savefig('cluster_3_average.png', dpi=100)
plt.show()
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=cluster_3_ward, cmap='rainbow')
plt.savefig('cluster_3_ward.png', dpi=100)
plt.show()

# k-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_without_class_scaled)
y_kmeans = kmeans.predict(df_without_class_scaled)

plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.savefig('kMeans.png', dpi=100)
plt.show()

# DBSCAN

dbs = DBSCAN(eps=3, min_samples=2).fit(df_without_class_scaled)
y_dbs = dbs.fit_predict(df_without_class_scaled)

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

    xy = df_without_class_scaled[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = df_without_class_scaled[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('DBSCAN.png', dpi=100)
plt.show()

# Gaussian Mixture

gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit_predict(df_without_class_scaled)
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=gmm, s=40, cmap='viridis')
plt.savefig('GaussianMixture.png', dpi=100)
plt.show()

# Birch

birch = cluster.Birch(n_clusters=3).fit_predict(df_without_class_scaled)
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=birch, s=40, cmap='viridis')
plt.savefig('birch.png', dpi=100)
plt.show()

# Spectral

spectral = cluster.SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors").fit_predict(df_without_class_scaled)
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=spectral, s=40, cmap='viridis')
plt.savefig('spectral.png', dpi=100)
plt.show()

# Affinaty

affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200).fit_predict(df_without_class_scaled)
plt.scatter(df_without_class_scaled[:, 0], df_without_class_scaled[:, 1], c=affinity_propagation, s=40, cmap='viridis')
plt.savefig('affinity.png', dpi=100)
plt.show()

# Evaluation using silhouette coefficient

# Aglomerative data
silhoute_coefficient_cluster_2_min_max = metrics.silhouette_score(df_without_class_scaled, cluster_2_min_max, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_2_complete = metrics.silhouette_score(df_without_class_scaled, cluster_2_complete, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_2_average = metrics.silhouette_score(df_without_class_scaled, cluster_2_average, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_2_ward = metrics.silhouette_score(df_without_class_scaled, cluster_2_ward, metric='euclidean', sample_size=50)

silhoute_coefficient_cluster_3_min_max= metrics.silhouette_score(df_without_class_scaled, cluster_3_min_max, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_3_complete = metrics.silhouette_score(df_without_class_scaled, cluster_3_complete, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_3_average = metrics.silhouette_score(df_without_class_scaled, cluster_3_average, metric='euclidean', sample_size=50)
silhoute_coefficient_cluster_3_ward = metrics.silhouette_score(df_without_class_scaled, cluster_3_ward, metric='euclidean', sample_size=50)

# Affinaty
silhoute_coefficient_affinity_propagation = metrics.silhouette_score(df_without_class_scaled, affinity_propagation, metric='euclidean', sample_size=50)

# Spectral
silhoute_coefficient_spectral = metrics.silhouette_score(df_without_class_scaled, spectral, metric='euclidean', sample_size=50)

# Birch
silhoute_coefficient_birch = metrics.silhouette_score(df_without_class_scaled, birch, metric='euclidean', sample_size=50)

# Gaussian
silhoute_coefficient_gmm = metrics.silhouette_score(df_without_class_scaled, gmm, metric='euclidean', sample_size=50)

# K-means
silhoute_coefficient_kmeans = metrics.silhouette_score(df_without_class_scaled, y_kmeans, metric='euclidean', sample_size=50)

print('Silhouette coefficient - Higher scores mean relates to better defined clusters and has 2 scores')
print('A) The mean distance between a sample and all other points in the same class')
print('B) The mean distance between a sample and all other points in the next nearest cluster')
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
print()


# Evaluation using adjusted Rand index

# Aglomerative data
adjusted_rand_score_cluster_2_min_max = metrics.adjusted_rand_score(df_class_scaled, cluster_2_min_max)
adjusted_rand_score_cluster_2_complete = metrics.adjusted_rand_score(df_class_scaled, cluster_2_complete)
adjusted_rand_score_cluster_2_average = metrics.adjusted_rand_score(df_class_scaled, cluster_2_average)
adjusted_rand_score_cluster_2_ward = metrics.adjusted_rand_score(df_class_scaled, cluster_2_ward)

adjusted_rand_score_cluster_3_min_max= metrics.adjusted_rand_score(df_class_scaled, cluster_3_min_max)
adjusted_rand_score_cluster_3_complete = metrics.adjusted_rand_score(df_class_scaled, cluster_3_complete)
adjusted_rand_score_cluster_3_average = metrics.adjusted_rand_score(df_class_scaled, cluster_3_average)
adjusted_rand_score_cluster_3_ward = metrics.adjusted_rand_score(df_class_scaled, cluster_3_ward)

# Affinaty
adjusted_rand_score_affinity_propagation = metrics.adjusted_rand_score(df_class_scaled, affinity_propagation)

# Spectral
adjusted_rand_score_spectral = metrics.adjusted_rand_score(df_class_scaled, spectral)

# Birch
adjusted_rand_score_birch = metrics.adjusted_rand_score(df_class_scaled, birch)

# Gaussian
adjusted_rand_score_gmm = metrics.adjusted_rand_score(df_class_scaled, gmm)

# K-means
adjusted_rand_score_kmeans = metrics.adjusted_rand_score(df_class_scaled, y_kmeans)

print('Adjusted Rand Score - The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings')
print()
print('Aglomerative:')
print()
print('cluster_2_min_max : ' + str(adjusted_rand_score_cluster_2_min_max))
print('cluster_2_complete : ' + str(adjusted_rand_score_cluster_2_complete))
print('cluster_2_average : ' + str(adjusted_rand_score_cluster_2_average))
print('cluster_2_ward : ' + str(adjusted_rand_score_cluster_2_ward))
print()
print('cluster_3_min_max : ' + str(adjusted_rand_score_cluster_3_min_max))
print('cluster_3_complete : ' + str(adjusted_rand_score_cluster_3_complete))
print('cluster_3_average : ' + str(adjusted_rand_score_cluster_3_average))
print('cluster_3_ward : ' + str(adjusted_rand_score_cluster_3_ward))
print()
print('K-Means : ' + str(adjusted_rand_score_kmeans))
print('Gaussian : ' + str(adjusted_rand_score_gmm))
print('Birch : ' + str(adjusted_rand_score_birch))
print('Spectral : ' + str(adjusted_rand_score_spectral))
print('Affinaty : ' + str(adjusted_rand_score_affinity_propagation))
print()

# Evaluation using mutual information based scores

# Aglomerative data
mutual_info_score_cluster_2_min_max = metrics.mutual_info_score(df_class_scaled, cluster_2_min_max)
mutual_info_score_cluster_2_complete = metrics.mutual_info_score(df_class_scaled, cluster_2_complete)
mutual_info_score_cluster_2_average = metrics.mutual_info_score(df_class_scaled, cluster_2_average)
mutual_info_score_cluster_2_ward = metrics.mutual_info_score(df_class_scaled, cluster_2_ward)

mutual_info_score_cluster_3_min_max= metrics.mutual_info_score(df_class_scaled, cluster_3_min_max)
mutual_info_score_cluster_3_complete = metrics.mutual_info_score(df_class_scaled, cluster_3_complete)
mutual_info_score_cluster_3_average = metrics.mutual_info_score(df_class_scaled, cluster_3_average)
mutual_info_score_cluster_3_ward = metrics.mutual_info_score(df_class_scaled, cluster_3_ward)

# Affinaty
mutual_info_score_affinity_propagation = metrics.mutual_info_score(df_class_scaled, affinity_propagation)

# Spectral
mutual_info_score_spectral = metrics.mutual_info_score(df_class_scaled, spectral)

# Birch
mutual_info_score_birch = metrics.mutual_info_score(df_class_scaled, birch)

# Gaussian
mutual_info_score_gmm = metrics.mutual_info_score(df_class_scaled, gmm)

# K-means
mutual_info_score_kmeans = metrics.mutual_info_score(df_class_scaled, y_kmeans)

print('Mutual Information Based Scores - The Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations')
print()
print('Aglomerative:')
print()
print('cluster_2_min_max : ' + str(mutual_info_score_cluster_2_min_max))
print('cluster_2_complete : ' + str(mutual_info_score_cluster_2_complete))
print('cluster_2_average : ' + str(mutual_info_score_cluster_2_average))
print('cluster_2_ward : ' + str(mutual_info_score_cluster_2_ward))
print()
print('cluster_3_min_max : ' + str(mutual_info_score_cluster_3_min_max))
print('cluster_3_complete : ' + str(mutual_info_score_cluster_3_complete))
print('cluster_3_average : ' + str(mutual_info_score_cluster_3_average))
print('cluster_3_ward : ' + str(mutual_info_score_cluster_3_ward))
print()
print('K-Means : ' + str(mutual_info_score_kmeans))
print('Gaussian : ' + str(mutual_info_score_gmm))
print('Birch : ' + str(mutual_info_score_birch))
print('Spectral : ' + str(mutual_info_score_spectral))
print('Affinaty : ' + str(mutual_info_score_affinity_propagation))
print()


# Evaluation using homogeneity, completeness and V-measure

# Aglomerative data
homogeneity_completeness_v_measure_cluster_2_min_max = metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_2_min_max)
homogeneity_completeness_v_measure_cluster_2_complete = metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_2_complete)
homogeneity_completeness_v_measure_cluster_2_average = metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_2_average)
homogeneity_completeness_v_measure_cluster_2_ward = metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_2_ward)

homogeneity_completeness_v_measure_cluster_3_min_max= metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_3_min_max)
homogeneity_completeness_v_measure_cluster_3_complete = metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_3_complete)
homogeneity_completeness_v_measure_cluster_3_average = metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_3_average)
homogeneity_completeness_v_measure_cluster_3_ward = metrics.homogeneity_completeness_v_measure(df_class_scaled, cluster_3_ward)

# Affinaty
homogeneity_completeness_v_measure_affinity_propagation = metrics.homogeneity_completeness_v_measure(df_class_scaled, affinity_propagation)

# Spectral
homogeneity_completeness_v_measure_spectral = metrics.homogeneity_completeness_v_measure(df_class_scaled, spectral)

# Birch
homogeneity_completeness_v_measure_birch = metrics.homogeneity_completeness_v_measure(df_class_scaled, birch)

# Gaussian
homogeneity_completeness_v_measure_gmm = metrics.homogeneity_completeness_v_measure(df_class_scaled, gmm)

# K-means
homogeneity_completeness_v_measure_kmeans = metrics.homogeneity_completeness_v_measure(df_class_scaled, y_kmeans)

print('homogeneity, completeness and V-measure Scores')
print()
print('Homogeneity - A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class')
print('Completeness - A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.')
print('V-measure Scores - The V-measure is the harmonic mean between homogeneity and completeness')
print()
print('Aglomerative:')
print()
print('cluster_2_min_max : ' + str(homogeneity_completeness_v_measure_cluster_2_min_max))
print('cluster_2_complete : ' + str(homogeneity_completeness_v_measure_cluster_2_complete))
print('cluster_2_average : ' + str(homogeneity_completeness_v_measure_cluster_2_average))
print('cluster_2_ward : ' + str(homogeneity_completeness_v_measure_cluster_2_ward))
print()
print('cluster_3_min_max : ' + str(homogeneity_completeness_v_measure_cluster_3_min_max))
print('cluster_3_complete : ' + str(homogeneity_completeness_v_measure_cluster_3_complete))
print('cluster_3_average : ' + str(homogeneity_completeness_v_measure_cluster_3_average))
print('cluster_3_ward : ' + str(homogeneity_completeness_v_measure_cluster_3_ward))
print()
print('K-Means : ' + str(homogeneity_completeness_v_measure_kmeans))
print('Gaussian : ' + str(homogeneity_completeness_v_measure_gmm))
print('Birch : ' + str(homogeneity_completeness_v_measure_birch))
print('Spectral : ' + str(homogeneity_completeness_v_measure_spectral))
print('Affinaty : ' + str(homogeneity_completeness_v_measure_affinity_propagation))
print()
