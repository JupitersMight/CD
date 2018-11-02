import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_swiss_roll
import time, warnings
from itertools import cycle, islice
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.cluster import AgglomerativeClustering


# FILES
DATASET_FILE = 'C:\\Users\\Leona\\PycharmProjects\\LABS\\iris.csv'

# NAME OF CLASS ATTRIBUTE
NAME_OF_CLASS_ATTRIBUTE = 'class'

# Load dataset
df_iris = pd.read_csv(DATASET_FILE, delimiter=',')

# Take out class attribute from table
df_iris_without_class = df_iris.iloc[:, df_iris.columns != NAME_OF_CLASS_ATTRIBUTE].values

# Create and fit scaler
scaler = StandardScaler()

# Fit the scaler
scaler.fit(df_iris_without_class)

# Tranform the data
df_iris_scaled = scaler.transform(df_iris_without_class)

# Data post transformation
print(df_iris_scaled)

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
euclidean = pairwise_distances(df_iris_without_class, metric='euclidean')
file = open('pairwise_euclidean', 'w')
file.write(str(euclidean))
file.close()

# Hierarchical and agglomerative clustering

# Create all the clusters with all the parameters
cluster_2_min_max = AgglomerativeClustering(n_clusters=2, linkage='single')
cluster_2_complete = AgglomerativeClustering(n_clusters=2, linkage='complete')
cluster_2_average = AgglomerativeClustering(n_clusters=2, linkage='average')
cluster_2_ward = AgglomerativeClustering(n_clusters=2, linkage='ward')

cluster_3_min_max = AgglomerativeClustering(n_clusters=3, linkage='single')
cluster_3_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
cluster_3_average = AgglomerativeClustering(n_clusters=3, linkage='average')
cluster_3_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Fit the cluster with the data
cluster_2_min_max.fit(df_iris_without_class)
cluster_2_complete.fit(df_iris_without_class)
cluster_2_average.fit(df_iris_without_class)
cluster_2_ward.fit(df_iris_without_class)

cluster_3_min_max.fit(df_iris_without_class)
cluster_3_complete.fit(df_iris_without_class)
cluster_3_average.fit(df_iris_without_class)
cluster_3_ward.fit(df_iris_without_class)

# Draw the clusters
plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_2_min_max.labels_, cmap='rainbow')
plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_2_complete.labels_, cmap='rainbow')
plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_2_average.labels_, cmap='rainbow')
plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_2_ward.labels_, cmap='rainbow')

plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_3_min_max.labels_, cmap='rainbow')
plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_3_complete.labels_, cmap='rainbow')
plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_3_average.labels_, cmap='rainbow')
plt.scatter(df_iris_without_class[:, 0], df_iris_without_class[:, 1], c=cluster_3_ward.labels_, cmap='rainbow')

