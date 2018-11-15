import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing

def normalize_values(column):
    min_value = 0
    max_value = 0
    first_value = 1
    for value in column:
        if value == 'na':
            continue
        if first_value == 1:
            min_value = float(value)
            max_value = float(value)
            first_value = 0
        if min_value > float(value):
            min_value = float(value)
        if max_value < float(value):
            max_value = float(value)
    result = []
    for value in column:
        if value == 'na':
            continue
        if max_value-min_value > 0:
            result.append((float(value) - min_value) / (max_value - min_value))

    return result

def draw_boxplot(dataset, filename):
    results = []
    for column_name in dataset.columns:
        if column_name == 'class':
            continue
        results.append(normalize_values(dataset[column_name]))

    plt.boxplot(results)
    plt.savefig(filename, dpi=100)
    plt.show()

def draw_heatmap_of_features(dataset, X, y, filename):
    selector = SelectKBest(chi2, k=4)
    X_new = selector.fit_transform(X, y)
    # Get idxs of columns to keep
    idxs_selected = selector.get_support(indices=True)
    columns = []
    for idx in idxs_selected:
        columns.append(dataset.columns[idx])
    X_new = pd.DataFrame(data=X_new, columns=columns)

    corr = X_new.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(X_new.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(X_new.columns)
    ax.set_yticklabels(X_new.columns)
    plt.savefig(filename, dpi=100)
    plt.show()

def draw_histogram(column, filename):
    sns.distplot(column)
    plt.savefig(filename, dpi=100)
    plt.legend()
    plt.show()


green = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\green.csv', delimiter=',')
hinselmann = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\hinselmann.csv', delimiter=',')
schiller = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\col\\schiller.csv', delimiter=',')
df_col = pd.concat([green, hinselmann, schiller])

draw_boxplot(green, 'green_col_boxplots.png')
draw_boxplot(hinselmann, 'boxplot_hinselmann_col_boxplots.png')
draw_boxplot(schiller, 'schiller_col_boxplots.png')
draw_boxplot(df_col, 'combined_col_boxplots.png')


X = df_col.iloc[:, 0:len(df_col.columns)-7].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)
y = df_col.iloc[:, len(df_col.columns)-1].values
draw_heatmap_of_features(df_col, X, y, 'heatmap_correlation_col.png')

draw_histogram(X[0], 'rgb_cervix_b_mean.png')
draw_histogram(X[1], 'rgb_cervix_b_mean_plus_std.png')
draw_histogram(X[2], 'hsv_cervix_v_mean.png')
draw_histogram(X[3], 'dist_to_center_cervix.png')

# APS

#TRAINING

df_aps = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstraining.csv')

result = []
i = 0
for column_name in df_aps.columns:
    if column_name == 'class':
        continue
    if column_name == 'bn_000':
        result.append(normalize_values(df_aps[column_name]))

draw_histogram(result, 'aps_training_histograma.png')

plt.boxplot(result)
plt.savefig('aps_training_boxplot.png', dpi=100)
plt.show()

data = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\median_replacement_aps_training.csv')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
draw_heatmap_of_features(data, X, y, 'heatmap_correlation_aps_training.png')



# TEST

df_aps = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\apstest.csv')

result = []
for column_name in df_aps.columns:
    if column_name == 'class':
        continue
    if column_name == 'bn_000':
        result.append(normalize_values(df_aps[column_name]))


draw_histogram(result, 'aps_test_histograma.png')
plt.show()

plt.boxplot(result)
plt.savefig('aps_test_boxplot.png', dpi=100)
plt.show()


data = pd.read_csv('C:\\Users\\Leona\\PycharmProjects\\LABS\\project\\median_replacement_aps_testing.csv')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
draw_heatmap_of_features(data, X, y, 'heatmap_correlation_aps_testing.png')

