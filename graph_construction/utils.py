import torch
from torch_geometric.data import Data, Batch
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import numpy as np
import pingouin as pg
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
import matplotlib.pyplot as plt


# Function to convert a correlation matrix into a graph Data object
def correlation_matrix_to_graph_data(corr_matrix):
    num_nodes = corr_matrix.shape[0]

    # Create edge index (connections between nodes)
    edge_index = np.nonzero(corr_matrix)  # Get indices where correlations are non-zero
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create edge attributes (the correlation values)
    edge_attr = torch.tensor(corr_matrix[edge_index[0], edge_index[1]], dtype=torch.float)

    x = torch.tensor(corr_matrix, dtype=torch.float)  # Use the correlation matrix itself as node features

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def pearson_correlation(data):
    return np.corrcoef(data, rowvar=False)

def cosine_similarity(data):
    return cosine_similarity(data.T)

def partial_correlation(data):
    return pg.partial_corr(data, x=0, y=1, covar=2)

def correlations_correlation(data):
    pearson_correlation_matrix = np.corrcoef(data, rowvar=False)

    # Calculate the Correlation's Correlation (High-order FC)
    # Step 1: Calculate the topographical profile (each row) for each ROI from the Pearson matrix
    n = pearson_correlation_matrix.shape[0]
    high_order_fc_matrix = np.zeros((n, n))

    # Step 2: Compute the correlation's correlation for each pair of ROIs
    for row_index in range(n):
        for col_index in range(row_index + 1, n):  # Only need to compute upper triangular part
            high_order_fc_matrix[row_index, col_index] = np.corrcoef(pearson_correlation_matrix[row_index, :], pearson_correlation_matrix[col_index, :])[0, 1]
            high_order_fc_matrix[col_index, row_index] = high_order_fc_matrix[row_index, col_index]  # Symmetric matrix
    return high_order_fc_matrix

def associated_high_order_fc(data):
    pearson_correlation_matrix = np.corrcoef(data, rowvar=False)

    # Calculate the Correlation's Correlation (High-order FC)
    n = pearson_correlation_matrix.shape[0]
    high_order_fc_matrix = np.zeros((n, n))

    for row_index in range(n):
        for col_index in range(row_index + 1, n):  # Only need to compute upper triangular part
            high_order_fc_matrix[row_index, col_index] = np.corrcoef(pearson_correlation_matrix[row_index, :], pearson_correlation_matrix[col_index, :])[0, 1]
            high_order_fc_matrix[col_index, row_index] = high_order_fc_matrix[row_index, col_index]  # Symmetric matrix

    # Calculate the Associated High-order Functional Connectivity Matrix
    associated_high_order_fc_matrix = np.zeros((n, n))

    for row_index in range(n):
        for col_index in range(row_index + 1, n):
            associated_high_order_fc_matrix[row_index, col_index] = np.corrcoef(pearson_correlation_matrix[row_index, :], high_order_fc_matrix[col_index, :])[0, 1]
            associated_high_order_fc_matrix[col_index, row_index] = associated_high_order_fc_matrix[row_index, col_index]  # Symmetric matrix

    return associated_high_order_fc_matrix

def euclidean_distance(data):
    n_vars = data.shape[1]
    # calculate the euclidean distance matrix
    euclidean_distance_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            euclidean_distance_matrix[i, j] = np.linalg.norm(data.iloc[:, i] - data.iloc[:, j])
            euclidean_distance_matrix[j, i] = euclidean_distance_matrix[i, j]
    return euclidean_distance_matrix

from sklearn.neighbors import kneighbors_graph

def knn_graph(data, K = 20):

    # Number of nearest neighbors
    K = 20

    # Calculate the KNN graph, using  weighted by the Gaussian similarity function of Euclidean distances,

    knn_graph = kneighbors_graph(data.T, K, mode='distance', metric='euclidean', include_self=False)

    return knn_graph

def spearman_correlation(data):
    return data.corr(method='spearman')

def kendall_correlation(data):
    return data.corr(method='kendall')

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression


def mutual_information(data):
    num_rois = data.shape[1]
    mutual_info_matrix = np.zeros((num_rois, num_rois))
    
    for i in range(num_rois):
        for j in range(num_rois):
            if i == j:
                mutual_info_matrix[i, j] = 0  # Mutual information with itself is not needed
            else:
                # Estimate mutual information between the two time series
                mutual_info_matrix[i, j] = mutual_info_regression(data.iloc[:, i].values.reshape(-1, 1), 
                                                                  data.iloc[:, j].values)[0]

    return mutual_info_matrix


def cross_correlation(data):
    num_rois = data.shape[1]
    cross_corr_matrix = np.zeros((num_rois, num_rois))

    for i in range(num_rois):    
        for j in range(num_rois):
            cross_corr_matrix[i, j] = np.corrcoef(data.iloc[:, i], data.iloc[:, j])[0, 1]
    return cross_corr_matrix

def granger_causality(data, max_lag=1):
    num_rois = data.shape[1]
    granger_matrix = np.zeros((num_rois, num_rois))

    for i in range(num_rois):
        for j in range(num_rois):
            if i != j:
                test_result = grangercausalitytests(data[[data.columns[i], data.columns[j]]], max_lag, verbose=False)
                p_values = [round(test[0]['ssr_ftest'][1], 4) for test in test_result.values()]
                granger_matrix[i, j] = np.min(p_values)
    
    return granger_matrix


def plot_correlation_matrix(correlation_matrix, method=""):
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", xticklabels=True, yticklabels=True)
    # plt.title("Cross-Correlation Matrix")
    plt.title("Correlation Matrix" + " of " + method)
    plt.show()    




