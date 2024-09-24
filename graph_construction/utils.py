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
# cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from lingam import DirectLiNGAM



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
    return cosine_similarity_sklearn(data.T)

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

    # knn_graph to adjacency matrix

    knn_graph_adj_matrix = np.zeros((data.shape[1], data.shape[1]))

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i != j:
                knn_graph_adj_matrix[i, j] = knn_graph[i, j]

    return knn_graph_adj_matrix

def spearman_correlation(data):
    matrix = data.corr(method='spearman')

    spearman_correlation_matrix = np.zeros((data.shape[1], data.shape[1]))

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            spearman_correlation_matrix[i, j] = matrix.iloc[i, j]


    return spearman_correlation_matrix

def kendall_correlation(data):
    matrix = data.corr(method='kendall')

    kendall_correlation_matrix = np.zeros((data.shape[1], data.shape[1]))

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            kendall_correlation_matrix[i, j] = matrix.iloc[i, j]


    return kendall_correlation_matrix

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



import numpy as np
from scipy.signal import coherence

def coherence_matrix(data, fs=1.0, nperseg=None):
    """
    Calculate the coherence matrix for a set of time series data.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): A 2D array where rows are time points and columns are different ROI time series.
    fs (float, optional): The sampling frequency of the data. Default is 1.0.
    nperseg (int, optional): Length of each segment for calculating coherence. Default is None, which uses scipy's default.

    Returns:
    numpy.ndarray: A symmetric matrix of coherence values.
    """
    # Convert DataFrame to numpy array if necessary
    if isinstance(data, pd.DataFrame):
        data = data.values

    n = data.shape[1]  # Number of ROIs
    coherence_matrix = np.zeros((n, n))  # Initialize an empty coherence matrix

    nperseg = nperseg or data.shape[0] // 4  # Default to 25% of the data length

    for i in range(n):
        for j in range(i, n):
            # Compute coherence between ROI i and ROI j
            f, Cxy = coherence(data[:, i], data[:, j], fs=fs, nperseg=nperseg)
            coherence_matrix[i, j] = np.mean(Cxy)  # Average coherence over all frequencies
            coherence_matrix[j, i] = coherence_matrix[i, j]  # Symmetric matrix

    return coherence_matrix



import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def time_delay_embedding(time_series, embedding_dim, time_delay):
    """
    Perform time-delay embedding for a given time series.

    Parameters:
    time_series (numpy.ndarray): The input time series to be embedded.
    embedding_dim (int): The embedding dimension.
    time_delay (int): The time delay.

    Returns:
    numpy.ndarray: The embedded time series in a higher dimensional space.
    """
    N = len(time_series)
    if embedding_dim * time_delay > N:
        raise ValueError(f"Time series too short for the desired embedding parameters. "
                         f"Series length: {N}, required length: {embedding_dim * time_delay}")
    
    # Generate the embedded matrix ensuring each element has the same length
    embedded = []
    for i in range(embedding_dim):
        start_idx = i * time_delay
        end_idx = N - (embedding_dim - 1) * time_delay
        embedded.append(time_series[start_idx:start_idx + end_idx])
    
    embedded = np.array(embedded).T  # Convert list of arrays to 2D numpy array and transpose
    return embedded

def generalised_synchronisation_matrix(data, embedding_dim=10, time_delay=1):
    """
    Calculate the generalised synchronisation matrix for a set of time series data.

    Parameters:
    data (pandas.DataFrame or numpy.ndarray): A 2D array where rows are time points and columns are different ROI time series.
    embedding_dim (int, optional): The embedding dimension for time-delay embedding. Default is 10.
    time_delay (int, optional): The time delay for time-delay embedding. Default is 1.

    Returns:
    numpy.ndarray: A symmetric matrix of generalised synchronisation values.
    """
    # If data is a pandas DataFrame, convert it to a numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values

    n = data.shape[1]  # Number of ROIs (time series)
    gs_matrix = np.zeros((n, n))  # Initialize an empty matrix for generalised synchronisation values

    for i in range(n):
        for j in range(i, n):
            # Get the two time series to compare
            time_series_1 = data[:, i]
            time_series_2 = data[:, j]
            
            # Check if time series are long enough for the embedding
            if len(time_series_1) < embedding_dim * time_delay or len(time_series_2) < embedding_dim * time_delay:
                print(f"Skipping pair ({i}, {j}) due to insufficient length for embedding.")
                continue
            
            # Perform time-delay embedding for each time series
            try:
                embedded_1 = time_delay_embedding(time_series_1, embedding_dim, time_delay)
                embedded_2 = time_delay_embedding(time_series_2, embedding_dim, time_delay)
            except ValueError as e:
                print(f"Skipping pair ({i}, {j}) due to embedding error: {e}")
                continue
            
            # Compute the distance matrix for each embedded time series
            dist_1 = cdist(embedded_1, embedded_1)
            dist_2 = cdist(embedded_2, embedded_2)
            
            # Find the nearest neighbors in the first embedded space
            nn_1 = NearestNeighbors(n_neighbors=2).fit(dist_1).kneighbors(dist_1, return_distance=False)[:, 1]
            
            # Calculate the synchronisation sum
            synchronisation_sum = 0
            for k in range(len(embedded_1)):
                l = nn_1[k]
                synchronisation_sum += dist_2[k, l]
            
            # Calculate the average synchronisation distance
            gs_value = synchronisation_sum / len(embedded_1)
            
            gs_matrix[i, j] = gs_value
            gs_matrix[j, i] = gs_value  # Symmetric matrix

    return gs_matrix


import numpy as np
import pandas as pd

def binarize_data(data, threshold=0.5):
    """
    Binarize the input data based on a threshold.

    Parameters:
    data (numpy.ndarray): Input data with shape (time_points, ROIs).
    threshold (float): Threshold for binarization.

    Returns:
    numpy.ndarray: Binarized data.
    """
    return (data > threshold).astype(int)

def calculate_conditional_probabilities(data):
    """
    Calculate the conditional probabilities for each pair of variables.

    Parameters:
    data (numpy.ndarray): Binarized data with shape (time_points, ROIs).

    Returns:
    tuple: (P(X), P(Y), P(X|Y), P(Y|X), P(X,Y))
    """
    n = data.shape[1]
    P_X = np.mean(data, axis=0)  # P(X) for each ROI
    P_Y = P_X.copy()  # P(Y) is the same as P(X) for each ROI

    P_X_given_Y = np.zeros((n, n))
    P_Y_given_X = np.zeros((n, n))
    P_XY = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                P_X_given_Y[i, j] = 1.0
                P_Y_given_X[i, j] = 1.0
                P_XY[i, j] = P_X[i]
            else:
                # Calculate joint probability P(X,Y)
                P_XY[i, j] = np.mean((data[:, i] == 1) & (data[:, j] == 1))
                # Calculate P(X|Y) = P(X,Y) / P(Y)
                if P_Y[j] != 0:
                    P_X_given_Y[i, j] = P_XY[i, j] / P_Y[j]
                # Calculate P(Y|X) = P(X,Y) / P(X)
                if P_X[i] != 0:
                    P_Y_given_X[i, j] = P_XY[i, j] / P_X[i]

    return P_X, P_Y, P_X_given_Y, P_Y_given_X, P_XY

def patels_conditional_dependence_measures_kappa(data, threshold=0.5):
    """
    Calculate the kappa (κ) matrix of Patel's Conditional Dependence Measures for the input data.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): Input data with shape (time_points, ROIs).
    threshold (float, optional): Threshold for binarization. Default is 0.5.

    Returns:
    numpy.ndarray: Kappa matrix with diagonal elements set to 1.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Step 1: Binarize the data
    binarized_data = binarize_data(data, threshold=threshold)

    # Step 2: Calculate conditional probabilities
    P_X, P_Y, P_X_given_Y, _, _ = calculate_conditional_probabilities(binarized_data)

    n = data.shape[1]
    kappa_matrix = np.zeros((n, n))

    # Step 3: Calculate kappa (κ) matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                kappa_matrix[i, j] = 1.0  # Set diagonal elements to 1
            else:
                # κ = P(X|Y) - P(X)
                kappa_matrix[i, j] = P_X_given_Y[i, j] - P_X[i]

    return kappa_matrix

def patels_conditional_dependence_measures_tau(data, threshold=0.5):
    """
    Calculate the tau (τ) matrix of Patel's Conditional Dependence Measures for the input data.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): Input data with shape (time_points, ROIs).
    threshold (float, optional): Threshold for binarization. Default is 0.5.

    Returns:
    numpy.ndarray: Tau matrix with diagonal elements set to 1.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Step 1: Binarize the data
    binarized_data = binarize_data(data, threshold=threshold)

    # Step 2: Calculate conditional probabilities
    _, _, P_X_given_Y, P_Y_given_X, _ = calculate_conditional_probabilities(binarized_data)

    n = data.shape[1]
    tau_matrix = np.zeros((n, n))

    # Step 3: Calculate tau (τ) matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                tau_matrix[i, j] = 1.0  # Set diagonal elements to 1
            else:
                # τ = P(X|Y) - P(Y|X)
                tau_matrix[i, j] = P_X_given_Y[i, j] - P_Y_given_X[i, j]

    return tau_matrix



def lingam(data):
    """
    Perform LiNGAM causal analysis on the given data.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): Input data with shape (samples, variables).
                                              Each column represents a different variable.

    Returns:
    tuple: (causal_matrix, causal_order)
           - causal_matrix: The estimated causal influence matrix.
           - causal_order: The estimated causal ordering of the variables.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Ensure that data is 2D
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D input data, got shape {data.shape}")
    
    print("Performing LiNGAM causal analysis...")
    
    # Create a DirectLiNGAM model and fit the data
    model = DirectLiNGAM()
    model.fit(data)
    
    # Get the causal matrix and causal order
    causal_matrix = model.adjacency_matrix_
    causal_order = model.causal_order_
    
    return causal_matrix

def plot_correlation_matrix(correlation_matrix, method=""):
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", xticklabels=True, yticklabels=True)
    # plt.title("Cross-Correlation Matrix")
    plt.title("Correlation Matrix" + " of " + method)
    plt.show()    




