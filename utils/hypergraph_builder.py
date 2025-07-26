import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def build_api_hypergraph(df):
    """
    Build a hypergraph where nodes are packages and hyperedges connect packages 
    using similar API calls
    
    Parameters:
    df (pandas.DataFrame): Dataframe containing package information
    
    Returns:
    torch.Tensor: Hypergraph incidence matrix H
    """
    # Get all unique API calls
    all_apis = set()
    for apis in df['api_calls']:
        if isinstance(apis, list):
            all_apis.update(apis)
    
    # Create incidence matrix H where H[i,j] = 1 if package i contains API j
    num_packages = len(df)
    num_apis = len(all_apis)
    
    # Map API calls to indices
    api_to_idx = {api: i for i, api in enumerate(all_apis)}
    
    # Initialize hypergraph incidence matrix
    H = torch.zeros(num_packages, num_apis)
    
    # Populate incidence matrix
    for i, apis in enumerate(df['api_calls']):
        if isinstance(apis, list):
            for api in apis:
                j = api_to_idx.get(api)
                if j is not None:
                    H[i, j] = 1.0
    
    return H

def build_knn_hypergraph(features, k=5):
    """
    Build a KNN-based hypergraph where each node and its k nearest neighbors form a hyperedge
    
    Parameters:
    features (numpy.ndarray): Feature matrix
    k (int): Number of nearest neighbors
    
    Returns:
    torch.Tensor: Hypergraph incidence matrix H
    """
    # Compute KNN graph
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(features)
    _, indices = nbrs.kneighbors(features)
    
    # Create hypergraph incidence matrix
    num_nodes = features.shape[0]
    H = torch.zeros(num_nodes, num_nodes)
    
    # For each node, create a hyperedge that connects it to its k nearest neighbors
    for i in range(num_nodes):
        for j in indices[i]:
            H[i, j] = 1.0
    
    return H

def build_multi_hypergraph(df, features):
    """
    Build a multi-view hypergraph that combines API-based and KNN-based hypergraphs
    
    Parameters:
    df (pandas.DataFrame): Dataframe containing package information
    features (numpy.ndarray): Feature matrix
    
    Returns:
    torch.Tensor: Combined hypergraph incidence matrix H
    """
    # Build individual hypergraphs
    H_api = build_api_hypergraph(df)
    H_knn = build_knn_hypergraph(features)
    
    # Combine hypergraphs (concatenate hyperedges)
    H_combined = torch.cat([H_api, H_knn], dim=1)
    
    return H_combined

def build_hypergraph(df, features, method='multi'):
    """
    Build a hypergraph based on the specified method
    
    Parameters:
    df (pandas.DataFrame): Dataframe containing package information
    features (numpy.ndarray): Feature matrix
    method (str): Method to use for hypergraph construction ('api', 'knn', or 'multi')
    
    Returns:
    torch.Tensor: Hypergraph incidence matrix H
    """
    if method == 'api':
        return build_api_hypergraph(df)
    elif method == 'knn':
        return build_knn_hypergraph(features)
    elif method == 'multi':
        return build_multi_hypergraph(df, features)
    else:
        raise ValueError(f"Unknown hypergraph construction method: {method}")