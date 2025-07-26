import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import seaborn as sns

def visualize_hypergraph(H, save_path=None, sample_size=100):
    """
    Visualize a hypergraph
    
    Parameters:
    H (torch.Tensor): Hypergraph incidence matrix
    save_path (str): Path to save the visualization
    sample_size (int): Number of nodes to sample for visualization
    """
    # Convert to numpy
    if isinstance(H, torch.Tensor):
        H = H.numpy()
    
    # Sample nodes if there are too many
    num_nodes = H.shape[0]
    if num_nodes > sample_size:
        indices = np.random.choice(num_nodes, sample_size, replace=False)
        H_sampled = H[indices, :]
    else:
        H_sampled = H
        indices = np.arange(num_nodes)
    
    # Create a bipartite graph representation
    G = nx.Graph()
    
    # Add nodes
    for i in range(H_sampled.shape[0]):
        G.add_node(f"node_{i}", bipartite=0)
    
    # Add hyperedges as nodes
    for j in range(H_sampled.shape[1]):
        G.add_node(f"edge_{j}", bipartite=1)
    
    # Add edges between nodes and hyperedges
    for i in range(H_sampled.shape[0]):
        for j in range(H_sampled.shape[1]):
            if H_sampled[i, j] > 0:
                G.add_edge(f"node_{i}", f"edge_{j}")
    
    # Create positions for the nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Plot the graph
    plt.figure(figsize=(12, 8))
    
    # Draw the nodes
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=[n for n, d in G.nodes(data=True) if d['bipartite'] == 0],
        node_color='skyblue', node_size=50, label='Packages'
    )
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=[n for n, d in G.nodes(data=True) if d['bipartite'] == 1],
        node_color='lightgreen', node_size=30, node_shape='s', label='Hyperedges'
    )
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    plt.title(f"Hypergraph Visualization (Sample of {H_sampled.shape[0]} nodes)")
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_hyperedge_distribution(H, save_path=None):
    """
    Plot the distribution of hyperedges per node
    
    Parameters:
    H (torch.Tensor): Hypergraph incidence matrix
    save_path (str): Path to save the plot
    """
    # Convert to numpy
    if isinstance(H, torch.Tensor):
        H = H.numpy()
    
    # Count hyperedges per node
    hyperedges_per_node = np.sum(H > 0, axis=1)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(hyperedges_per_node, kde=True)
    plt.xlabel('Number of Hyperedges')
    plt.ylabel('Number of Nodes')
    plt.title('Distribution of Hyperedges per Node')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_node_distribution(H, save_path=None):
    """
    Plot the distribution of nodes per hyperedge
    
    Parameters:
    H (torch.Tensor): Hypergraph incidence matrix
    save_path (str): Path to save the plot
    """
    # Convert to numpy
    if isinstance(H, torch.Tensor):
        H = H.numpy()
    
    # Count nodes per hyperedge
    nodes_per_hyperedge = np.sum(H > 0, axis=0)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(nodes_per_hyperedge, kde=True)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of Hyperedges')
    plt.title('Distribution of Nodes per Hyperedge')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()