import numpy as np
import scipy.sparse as sp
import scanpy as sc
import pandas as pd

import torch
from torch_geometric.data import Data

def create_bipartite_graph(rna_data, atac_features):
    # returns a bipartite graph with symmetric adjacency matrix
    num_cells, num_genes = rna_data.shape
    num_nodes = num_cells + num_genes
    
    # Step 1: Create the adjacency matrix
    cell_gene_adjacency = sp.csr_matrix(rna_data)
    adjacency_matrix = sp.bmat([[None, cell_gene_adjacency], [cell_gene_adjacency.T, None]]).tocsr()
    
    # Step 2: Create the node representation matrix
    node_representation = np.zeros((num_nodes, atac_features.shape[1] + 1))
    
    # Set cell node features from atac_features and mark cell nodes with 1 in the last dimension
    node_representation[:num_cells, :-1] = atac_features
    node_representation[:num_cells, -1] = 1
    
    # Set gene nodes to have a -1 in the last column
    node_representation[num_cells:, -1] = 0
    
    return adjacency_matrix, node_representation


def convert_to_torch_geometric_data(adjacency_matrix, node_representation):
    # Step 1: Convert adjacency matrix to edge_index format
    # `coo_matrix` format allows easy access to row and col indices for non-zero entries
    adj_coo = adjacency_matrix.tocoo()
    edge_index = np.vstack((adj_coo.row, adj_coo.col))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Step 2: Convert node representation to a PyTorch tensor
    x = torch.tensor(node_representation, dtype=torch.float)

    # Step 3: Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    return data
