# for node vector representation
import mira
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# attributed graph embedding
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from anndata import AnnData

from preprocess import rna_data_preprocess, remove_duplicate_var_names, filter_rna_data
from model_training import train_topic_predict
from graph_utils import create_bipartite_graph, convert_to_torch_geometric_data
from gene_corr.corr_network import s_corr_network
from graph_convolution import AGC
from args import parse_args

def main():
    args = parse_args()
    adata = sc.read_10x_h5(args.adata, gex_only=False)
    remove_duplicate_var_names(adata)
    rna_data = adata[:,adata.var['feature_types'] == 'Gene Expression'].copy()
    atac_data = adata[:,adata.var['feature_types'] == 'Peaks'].copy()

    # preprocessing
    rna_data = rna_data_preprocess(rna_data, exogenous_key=-0.1, endogenous_key=0.8)
    overlapping_barcodes = np.intersect1d(rna_data.obs_names, atac_data.obs_names) # make sure barcodes are matched between modes
    atac_data = atac_data[[i for i in overlapping_barcodes],:]
    atac_data = atac_data.copy()

    rna_data, atac_data = filter_rna_data(rna_data, atac_data, cell_threshold = 500, gene_threshold = 50)

    # load saved model to predict topics for accessibility
    model_path = 'model/pbmc_atac_model.pth'
    if os.path.exists(model_path):
        print("Model file found. Loading the model...")
        atac_model = mira.topics.load_model(model_path)
    else:
        print("Model file not found. Training a new model...")
        atac_model = train_topic_predict(atac_data, 'pbmc_atac_model.pth')

    atac_model.predict(atac_data)
    atac_model.get_umap_features(atac_data)

    print("Topic composition shape:", atac_data.obsm['X_topic_compositions'].shape)
    print("Umap features shape:", atac_data.obsm['X_umap_features'].shape)

    if args.network_type == "gene_cell":
        # prepare node representation and adjacency matrix for attributed graph
        adjacency_matrix, node_representation = create_bipartite_graph(rna_data.X, atac_data.obsm['X_topic_compositions'])
    elif args.network_type == "gene_corr":
        # correlation network
        adjacency_matrix = s_corr_network(rna_data.X, tau = 0.5)
        node_representation = atac_data.obsm['X_topic_compositions']
    # data = convert_to_torch_geometric_data(adjacency_matrix, node_representation)

    # features must be nonnegative so that the similarity matrix W is the kernel matrix XX^T.
    if np.any(node_representation < 0):
        raise ValueError("Feature matrix contains negative values. Please ensure all features are nonnegative.")
    else:
        print("Feature matrix is nonnegative.")

    adjacency_matrix = adjacency_matrix.tocoo()
    AGC(adjacency_matrix, node_representation)

if __name__ == "__main__":
    
    main()