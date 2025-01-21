import scanpy as sc
import numpy as np


def rna_data_preprocess(rna_data, exogenous_key, endogenous_key):
    rna_data.var.index = rna_data.var.index.str.upper()
    rna_data.var_names_make_unique()
    rna_data = rna_data[:, ~rna_data.var.index.str.startswith('GM')]

    sc.pp.filter_cells(rna_data, min_counts = 400)
    sc.pp.filter_genes(rna_data, min_cells=15)

    rna_data.var['mt'] = rna_data.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(rna_data, qc_vars=['mt'], percent_top=None,
                            log1p=False, inplace=True)

    rna_data = rna_data[rna_data.obs.pct_counts_mt < 15, :]
    rna_data = rna_data[rna_data.obs.n_genes_by_counts < 8000, :]
    sc.pp.filter_genes(rna_data, min_cells=15)

    rna_data.raw = rna_data # save raw counts
    sc.pp.normalize_total(rna_data, target_sum=1e4)
    sc.pp.log1p(rna_data)

    sc.pp.highly_variable_genes(rna_data, min_disp = exogenous_key)
    rna_data.layers['norm'] = rna_data.X # save normalized count data
    rna_data.X = rna_data.raw.X # and reload raw counts
    rna_data = rna_data[:, rna_data.var.highly_variable] 
    rna_data.var['exog_feature'] = rna_data.var.highly_variable # set column "exog_features" to all genes that met dispersion threshold
    rna_data.var.highly_variable = (rna_data.var.dispersions_norm > endogenous_key) & rna_data.var.exog_feature

    return rna_data


def remove_duplicate_var_names(adata):
    var_names = adata.var_names
    _, unique_indices = np.unique(var_names, return_index=True)
    unique_indices = np.sort(unique_indices)
    adata = adata[:, unique_indices].copy()

    return adata


def filter_rna_data(rna_data, atac_data, cell_threshold, gene_threshold):
    # Get boolean masks for cells and genes to be removed
    cell_nonzero_counts = (rna_data.X > 0).sum(axis=1).A1
    gene_nonzero_counts = (rna_data.X > 0).sum(axis=0).A1

    # Identify rows (cells) and columns (genes) to remove
    cells_to_remove = cell_nonzero_counts < cell_threshold
    genes_to_remove = gene_nonzero_counts < gene_threshold

    # Count the number of removals
    num_cells_removed = np.sum(cells_to_remove)
    num_genes_removed = np.sum(genes_to_remove)

    print(f"Cells removed: {num_cells_removed}")
    print(f"Genes removed: {num_genes_removed}")

    # Apply the mask to filter the data
    filtered_rna_data = rna_data[~cells_to_remove, :][:, ~genes_to_remove].copy()
    filtered_atac_data = atac_data[~cells_to_remove, :].copy()

    print("Filtered RNA shape:", filtered_rna_data.shape)
    print("Filtered ATAC shape:", filtered_atac_data.shape)

    return filtered_rna_data, filtered_atac_data