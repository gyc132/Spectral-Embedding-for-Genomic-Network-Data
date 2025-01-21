from gene_corr.s_correlation import cal_s_correlation
import scipy.sparse as sp
from tqdm import tqdm
import numpy as np

def s_corr_network(rna_data, tau):
    num_genes = rna_data.shape[0]
    adjacency_matrix = sp.lil_matrix((num_genes, num_genes))
    rna_data = rna_data.toarray() if sp.issparse(rna_data) else np.array(rna_data)

    for i in tqdm(range(num_genes)):
        for j in range(i + 1, num_genes):
            s_corr = cal_s_correlation(rna_data[i], rna_data[j])
            if s_corr > tau:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1 

    return adjacency_matrix.tocsr()