import scipy.io as sio
import time
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
# from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)


    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist

def eucl_dist(prelabel, feature):
    k = len(np.unique(prelabel))
    intra_dist = 0

    for i in range(k):
        Data_i = feature[np.where(prelabel == i)]

        Dis = euclidean_distances(Data_i, Data_i)
        n_i = Data_i.shape[0]
        if n_i == 0 or n_i == 1:
            intra_dist = intra_dist
        else:
            intra_dist = intra_dist + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(Dis))

    return intra_dist


def AGC(adj, feature, max_iter=60, rep=5, num_com=0, dist=square_dist):
    if sp.issparse(feature): feature = feature.todense()
    adj = sp.coo_matrix(adj)
    if num_com == 0: num_com = int((adj.shape[0])**0.5)
    intra_list = []
    intra_list.append(10000)

    t = time.time()
    adj_normalized = preprocess_adj(adj)
    graph_filter = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2 # The graph filter G = I - L/2

    iter_idx = 0
    output_labels = []
    while True:
        iter_idx += 1
        intraD = np.zeros(rep)
        feature = graph_filter.dot(feature)
        u, s, v = sp.linalg.svds(feature, k=min(num_com, feature.shape[1]-1), which='LM')

        for i in range(rep):
            kmeans = KMeans(n_clusters=num_com).fit(u)
            predict_labels = kmeans.predict(u)
            intraD[i] = dist(predict_labels, feature)
        
        intramean = np.mean(intraD)
        intra_list.append(intramean)
        print(f"iter_idx:{iter_idx};intramean:{intramean}")

        if intra_list[iter_idx] > intra_list[iter_idx - 1] or iter_idx > max_iter:
            print(f'bestpower: {iter_idx - 1}')
            print(time.time() - t)
            return output_labels
        else:
            output_labels = predict_labels


if __name__ == "__main__":
    adj = sp.coo_matrix([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    feature = np.array([
        [1.0, 0.5],
        [0.5, 1.0],
        [0.8, 0.2],
        [0.1, 0.9]
    ])
    
    labels = AGC(adj, feature, max_iter=10, rep=3, num_com=2)
    print("Predicted cluster labels:", labels)
