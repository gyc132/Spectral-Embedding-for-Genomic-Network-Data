from email_eu_core_utils import load_graph, load_node_attributes, visualize_graph

adj_matrix = load_graph('data/email-Eu-core/email-Eu-core.txt', 1005)
# print(adj_matrix)

isolated = []
for idx, row in enumerate(adj_matrix):
    if sum(row) == 0: isolated.append(idx)

node_departments = load_node_attributes('data/email-Eu-core/email-Eu-core-department-labels.txt')
# print(node_departments)

# visualize_graph(adj_matrix, node_departments)

import numpy as np
from sknetwork.clustering import Louvain, get_modularity
from newman_mm import NewmanMM
import matplotlib.pyplot as plt
import networkx as nx

louvain = Louvain()
labels = louvain.fit_predict(adj_matrix)
modularity = get_modularity(adj_matrix, labels)

print("Detected communities:", np.unique(labels))
print(f"Modularity: {modularity:.4f}")

# # Visualize the graph with the detected communities
# G = nx.from_numpy_array(adj_matrix)
# pos = nx.spring_layout(G)
# nx.draw(G, pos, node_color=labels, with_labels=True, cmap=plt.get_cmap('tab20'))
# plt.show()

newman = NewmanMM(adj_matrix)
splits_log = newman.main(adj_matrix)
print(splits_log)