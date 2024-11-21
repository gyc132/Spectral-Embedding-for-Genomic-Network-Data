import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def load_graph(file_path, num_nodes):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    with open(file_path, 'r') as file:
        for line in file:
            node_a, node_b = map(int, line.split())
            if node_a == node_b: continue
            adjacency_matrix[node_a, node_b] = 1
            adjacency_matrix[node_b, node_a] = 1
    
    return adjacency_matrix

def load_node_attributes(file_path):
    node_attributes = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            node, attr = map(int, line.split())
            node_attributes[node] = attr
    
    return node_attributes

def visualize_graph(adjacency_matrix, attribute):
    G = nx.Graph()

    num_nodes = adjacency_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    node_colors = [attribute.get(node, 0) for node in range(num_nodes)]
    unique_attributes = list(set(node_colors))
    color_map = plt.get_cmap('tab20')
    norm = mcolors.Normalize(vmin=min(unique_attributes), vmax=max(unique_attributes))
    node_color_list = [color_map(norm(attribute.get(node, 0))) for node in range(num_nodes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_color_list, node_size=500, font_size=10, font_weight='bold')

    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Department")

    plt.title("Graph Visualization with Department Attributes")
    plt.show()