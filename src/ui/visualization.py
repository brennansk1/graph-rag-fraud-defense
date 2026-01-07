from pyvis.network import Network
import networkx as nx

def visualize_graph(G):
    net = Network(notebook=True)
    net.from_nx(G)
    return net