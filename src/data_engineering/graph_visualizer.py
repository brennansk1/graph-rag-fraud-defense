"""
Graph Visualizer: Visualize fraud rings using NetworkX and PyVis.
"""

import pandas as pd
import networkx as nx
from pyvis.network import Network
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphVisualizer:
    def __init__(self):
        """Initialize the graph visualizer."""
        self.G = nx.Graph()
    
    def load_from_csv(self, users_csv, edges_csv):
        """Load graph data from CSV files."""
        logger.info("Loading graph data from CSV files...")
        
        # Load users
        users_df = pd.read_csv(users_csv)
        for idx, row in users_df.iterrows():
            self.G.add_node(
                int(row['user_id']),
                name=row['name'],
                is_fraud=int(row['is_fraud']),
                ssn=row['ssn'],
                device_id=row['device_id'],
                zip_code=row['zip_code']
            )
        
        logger.info(f"Loaded {len(self.G.nodes())} nodes")
        
        # Load edges
        edges_df = pd.read_csv(edges_csv)
        for idx, row in edges_df.iterrows():
            self.G.add_edge(
                int(row['source']),
                int(row['target']),
                edge_type=row['type']
            )
        
        logger.info(f"Loaded {len(self.G.edges())} edges")
    
    def get_fraud_ring_subgraph(self, ring_id=0, ring_size=10):
        """Extract a specific fraud ring for visualization."""
        # Get fraud nodes
        fraud_nodes = [node for node, attr in self.G.nodes(data=True) if attr.get('is_fraud', 0) == 1]
        
        if not fraud_nodes:
            logger.warning("No fraud nodes found in graph")
            return None
        
        # Get a subgraph around the first fraud ring
        start_idx = ring_id * ring_size
        end_idx = min(start_idx + ring_size, len(fraud_nodes))
        ring_nodes = fraud_nodes[start_idx:end_idx]
        
        # Include neighbors
        neighbors = set()
        for node in ring_nodes:
            neighbors.update(self.G.neighbors(node))
        
        all_nodes = set(ring_nodes) | neighbors
        subgraph = self.G.subgraph(all_nodes).copy()
        
        logger.info(f"Extracted fraud ring with {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges")
        return subgraph
    
    def visualize_with_pyvis(self, subgraph=None, output_file='fraud_ring_visualization.html'):
        """Visualize the graph using PyVis."""
        if subgraph is None:
            subgraph = self.G
        
        logger.info(f"Creating PyVis visualization with {len(subgraph.nodes())} nodes...")
        
        net = Network(height='750px', width='100%', directed=False, notebook=False)
        net.from_nx(subgraph)
        
        # Color nodes based on fraud status
        for node in net.nodes:
            node_data = subgraph.nodes[node['id']]
            if node_data.get('is_fraud', 0) == 1:
                node['color'] = 'red'
                node['title'] = f"FRAUD: {node_data.get('name', 'Unknown')}"
            else:
                node['color'] = 'green'
                node['title'] = f"CLEAN: {node_data.get('name', 'Unknown')}"
        
        net.show(output_file)
        logger.info(f"Visualization saved to {output_file}")
    
    def print_graph_stats(self):
        """Print basic graph statistics."""
        logger.info("=== Graph Statistics ===")
        logger.info(f"Total nodes: {len(self.G.nodes())}")
        logger.info(f"Total edges: {len(self.G.edges())}")
        
        fraud_nodes = [node for node, attr in self.G.nodes(data=True) if attr.get('is_fraud', 0) == 1]
        logger.info(f"Fraud nodes: {len(fraud_nodes)}")
        logger.info(f"Clean nodes: {len(self.G.nodes()) - len(fraud_nodes)}")
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.G)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 nodes by degree centrality:")
        for node, centrality in top_nodes:
            is_fraud = self.G.nodes[node].get('is_fraud', 0)
            logger.info(f"  Node {node} (Fraud: {is_fraud}): {centrality:.4f}")

if __name__ == "__main__":
    visualizer = GraphVisualizer()
    visualizer.load_from_csv('data/processed/users.csv', 'data/processed/graph_data.csv')
    visualizer.print_graph_stats()
    
    # Visualize the first fraud ring
    fraud_ring = visualizer.get_fraud_ring_subgraph(ring_id=0, ring_size=10)
    if fraud_ring:
        visualizer.visualize_with_pyvis(fraud_ring, 'fraud_ring_0.html')
    
    # Visualize the entire graph
    visualizer.visualize_with_pyvis(output_file='full_graph_visualization.html')
