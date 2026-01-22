"""
Feature Engineering: Calculate graph features for model training.
"""

import pandas as pd
import networkx as nx
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, users_csv, edges_csv):
        """Initialize feature engineer with graph data."""
        self.users_df = pd.read_csv(users_csv)
        self.edges_df = pd.read_csv(edges_csv)
        self.G = self._build_graph()
    
    def _build_graph(self):
        """Build NetworkX graph from CSV data."""
        G = nx.Graph()
        
        # Add nodes
        for idx, row in self.users_df.iterrows():
            G.add_node(int(row['user_id']), is_fraud=int(row['is_fraud']))
        
        # Add edges
        for idx, row in self.edges_df.iterrows():
            G.add_edge(int(row['source']), int(row['target']), edge_type=row['type'])
        
        logger.info(f"Built graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    def calculate_degree_centrality(self):
        """Calculate degree centrality for all nodes."""
        logger.info("Calculating degree centrality...")
        centrality = nx.degree_centrality(self.G)
        return pd.Series(centrality, name='degree_centrality')
    
    def calculate_pagerank(self):
        """Calculate PageRank for all nodes."""
        logger.info("Calculating PageRank...")
        pagerank = nx.pagerank(self.G)
        return pd.Series(pagerank, name='pagerank')
    
    def calculate_clustering_coefficient(self):
        """Calculate clustering coefficient for all nodes."""
        logger.info("Calculating clustering coefficient...")
        clustering = nx.clustering(self.G)
        return pd.Series(clustering, name='clustering_coefficient')
    
    def calculate_betweenness_centrality(self):
        """Calculate betweenness centrality for all nodes."""
        logger.info("Calculating betweenness centrality...")
        betweenness = nx.betweenness_centrality(self.G)
        return pd.Series(betweenness, name='betweenness_centrality')
    
    def calculate_closeness_centrality(self):
        """Calculate closeness centrality for all nodes."""
        logger.info("Calculating closeness centrality...")
        closeness = nx.closeness_centrality(self.G)
        return pd.Series(closeness, name='closeness_centrality')
    
    def calculate_connected_components(self):
        """Calculate connected component size for each node."""
        logger.info("Calculating connected components...")
        components = nx.connected_components(self.G)
        component_map = {}
        for i, component in enumerate(components):
            for node in component:
                component_map[node] = len(component)
        return pd.Series(component_map, name='component_size')
    
    def calculate_neighbor_fraud_ratio(self):
        """Calculate the ratio of fraudulent neighbors for each node."""
        logger.info("Calculating neighbor fraud ratio...")
        fraud_ratio = {}
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if len(neighbors) == 0:
                fraud_ratio[node] = 0
            else:
                fraud_neighbors = sum(1 for n in neighbors if self.G.nodes[n].get('is_fraud', 0) == 1)
                fraud_ratio[node] = fraud_neighbors / len(neighbors)
        return pd.Series(fraud_ratio, name='neighbor_fraud_ratio')
    
    def create_feature_matrix(self):
        """Create a feature matrix with all graph features."""
        logger.info("Creating feature matrix...")
        
        features = pd.DataFrame({'user_id': list(self.G.nodes())})
        features.set_index('user_id', inplace=True)
        
        # Add graph features
        features['degree_centrality'] = self.calculate_degree_centrality()
        features['pagerank'] = self.calculate_pagerank()
        features['clustering_coefficient'] = self.calculate_clustering_coefficient()
        features['betweenness_centrality'] = self.calculate_betweenness_centrality()
        features['closeness_centrality'] = self.calculate_closeness_centrality()
        features['component_size'] = self.calculate_connected_components()
        features['neighbor_fraud_ratio'] = self.calculate_neighbor_fraud_ratio()
        
        # Add target variable
        features['is_fraud'] = features.index.map(lambda x: self.G.nodes[x].get('is_fraud', 0))
        
        # Fill NaN values with 0
        features = features.fillna(0)
        
        logger.info(f"Created feature matrix with shape {features.shape}")
        return features
    
    def save_features(self, output_path='data/processed/graph_features.csv'):
        """Save feature matrix to CSV."""
        features = self.create_feature_matrix()
        features.to_csv(output_path)
        logger.info(f"Features saved to {output_path}")
        return features

if __name__ == "__main__":
    engineer = FeatureEngineer('data/processed/users.csv', 'data/processed/graph_data.csv')
    features = engineer.save_features()
    
    logger.info("\n=== Feature Statistics ===")
    logger.info(features.describe())
    
    # Print fraud vs clean statistics
    fraud_features = features[features['is_fraud'] == 1]
    clean_features = features[features['is_fraud'] == 0]
    
    logger.info("\n=== Fraud vs Clean Comparison ===")
    logger.info(f"Fraud samples: {len(fraud_features)}")
    logger.info(f"Clean samples: {len(clean_features)}")
    logger.info(f"\nFraud - Mean PageRank: {fraud_features['pagerank'].mean():.6f}")
    logger.info(f"Clean - Mean PageRank: {clean_features['pagerank'].mean():.6f}")
    logger.info(f"\nFraud - Mean Degree Centrality: {fraud_features['degree_centrality'].mean():.6f}")
    logger.info(f"Clean - Mean Degree Centrality: {clean_features['degree_centrality'].mean():.6f}")
