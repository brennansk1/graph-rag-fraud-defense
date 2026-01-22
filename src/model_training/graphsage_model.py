import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super().__init__()
        self.graphsage = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels
        )
    
    def forward(self, x, edge_index):
        return self.graphsage(x, edge_index)

def create_graph_data(features_csv, edges_csv):
    """Create PyTorch Geometric Data object from CSV files."""
    logger.info("Creating graph data for GraphSAGE...")
    
    # Load features
    features_df = pd.read_csv(features_csv, index_col='user_id')
    X = torch.tensor(features_df.drop('is_fraud', axis=1).values, dtype=torch.float32)
    y = torch.tensor(features_df['is_fraud'].values, dtype=torch.long)
    
    # Load edges
    edges_df = pd.read_csv(edges_csv)
    edge_index = torch.tensor(
        [edges_df['source'].values, edges_df['target'].values],
        dtype=torch.long
    )
    
    # Create Data object
    data = Data(x=X, edge_index=edge_index, y=y)
    logger.info(f"Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
    
    return data

def train_graphsage(data, num_epochs=50, learning_rate=0.01, hidden_channels=64, num_layers=2):
    """Train GraphSAGE model."""
    logger.info("Training GraphSAGE model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = GraphSAGEModel(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move data to device
    data = data.to(device)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    logger.info("Training complete!")
    return model

def evaluate_graphsage(model, data):
    """Evaluate GraphSAGE model."""
    logger.info("Evaluating GraphSAGE model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accuracy = (pred == data.y).sum().item() / len(data.y)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # Create graph data
    data = create_graph_data('data/processed/graph_features.csv', 'data/processed/graph_data.csv')
    
    # Train model
    model = train_graphsage(data, num_epochs=50, hidden_channels=64, num_layers=2)
    
    # Evaluate model
    accuracy = evaluate_graphsage(model, data)
    
    # Save model
    torch.save(model.state_dict(), 'models/graphsage_model.pt')
    logger.info("Model saved to models/graphsage_model.pt")