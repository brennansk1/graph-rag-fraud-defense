import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def create_graph_data(features_csv, edges_csv):
    """Create PyTorch Geometric Data object with train/test masks."""
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
    
    # Create masks
    indices = np.arange(len(features_df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    train_mask = torch.zeros(len(features_df), dtype=torch.bool)
    train_mask[train_idx] = True
    
    test_mask = torch.zeros(len(features_df), dtype=torch.bool)
    test_mask[test_idx] = True
    
    # Create Data object
    data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    logger.info(f"Graph created: {data.num_nodes} nodes, {data.num_edges} edges")
    logger.info(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")
    
    return data

def train_graphsage(data, num_epochs=100, learning_rate=0.01, hidden_channels=64):
    """Train GraphSAGE model with class weights."""
    logger.info("Training GraphSAGE model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Calculate class weights
    fraud_count = (data.y == 1).sum().item()
    clean_count = (data.y == 0).sum().item()
    weight = torch.tensor([1.0, clean_count / fraud_count], device=device)
    logger.info(f"Class weights: {weight}")
    
    model = GraphSAGEModel(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    data = data.to(device)
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return model

def evaluate_graphsage(model, data):
    """Evaluate model with full metrics."""
    logger.info("Evaluating GraphSAGE model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred = out.argmax(dim=1)
        y_prob = F.softmax(out, dim=1)[:, 1]
    
    # Filter for test mask
    y_test = data.y[data.test_mask].cpu().numpy()
    pred_test = y_pred[data.test_mask].cpu().numpy()
    prob_test = y_prob[data.test_mask].cpu().numpy()
    
    # Metrics
    logger.info("\n=== GraphSAGE Results ===")
    logger.info(classification_report(y_test, pred_test))
    
    roc_auc = roc_auc_score(y_test, prob_test)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    logger.info(f"True Negatives: {tn}, False Positives: {fp}")
    logger.info(f"False Negatives: {fn}, True Positives: {tp}")
    
    # Recall @ 1% FPR
    fpr, tpr, thresholds = roc_curve(y_test, prob_test)
    fpr_1pct_idx = np.argmin(np.abs(fpr - 0.01))
    recall_at_1pct_fpr = tpr[fpr_1pct_idx]
    logger.info(f"Recall @ 1% FPR: {recall_at_1pct_fpr:.4f}")
    
    return roc_auc

if __name__ == "__main__":
    data = create_graph_data('data/processed/graph_features.csv', 'data/processed/graph_data.csv')
    model = train_graphsage(data, num_epochs=50, hidden_channels=128)
    evaluate_graphsage(model, data)
    torch.save(model.state_dict(), 'models/graphsage_model.pt')
    logger.info("Model saved to models/graphsage_model.pt")