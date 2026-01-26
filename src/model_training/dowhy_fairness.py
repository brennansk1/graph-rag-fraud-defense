from dowhy import CausalModel
import pandas as pd
import numpy as np
import logging
import torch
import torch.nn.functional as F
from src.model_training.graphsage_model import GraphSAGEModel, create_graph_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_predictions(model_path, features_path, edges_path):
    """Generate predictions using the trained GraphSAGE model."""
    logger.info("Generating predictions from GraphSAGE model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = create_graph_data(features_path, edges_path)
    data = data.to(device)
    
    # Load model
    model = GraphSAGEModel(
        in_channels=data.num_features,
        hidden_channels=128,  # Must match training
        out_channels=2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred = out.argmax(dim=1).cpu().numpy()
        y_prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        
    return pd.DataFrame({
        'model_pred': y_pred,
        'model_prob': y_prob
    }, index=pd.read_csv(features_path, index_col='user_id').index)

def test_fairness_bias(data):
    """
    Test for bias in fraud detection using DoWhy.
    Hypothesis: The model should rely on graph structure (fraud patterns),
    not on demographic attributes like zip code.
    """
    logger.info("Testing for fairness bias using DoWhy...")
    
    # Create binary zip code treatment (fraud-prone zip vs others)
    data['zip_code'] = data['zip_code'].astype(str)
    data['fraud_zip'] = (data['zip_code'] == '90210').astype(int)
    
    # Graph: Zip causes Fraud (maybe), Zip causes Prediction? 
    # Use NetworkX graph to avoid pydot/pygraphviz dependency issues
    import networkx as nx
    causal_graph = nx.DiGraph()
    causal_graph.add_edge('fraud_zip', 'model_pred')
    causal_graph.add_edge('is_fraud', 'model_pred')
    causal_graph.add_edge('is_fraud', 'fraud_zip')
    
    try:
        model = CausalModel(
            data=data,
            treatment='fraud_zip',
            outcome='model_pred',
            common_causes=['is_fraud'],
            graph=causal_graph
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        logger.info(f"Identified estimand: {identified_estimand}")
        
        # Estimate using propensity score matching
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching"
        )
        
        logger.info(f"\n=== Fairness Analysis Results ===")
        logger.info(f"ATE (Average Treatment Effect): {estimate.value:.4f}")
        logger.info("Interpretation: If ATE is close to 0, the model is fair (not biased by zip code)")
        
        return estimate
    
    except Exception as e:
        logger.error(f"Error in fairness analysis: {e}")
        return None

def analyze_demographic_parity(data):
    """
    Analyze demographic parity.
    """
    logger.info("Analyzing demographic parity...")
    
    fraud_zip_group = data[data['zip_code'] == '90210']
    other_zip_group = data[data['zip_code'] != '90210']
    
    fraud_zip_pred_rate = fraud_zip_group['model_pred'].mean()
    other_zip_pred_rate = other_zip_group['model_pred'].mean()
    
    logger.info(f"\n=== Demographic Parity Analysis ===")
    logger.info(f"Fraud-prone zip prediction rate: {fraud_zip_pred_rate:.4f}")
    logger.info(f"Other zip prediction rate: {other_zip_pred_rate:.4f}")
    logger.info(f"Difference: {abs(fraud_zip_pred_rate - other_zip_pred_rate):.4f}")
    
    return {
        'fraud_zip_rate': fraud_zip_pred_rate,
        'other_zip_rate': other_zip_pred_rate
    }

if __name__ == "__main__":
    # 1. Generate Predictions
    preds_df = generate_predictions(
        'models/graphsage_model.pt',
        'data/processed/graph_features.csv',
        'data/processed/graph_data.csv'
    )
    
    # 2. Load Demographic Data (Users)
    users_df = pd.read_csv('data/processed/users.csv', index_col='user_id')
    
    # 3. Merge
    # Note: users.csv has 'is_fraud' and 'zip_code'
    data = users_df.join(preds_df[['model_pred', 'model_prob']])
    
    # 4. Run Analysis
    test_fairness_bias(data)
    analyze_demographic_parity(data)