from dowhy import CausalModel
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fairness_bias(features_csv, model_predictions_csv):
    """
    Test for bias in fraud detection using DoWhy.
    
    Hypothesis: The model should rely on graph structure (fraud patterns),
    not on demographic attributes like zip code.
    """
    logger.info("Testing for fairness bias using DoWhy...")
    
    # Load features
    features_df = pd.read_csv(features_csv, index_col='user_id')
    
    # Load model predictions
    predictions_df = pd.read_csv(model_predictions_csv, index_col='user_id')
    
    # Merge data
    data = features_df.join(predictions_df)
    
    # Create binary zip code treatment (fraud-prone zip vs others)
    data['fraud_zip'] = (data['zip_code'] == '90210').astype(int)
    
    # Treatment: fraud-prone zip code
    # Outcome: model prediction of fraud
    # Confounder: actual fraud status (is_fraud)
    
    gml_graph = """
    digraph {
        fraud_zip -> model_pred;
        is_fraud -> model_pred;
        is_fraud -> fraud_zip;
    }
    """
    
    try:
        model = CausalModel(
            data=data,
            treatment='fraud_zip',
            outcome='model_pred',
            common_causes=['is_fraud'],
            graph=gml_graph
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

def analyze_demographic_parity(features_csv, model_predictions_csv):
    """
    Analyze demographic parity: P(pred=1|zip=fraud_zip) vs P(pred=1|zip!=fraud_zip)
    """
    logger.info("Analyzing demographic parity...")
    
    features_df = pd.read_csv(features_csv, index_col='user_id')
    predictions_df = pd.read_csv(model_predictions_csv, index_col='user_id')
    
    data = features_df.join(predictions_df)
    
    fraud_zip_group = data[data['zip_code'] == '90210']
    other_zip_group = data[data['zip_code'] != '90210']
    
    fraud_zip_pred_rate = fraud_zip_group['model_pred'].mean()
    other_zip_pred_rate = other_zip_group['model_pred'].mean()
    
    logger.info(f"\n=== Demographic Parity Analysis ===")
    logger.info(f"Fraud-prone zip prediction rate: {fraud_zip_pred_rate:.4f}")
    logger.info(f"Other zip prediction rate: {other_zip_pred_rate:.4f}")
    logger.info(f"Difference: {abs(fraud_zip_pred_rate - other_zip_pred_rate):.4f}")
    logger.info("Interpretation: Smaller difference indicates better fairness")
    
    return {
        'fraud_zip_rate': fraud_zip_pred_rate,
        'other_zip_rate': other_zip_pred_rate,
        'difference': abs(fraud_zip_pred_rate - other_zip_pred_rate)
    }

if __name__ == "__main__":
    # This would be run after model training
    # estimate = test_fairness_bias('data/processed/graph_features.csv', 'data/processed/model_predictions.csv')
    # parity = analyze_demographic_parity('data/processed/graph_features.csv', 'data/processed/model_predictions.csv')
    logger.info("Fairness testing module ready. Run after model training.")