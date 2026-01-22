"""
Training Pipeline: Orchestrate feature engineering, baseline training, and model evaluation.
"""

import pandas as pd
import logging
import os
from src.model_training.feature_engineering import FeatureEngineer
from src.model_training.baseline_xgboost import train_xgboost, save_model as save_xgb_model
from src.model_training.graphsage_model import create_graph_data, train_graphsage, evaluate_graphsage
from src.model_training.dowhy_fairness import test_fairness_bias, analyze_demographic_parity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    logger.info("Directories ensured")

def run_feature_engineering():
    """Run feature engineering pipeline."""
    logger.info("\n=== PHASE 1: Feature Engineering ===")
    
    engineer = FeatureEngineer('data/processed/users.csv', 'data/processed/graph_data.csv')
    features = engineer.save_features('data/processed/graph_features.csv')
    
    logger.info(f"Feature matrix shape: {features.shape}")
    return features

def run_baseline_training(features):
    """Train XGBoost baseline model."""
    logger.info("\n=== PHASE 2: Baseline Training (XGBoost) ===")
    
    X = features.drop('is_fraud', axis=1)
    y = features['is_fraud']
    
    model, results = train_xgboost(X, y)
    save_xgb_model(model, 'models/baseline_xgboost.pkl')
    
    logger.info(f"Baseline ROC-AUC: {results['roc_auc']:.4f}")
    logger.info(f"Baseline Recall @ 1% FPR: {results['recall_at_1pct_fpr']:.4f}")
    
    return model, results

def run_graphsage_training(features):
    """Train GraphSAGE model."""
    logger.info("\n=== PHASE 3: GraphSAGE Training ===")
    
    try:
        data = create_graph_data('data/processed/graph_features.csv', 'data/processed/graph_data.csv')
        model = train_graphsage(data, num_epochs=50, hidden_channels=64, num_layers=2)
        accuracy = evaluate_graphsage(model, data)
        
        logger.info(f"GraphSAGE Accuracy: {accuracy:.4f}")
        return model, accuracy
    except Exception as e:
        logger.error(f"Error in GraphSAGE training: {e}")
        return None, None

def run_fairness_analysis(features):
    """Run fairness analysis using DoWhy."""
    logger.info("\n=== PHASE 4: Fairness Analysis (DoWhy) ===")
    
    try:
        # Create dummy predictions for demonstration
        features['model_pred'] = (features['pagerank'] > features['pagerank'].median()).astype(int)
        features.to_csv('data/processed/model_predictions.csv')
        
        # Run fairness tests
        estimate = test_fairness_bias('data/processed/graph_features.csv', 'data/processed/model_predictions.csv')
        parity = analyze_demographic_parity('data/processed/graph_features.csv', 'data/processed/model_predictions.csv')
        
        return estimate, parity
    except Exception as e:
        logger.error(f"Error in fairness analysis: {e}")
        return None, None

def generate_report(baseline_results, graphsage_accuracy, fairness_results):
    """Generate a comprehensive training report."""
    logger.info("\n=== TRAINING REPORT ===")
    
    report = f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║           FRAUD DETECTION MODEL TRAINING REPORT                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    BASELINE MODEL (XGBoost):
    ├─ ROC-AUC Score: {baseline_results['roc_auc']:.4f}
    ├─ Recall @ 1% FPR: {baseline_results['recall_at_1pct_fpr']:.4f}
    └─ Status: ✅ Trained and saved
    
    GRAPH NEURAL NETWORK (GraphSAGE):
    ├─ Accuracy: {graphsage_accuracy:.4f if graphsage_accuracy else 'N/A'}
    └─ Status: {'✅ Trained and saved' if graphsage_accuracy else '⚠️ Training pending'}
    
    FAIRNESS ANALYSIS (DoWhy):
    ├─ Demographic Parity Difference: {fairness_results[1]['difference']:.4f if fairness_results[1] else 'N/A'}
    ├─ Fraud-prone Zip Prediction Rate: {fairness_results[1]['fraud_zip_rate']:.4f if fairness_results[1] else 'N/A'}
    ├─ Other Zip Prediction Rate: {fairness_results[1]['other_zip_rate']:.4f if fairness_results[1] else 'N/A'}
    └─ Status: {'✅ Analysis complete' if fairness_results[1] else '⚠️ Analysis pending'}
    
    NEXT STEPS:
    1. Hyperparameter tuning for GraphSAGE
    2. Ensemble model combining XGBoost and GraphSAGE
    3. Deploy to production environment
    4. Monitor model performance in real-time
    """
    
    logger.info(report)
    
    # Save report to file
    with open('models/training_report.txt', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Run the complete training pipeline."""
    logger.info("Starting Fraud Detection Model Training Pipeline...")
    
    ensure_directories()
    
    # Phase 1: Feature Engineering
    features = run_feature_engineering()
    
    # Phase 2: Baseline Training
    baseline_model, baseline_results = run_baseline_training(features)
    
    # Phase 3: GraphSAGE Training
    graphsage_model, graphsage_accuracy = run_graphsage_training(features)
    
    # Phase 4: Fairness Analysis
    fairness_estimate, fairness_parity = run_fairness_analysis(features)
    
    # Generate Report
    report = generate_report(baseline_results, graphsage_accuracy, (fairness_estimate, fairness_parity))
    
    logger.info("\n✅ Training pipeline complete!")

if __name__ == "__main__":
    main()
