import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_xgboost(X, y, test_size=0.2, random_state=42):
    """Train XGBoost baseline model."""
    logger.info(f"Training XGBoost with {len(X)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    logger.info("\n=== XGBoost Baseline Results ===")
    logger.info(classification_report(y_test, y_pred))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    logger.info(f"True Negatives: {tn}, False Positives: {fp}")
    logger.info(f"False Negatives: {fn}, True Positives: {tp}")
    
    # False Positive Rate at different thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fpr_1pct_idx = np.argmin(np.abs(fpr - 0.01))
    recall_at_1pct_fpr = tpr[fpr_1pct_idx]
    logger.info(f"Recall @ 1% FPR: {recall_at_1pct_fpr:.4f}")
    
    return model, {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc': roc_auc,
        'recall_at_1pct_fpr': recall_at_1pct_fpr
    }

def save_model(model, path='models/baseline_xgboost.pkl'):
    """Save trained model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")

def load_model(path='models/baseline_xgboost.pkl'):
    """Load trained model from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model

if __name__ == "__main__":
    # Load features
    features = pd.read_csv('data/processed/graph_features.csv', index_col='user_id')
    
    X = features.drop('is_fraud', axis=1)
    y = features['is_fraud']
    
    model, results = train_xgboost(X, y)
    save_model(model)