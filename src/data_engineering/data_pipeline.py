"""
Data Pipeline: Orchestrate data generation, fraud injection, and ingestion.
"""

import logging
import os
from src.data_engineering.faker_script import generate_users
from src.data_engineering.injector import inject_fraud_rings
from src.data_engineering.gremlin_loader import GremlinLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    logger.info("Directories ensured")

def run_data_generation():
    """Generate legitimate user data."""
    logger.info("\n=== PHASE 1: Data Generation ===")
    
    logger.info("Generating 50,000 legitimate users...")
    df = generate_users(n=50000)
    df.to_csv('data/raw/legitimate_users.csv', index=False)
    
    logger.info(f"Generated {len(df)} users")
    return df

def run_fraud_injection(df):
    """Inject fraud patterns into the dataset."""
    logger.info("\n=== PHASE 2: Fraud Injection ===")
    
    logger.info("Injecting fraud patterns...")
    df_fraud = inject_fraud_rings(df)
    
    logger.info(f"Fraud injection complete: {len(df_fraud)} users processed")
    return df_fraud

def run_gremlin_ingestion():
    """Load data into Gremlin Server."""
    logger.info("\n=== PHASE 3: Gremlin Ingestion ===")
    
    try:
        loader = GremlinLoader(host='localhost', port=8182)
        
        logger.info("Loading users into Gremlin...")
        loader.load_users('data/processed/users.csv')
        
        logger.info("Loading edges into Gremlin...")
        loader.load_edges('data/processed/graph_data.csv')
        
        loader.close()
        logger.info("✅ Gremlin ingestion complete")
        return True
    except Exception as e:
        logger.error(f"Error connecting to Gremlin Server: {e}")
        logger.warning("Gremlin Server may not be running. Continuing with local processing...")
        return False

def generate_report():
    """Generate a data pipeline report."""
    logger.info("\n=== DATA PIPELINE REPORT ===")
    
    import pandas as pd
    
    try:
        users_df = pd.read_csv('data/processed/users.csv')
        edges_df = pd.read_csv('data/processed/graph_data.csv')
        
        fraud_count = (users_df['is_fraud'] == 1).sum()
        clean_count = (users_df['is_fraud'] == 0).sum()
        
        report = f"""
        ╔════════════════════════════════════════════════════════════════╗
        ║              FRAUD CAPSTONE DATA PIPELINE REPORT               ║
        ╚════════════════════════════════════════════════════════════════╝
        
        DATA GENERATION:
        ├─ Total Users: {len(users_df):,}
        ├─ Fraud Users: {fraud_count:,} ({fraud_count/len(users_df)*100:.2f}%)
        ├─ Clean Users: {clean_count:,} ({clean_count/len(users_df)*100:.2f}%)
        └─ Status: ✅ Complete
        
        FRAUD PATTERNS INJECTED:
        ├─ Shared SSN Attack: 10 users with 1 SSN (clique)
        ├─ Star Topologies: 10 fraud rings (1 hub + 9 spokes each)
        ├─ Device Farm: 50 users sharing 1 device (clique)
        └─ Demographic Correlation: Fraud users in zip code 90210
        
        GRAPH STRUCTURE:
        ├─ Total Nodes: {len(users_df):,}
        ├─ Total Edges: {len(edges_df):,}
        ├─ Edge Types: {edges_df['type'].unique().tolist()}
        └─ Status: ✅ Complete
        
        DATA FILES:
        ├─ data/raw/legitimate_users.csv (50,000 clean users)
        ├─ data/processed/users.csv (50,000 users with fraud labels)
        ├─ data/processed/graph_data.csv ({len(edges_df):,} edges)
        └─ Status: ✅ Ready for model training
        
        NEXT STEPS:
        1. Feature engineering (graph centrality measures)
        2. Baseline model training (XGBoost)
        3. GraphSAGE model training
        4. Fairness analysis (DoWhy)
        """
        
        logger.info(report)
        
        # Save report to file
        with open('data/processed/data_pipeline_report.txt', 'w') as f:
            f.write(report)
        
        return report
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None

def main():
    """Run the complete data pipeline."""
    logger.info("Starting Fraud Detection Data Pipeline...")
    
    ensure_directories()
    
    # Phase 1: Data Generation
    df = run_data_generation()
    
    # Phase 2: Fraud Injection
    df_fraud = run_fraud_injection(df)
    
    # Phase 3: Gremlin Ingestion (optional)
    gremlin_success = run_gremlin_ingestion()
    
    # Generate Report
    report = generate_report()
    
    logger.info("\n✅ Data pipeline complete!")

if __name__ == "__main__":
    main()
