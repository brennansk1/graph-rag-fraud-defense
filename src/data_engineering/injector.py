import pandas as pd
import networkx as nx

# Code to inject fraud rings

def inject_fraud_rings(df):
    # Add logic for shared SSN, device farm, etc.
    # Placeholder
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/legitimate_users.csv')
    df_fraud = inject_fraud_rings(df)
    df_fraud.to_csv('data/processed/graph_data.csv', index=False)
    print("Injected fraud patterns.")