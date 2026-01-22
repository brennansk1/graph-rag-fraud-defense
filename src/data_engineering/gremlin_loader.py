"""
Gremlin Loader: Load CSV data into the local Gremlin Server.
"""

from gremlin_python.driver import client
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GremlinLoader:
    def __init__(self, host='localhost', port=8182):
        """Initialize connection to Gremlin Server."""
        self.host = host
        self.port = port
        self.client = client.Client(f'ws://{host}:{port}/gremlin', 'g')
        logger.info(f"Connected to Gremlin Server at {host}:{port}")
    
    def load_users(self, csv_path):
        """Load users from CSV into the graph."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} users from {csv_path}")
        
        for idx, row in df.iterrows():
            user_id = int(row['user_id'])
            name = row['name']
            ssn = row['ssn']
            email = row['email']
            phone = row['phone']
            is_fraud = int(row['is_fraud'])
            device_id = row['device_id']
            zip_code = row['zip_code']
            
            # Gremlin query to add vertex
            query = f"""
            g.addV('User')
                .property('user_id', {user_id})
                .property('name', '{name}')
                .property('ssn', '{ssn}')
                .property('email', '{email}')
                .property('phone', '{phone}')
                .property('is_fraud', {is_fraud})
                .property('device_id', '{device_id}')
                .property('zip_code', '{zip_code}')
            """
            try:
                self.client.submit(query).all().result()
            except Exception as e:
                logger.error(f"Error loading user {user_id}: {e}")
        
        logger.info("Users loaded successfully")
    
    def load_edges(self, csv_path):
        """Load edges from CSV into the graph."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} edges from {csv_path}")
        
        for idx, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_type = row['type']
            
            # Gremlin query to add edge
            query = f"""
            g.V().has('user_id', {source})
                .addE('{edge_type}')
                .to(g.V().has('user_id', {target}))
            """
            try:
                self.client.submit(query).all().result()
            except Exception as e:
                logger.error(f"Error loading edge {source}->{target}: {e}")
        
        logger.info("Edges loaded successfully")
    
    def close(self):
        """Close the connection to Gremlin Server."""
        self.client.close()
        logger.info("Connection closed")

if __name__ == "__main__":
    loader = GremlinLoader()
    try:
        loader.load_users('data/processed/users.csv')
        loader.load_edges('data/processed/graph_data.csv')
        logger.info("Data ingestion complete!")
    finally:
        loader.close()
