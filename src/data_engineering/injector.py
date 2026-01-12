import pandas as pd
import networkx as nx
import random
from faker import Faker

fake = Faker()

def inject_fraud_rings(df):
    df = df.copy()
    df['user_id'] = range(len(df))
    df['is_fraud'] = 0
    df['device_id'] = [fake.uuid4() for _ in range(len(df))]

    # Extract zip code from address
    def extract_zip(address):
        parts = address.split()
        for part in parts:
            if len(part) == 5 and part.isdigit():
                return part
        return '00000'
    df['zip_code'] = df['address'].apply(extract_zip)

    edges = []

    # Shared SSN attack: 10 users with same SSN
    shared_ssn_users = random.sample(list(df['user_id']), 10)
    shared_ssn = fake.ssn()
    for uid in shared_ssn_users:
        df.loc[df['user_id'] == uid, 'ssn'] = shared_ssn
        df.loc[df['user_id'] == uid, 'is_fraud'] = 1
        # Add edges between them (clique)
        for other_uid in shared_ssn_users:
            if uid != other_uid:
                edges.append({'source': uid, 'target': other_uid, 'type': 'shared_ssn'})

    # Star topologies: 10 fraud rings, each with 1 hub and 9 spokes
    for _ in range(10):
        available_users = [uid for uid in df['user_id'] if df.loc[df['user_id'] == uid, 'is_fraud'].values[0] == 0]
        if len(available_users) < 10:
            break
        ring_users = random.sample(available_users, 10)
        hub = ring_users[0]
        spokes = ring_users[1:]
        df.loc[df['user_id'].isin(ring_users), 'is_fraud'] = 1
        # Demographic correlation: set zip to a fraud-prone zip
        fraud_zip = '90210'  # e.g., Beverly Hills, but for fraud
        df.loc[df['user_id'].isin(ring_users), 'zip_code'] = fraud_zip
        for spoke in spokes:
            edges.append({'source': hub, 'target': spoke, 'type': 'fraud_ring'})

    # Device farm: 50 users sharing 1 device
    available_users = [uid for uid in df['user_id'] if df.loc[df['user_id'] == uid, 'is_fraud'].values[0] == 0]
    if len(available_users) >= 50:
        device_users = random.sample(available_users, 50)
        device_id = fake.uuid4()
        df.loc[df['user_id'].isin(device_users), 'device_id'] = device_id
        df.loc[df['user_id'].isin(device_users), 'is_fraud'] = 1
        # Add edges between them
        for i in range(len(device_users)):
            for j in range(i+1, len(device_users)):
                edges.append({'source': device_users[i], 'target': device_users[j], 'type': 'device_farm'})

    # Save users
    df.to_csv('data/processed/users.csv', index=False)

    # Save edges
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv('data/processed/graph_data.csv', index=False)

    print(f"Injected fraud patterns. Users: {len(df)}, Edges: {len(edges)}")
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/legitimate_users.csv')
    df_fraud = inject_fraud_rings(df)
    print("Fraud injection complete.")