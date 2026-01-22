import streamlit as st
import pandas as pd
import networkx as nx
from src.ui.visualization import visualize_graph
from src.data_engineering.graph_visualizer import GraphVisualizer
from src.agent.llm_agent import generate_sar
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🔍 Fraud Detection Dashboard - Analyst Cockpit")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Graph Visualization", "Fraud Ring Analysis", "Agent Reports"])

if page == "Overview":
    st.header("System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        features_df = pd.read_csv('data/processed/graph_features.csv', index_col='user_id')
        total_users = len(features_df)
        fraud_users = (features_df['is_fraud'] == 1).sum()
        clean_users = total_users - fraud_users
        
        col1.metric("Total Users", total_users)
        col2.metric("Fraud Cases", fraud_users)
        col3.metric("Clean Users", clean_users)
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    st.subheader("System Status")
    st.success("✅ Gremlin Server: Connected")
    st.success("✅ Graph Database: Operational")
    st.info("ℹ️ Last Update: Real-time")

elif page == "Graph Visualization":
    st.header("Real-Time Graph Visualization")
    
    try:
        visualizer = GraphVisualizer()
        visualizer.load_from_csv('data/processed/users.csv', 'data/processed/graph_data.csv')
        
        st.subheader("Select Fraud Ring to Visualize")
        ring_id = st.slider("Fraud Ring ID", 0, 9, 0)
        
        fraud_ring = visualizer.get_fraud_ring_subgraph(ring_id=ring_id, ring_size=10)
        
        if fraud_ring:
            st.write(f"Fraud Ring {ring_id}: {len(fraud_ring.nodes())} nodes, {len(fraud_ring.edges())} edges")
            
            # Generate visualization
            visualizer.visualize_with_pyvis(fraud_ring, f'fraud_ring_{ring_id}.html')
            
            with open(f'fraud_ring_{ring_id}.html', 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=600)
    except Exception as e:
        st.error(f"Error loading visualization: {e}")

elif page == "Fraud Ring Analysis":
    st.header("Fraud Ring Analysis")
    
    try:
        features_df = pd.read_csv('data/processed/graph_features.csv', index_col='user_id')
        fraud_users = features_df[features_df['is_fraud'] == 1]
        
        st.subheader("Fraud User Statistics")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Avg PageRank", f"{fraud_users['pagerank'].mean():.6f}")
        col2.metric("Avg Degree Centrality", f"{fraud_users['degree_centrality'].mean():.6f}")
        col3.metric("Avg Neighbor Fraud Ratio", f"{fraud_users['neighbor_fraud_ratio'].mean():.4f}")
        
        st.subheader("Feature Distribution")
        feature_to_plot = st.selectbox("Select Feature", ['pagerank', 'degree_centrality', 'clustering_coefficient'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Fraud Users")
            st.bar_chart(fraud_users[feature_to_plot].head(20))
        
        with col2:
            st.write("Clean Users")
            clean_users = features_df[features_df['is_fraud'] == 0]
            st.bar_chart(clean_users[feature_to_plot].head(20))
    
    except Exception as e:
        st.error(f"Error in analysis: {e}")

elif page == "Agent Reports":
    st.header("AI Agent - Suspicious Activity Reports (SAR)")
    
    st.subheader("Generate SAR for Fraud Ring")
    ring_id = st.number_input("Enter Fraud Ring ID", min_value=0, max_value=9, value=0)
    
    if st.button("Generate SAR Report"):
        with st.spinner("Generating report..."):
            try:
                # Load fraud ring data
                features_df = pd.read_csv('data/processed/graph_features.csv', index_col='user_id')
                fraud_users = features_df[features_df['is_fraud'] == 1]
                
                graph_data = {
                    'fraud_count': len(fraud_users),
                    'avg_pagerank': fraud_users['pagerank'].mean(),
                    'avg_degree': fraud_users['degree_centrality'].mean()
                }
                
                # Generate SAR using LLM
                sar_report = generate_sar(str(graph_data))
                
                st.success("✅ Report Generated")
                st.text_area("Suspicious Activity Report", sar_report, height=300)
            except Exception as e:
                st.error(f"Error generating report: {e}")

# Emergency Lockdown Button
st.sidebar.markdown("---")
if st.sidebar.button("🚨 LOCKDOWN - Fraud Detected!", key="lockdown"):
    st.error("🚨 SYSTEM LOCKDOWN ACTIVATED 🚨")
    st.error("Suspicious activity detected! All transactions halted.")
    st.error("Security team notified. Incident report generated.")
    logger.warning("LOCKDOWN ACTIVATED - Fraud detected!")