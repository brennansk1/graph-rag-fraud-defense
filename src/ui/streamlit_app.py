import streamlit as st

st.title("Fraud Detection Dashboard")

st.header("Real-Time Graph Visualization")
# Add visualization component

st.header("Transaction Simulator")
# Add input for simulating transaction

st.header("Agent Log")
# Display logs

if st.button("Simulate Fraud Attack"):
    st.error("LOCKDOWN: Fraud Detected!")
    # Trigger logic