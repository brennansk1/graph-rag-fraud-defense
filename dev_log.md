# Development Log

## Week 1: Environment Setup (Jan 7 – Jan 11)

- Initialized Git repository structure.
- Set up local Docker environment (Gremlin Server, Jupyter, PyTorch).
- Created basic Faker script to generate 1,000 legitimate user rows.

## Week 2: GraphFaker Pipeline (Jan 12 – Jan 18)

- Modified faker_script.py to generate 50,000 users instead of 1,000.
- Implemented injector.py with fraud patterns:
  - Shared SSN attack: 10 users sharing 1 SSN, connected in a clique.
  - Star topologies: 10 fraud rings, each with 1 hub and 9 spokes.
  - Device farm: 50 users sharing 1 device ID, connected in a clique.
- Added demographic correlations: Fraud users in star rings have zip code '90210'.
- Generated graph_data.csv with 1,405 edges and users.csv with 50,000 nodes, ready for Gremlin ingestion.

## Week 3: Ingestion & Visualization (Jan 19 – Jan 25)

- Created gremlin_loader.py to load CSV data into Gremlin Server.
- Implemented graph_visualizer.py with PyVis for fraud ring visualization.
- Created feature_engineering.py to calculate graph features:
  - Degree centrality, PageRank, clustering coefficient
  - Betweenness and closeness centrality
  - Connected component size, neighbor fraud ratio
- Enhanced baseline_xgboost.py with comprehensive metrics (ROC-AUC, Recall @ 1% FPR).
- Implemented full GraphSAGE model with training and evaluation.
- Enhanced dowhy_fairness.py with demographic parity analysis.
- Updated streamlit_app.py with full "Analyst Cockpit" dashboard:
  - Overview page with system metrics
  - Graph visualization page with fraud ring selection
  - Fraud ring analysis with feature distributions
  - Agent reports page for SAR generation
  - Emergency lockdown button
- Enhanced llm_agent.py with SAR generation, fraud detection explanation, and fraud ring analysis.

## Decision: LLM Model Selection for Agent Development

Date: January 7, 2026

Decided to use Ollama with Llama 3.2 3B (Instruct) model for local development on the laptop before migrating to AWS Bedrock.

Reasoning:
- Best balance for small models
- Performance: 15-25 tokens/sec
- RAM usage: ~2.5 GB
- Use case: Rapid prototyping of Agent logic and SAR reports.