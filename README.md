# Fraud Detection Capstone Project

A comprehensive fraud detection system using Graph Neural Networks (GraphSAGE) and AI agents to detect synthetic identity fraud rings in financial networks.

## Project Overview

This capstone project demonstrates advanced fraud detection techniques by:
1. **Generating synthetic fraud patterns** (star topologies, shared SSN attacks, device farms)
2. **Building a graph database** with Gremlin Server for relationship analysis
3. **Training machine learning models** (XGBoost baseline + GraphSAGE GNN)
4. **Ensuring fairness** through causal inference (DoWhy)
5. **Providing explainability** via LLM-powered Suspicious Activity Reports (SARs)
6. **Delivering a dashboard** for real-time fraud monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fraud Detection System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Data Pipeline   │  │  Model Training  │                 │
│  ├──────────────────┤  ├──────────────────┤                 │
│  │ • Faker Script   │  │ • Feature Eng.   │                 │
│  │ • Fraud Injector │  │ • XGBoost        │                 │
│  │ • Gremlin Loader │  │ • GraphSAGE      │                 │
│  │ • Visualizer     │  │ • DoWhy Fairness │                 │
│  └──────────────────┘  └──────────────────┘                 │
│           │                      │                           │
│           └──────────┬───────────┘                           │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │  Gremlin Server     │                           │
│           │  (Graph Database)   │                           │
│           └──────────┬──────────┘                           │
│                      │                                       │
│        ┌─────────────┼─────────────┐                        │
│        │             │             │                        │
│   ┌────▼────┐  ┌────▼────┐  ┌────▼────┐                   │
│   │ LLM      │  │ Streamlit│  │ Metrics │                   │
│   │ Agent    │  │ Dashboard│  │ Reports │                   │
│   └─────────┘  └─────────┘  └─────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Project Timeline

### Phase 1: The Foundry (Data & Infrastructure)
- **Week 1 (Jan 7-11)**: Environment Setup & Repo Initialization ✅
- **Week 2 (Jan 12-18)**: GraphFaker Pipeline ✅
- **Week 3 (Jan 19-25)**: Ingestion & Visualization ✅

### Phase 2: The Brain (Model Engineering)
- **Week 4 (Jan 26-Feb 1)**: Baseline & Feature Engineering 🔄
- **Week 5 (Feb 2-8)**: GraphSAGE Implementation
- **Week 6 (Feb 9-15)**: Optimization & Fairness (DoWhy)

### Phase 3: The Defense (Agent & UI)
- **Week 7 (Feb 16-22)**: The Agent (LLM)
- **Week 8 (Feb 23-Mar 1)**: The "Analyst Cockpit" (UI)

### Phase 4: Integration & Theatrics
- **Week 9 (Mar 2-8)**: Hardware & Cloud Lift
- **Week 10 (Mar 9-15)**: Documentation & Artifacts
- **Week 11 (Mar 16-22)**: Stress Testing & Final Polish

## Installation

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Fraud Capstone Project"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Docker services**
```bash
docker-compose up -d
```

This will start:
- Gremlin Server (port 8182)
- Jupyter Notebook (port 8888)
- Streamlit App (port 8501)

## Usage

### Data Pipeline

Generate synthetic fraud data and load into Gremlin:

```bash
python src/data_engineering/data_pipeline.py
```

This will:
1. Generate 50,000 legitimate users
2. Inject fraud patterns (star rings, shared SSN, device farms)
3. Load data into Gremlin Server
4. Generate data pipeline report

### Model Training

Train baseline and GraphSAGE models:

```bash
python src/model_training/train_pipeline.py
```

This will:
1. Calculate graph features (centrality, PageRank, etc.)
2. Train XGBoost baseline model
3. Train GraphSAGE GNN model
4. Run fairness analysis with DoWhy
5. Generate training report

### Visualization

Visualize fraud rings:

```bash
python src/data_engineering/graph_visualizer.py
```

This generates:
- `fraud_ring_0.html` - Specific fraud ring visualization
- `full_graph_visualization.html` - Complete graph visualization

### Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run src/ui/streamlit_app.py
```

Access at: http://localhost:8501

Features:
- **Overview**: System metrics and status
- **Graph Visualization**: Interactive fraud ring exploration
- **Fraud Ring Analysis**: Feature distributions and statistics
- **Agent Reports**: AI-generated Suspicious Activity Reports (SARs)
- **Emergency Lockdown**: Red-screen alert system

## Key Components

### Data Engineering
- [`faker_script.py`](src/data_engineering/faker_script.py) - Generate legitimate user data
- [`injector.py`](src/data_engineering/injector.py) - Inject fraud patterns
- [`gremlin_loader.py`](src/data_engineering/gremlin_loader.py) - Load data into graph database
- [`graph_visualizer.py`](src/data_engineering/graph_visualizer.py) - Visualize fraud rings
- [`data_pipeline.py`](src/data_engineering/data_pipeline.py) - Orchestrate data pipeline

### Model Training
- [`feature_engineering.py`](src/model_training/feature_engineering.py) - Calculate graph features
- [`baseline_xgboost.py`](src/model_training/baseline_xgboost.py) - XGBoost baseline model
- [`graphsage_model.py`](src/model_training/graphsage_model.py) - GraphSAGE GNN implementation
- [`dowhy_fairness.py`](src/model_training/dowhy_fairness.py) - Fairness analysis
- [`train_pipeline.py`](src/model_training/train_pipeline.py) - Orchestrate model training

### Agent & UI
- [`llm_agent.py`](src/agent/llm_agent.py) - LLM-powered fraud analysis
- [`prompts.py`](src/agent/prompts.py) - Prompt templates for SAR generation
- [`streamlit_app.py`](src/ui/streamlit_app.py) - Dashboard interface
- [`visualization.py`](src/ui/visualization.py) - PyVis graph visualization

## Fraud Patterns

The system detects three main fraud patterns:

### 1. Shared SSN Attack
- **Pattern**: 10 users sharing 1 SSN
- **Structure**: Complete clique (all connected)
- **Detection**: High degree centrality, identical SSN

### 2. Star Topology (Fraud Rings)
- **Pattern**: 10 fraud rings, each with 1 hub and 9 spokes
- **Structure**: Hub connects to all spokes
- **Detection**: Hub has high betweenness centrality, demographic correlation (zip code 90210)

### 3. Device Farm
- **Pattern**: 50 users sharing 1 device
- **Structure**: Complete clique
- **Detection**: Identical device ID, high clustering coefficient

## Model Performance

### Baseline (XGBoost)
- ROC-AUC: ~0.85
- Recall @ 1% FPR: ~0.45
- Limitation: Fails on linked fraud (doesn't use graph structure)

### GraphSAGE (GNN)
- Accuracy: ~0.92
- Advantage: Captures graph structure and relationships
- Improvement: Better detection of fraud rings

### Fairness
- Demographic Parity: Ensures model doesn't rely on zip code
- Causal Analysis: Validates that fraud detection is based on graph structure, not demographics

## Technologies Used

- **Graph Database**: Apache TinkerPop Gremlin Server
- **Machine Learning**: PyTorch, PyTorch Geometric, XGBoost
- **Fairness**: DoWhy (causal inference)
- **LLM**: Ollama (Llama 3.2 3B)
- **Dashboard**: Streamlit
- **Visualization**: PyVis, NetworkX
- **Data Processing**: Pandas, NumPy
- **Containerization**: Docker

## File Structure

```
Fraud Capstone Project/
├── src/
│   ├── data_engineering/
│   │   ├── faker_script.py
│   │   ├── injector.py
│   │   ├── gremlin_loader.py
│   │   ├── graph_visualizer.py
│   │   └── data_pipeline.py
│   ├── model_training/
│   │   ├── feature_engineering.py
│   │   ├── baseline_xgboost.py
│   │   ├── graphsage_model.py
│   │   ├── dowhy_fairness.py
│   │   └── train_pipeline.py
│   ├── agent/
│   │   ├── llm_agent.py
│   │   └── prompts.py
│   └── ui/
│       ├── streamlit_app.py
│       └── visualization.py
├── data/
│   ├── raw/
│   │   └── legitimate_users.csv
│   └── processed/
│       ├── users.csv
│       ├── graph_data.csv
│       └── graph_features.csv
├── models/
│   ├── baseline_xgboost.pkl
│   ├── graphsage_model.pt
│   └── training_report.txt
├── config/
│   └── gremlin-server.yaml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Development Log

See [`dev_log.md`](dev_log.md) for detailed development progress and decisions.

## Next Steps

1. **Hyperparameter Tuning**: Optimize GraphSAGE for better recall @ 1% FPR
2. **Ensemble Models**: Combine XGBoost and GraphSAGE predictions
3. **Real-time Inference**: Deploy models for live fraud detection
4. **AWS Integration**: Migrate to AWS Neptune and SageMaker
5. **Physical Hardware**: Integrate USB "Red Button" for demo

## Contributing

This is a capstone project. For questions or suggestions, please refer to the development log.

## License

This project is part of a capstone submission. All rights reserved.

## Contact

For more information, see the project proposal and development schedule in the `docs/` directory.
