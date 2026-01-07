# Fraud Capstone Project: GraphRAG Defense System

## Overview

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) architecture to detect Synthetic Identity Fraud (SIF) in financial services. It uses GraphSAGE for inductive learning on graph topologies, DoWhy for causal fairness validation, and an LLM agent for automated investigation reports.

## Technologies

- Python
- PyTorch Geometric
- DoWhy
- Streamlit
- Docker
- Gremlin Server
- Ollama/Llama 3

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Docker: `docker-compose up`
4. Run the Streamlit app: `streamlit run src/ui/streamlit_app.py`

## Project Structure

- `src/`: Source code
  - `data_engineering/`: Data generation and processing
  - `model_training/`: Model training scripts
  - `agent/`: LLM agent code
  - `ui/`: User interface
- `data/`: Generated and processed data
- `models/`: Trained models
- `notebooks/`: Jupyter notebooks
- `tests/`: Unit tests
- `docs/`: Documentation

## Schedule

See Development-Schedule.md for the 11-week plan.

## License

[Add license if needed]