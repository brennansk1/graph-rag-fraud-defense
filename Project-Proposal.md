CAPSTONE PROJECT PROPOSAL
Utah Valley University | Department of Computer Science
TO: [Instructor Name]
FROM: Brennan
DATE: [Current Date]
SUBJECT: Project Proposal: Deep Graph Intelligence (GraphRAG Defense System)
1. Abstract
The financial services industry faces a projected $23 billion threat by 2030: Synthetic Identity Fraud (SIF). Unlike traditional third-party fraud, SIF involves "Frankenstein" identities—combinations of real (often stolen) Social Security numbers and fake PII—that hide in the structural relationships between entities, making them invisible to standard tabular models (XGBoost).
This project moves beyond theoretical analysis to build a fully functional, "Human-in-the-Loop" Defense System. I propose a Graph Retrieval-Augmented Generation (GraphRAG) architecture designed to bridge the gap between high-performance Deep Learning and regulatory compliance (SR 11-7).
Leveraging a hybrid Cloud/Local architecture tailored for the Utah "Silicon Slopes" fintech ecosystem (SoFi, Galileo, Goldman Sachs), this system will:
Model Topology: Use Amazon Neptune and GraphSAGE (Inductive Learning) to detect "Star Topologies" (fraud rings) in real-time (<200ms).
Validate Causality: Utilize Fairness Engineering (via the dowhy library) to distinguish between spurious correlations (bias) and true causal drivers of risk.
Automate Investigation: Deploy an Amazon Bedrock Agent to autonomously explain high-risk cases, reducing manual review time by synthesizing technical graph data into natural language Suspicious Activity Reports (SARs).
2. Problem Statement & Business Context
Synthetic Identity Fraud costs lenders billions, yet current defenses create a "Recall vs. Precision" paradox.
The "Frankenstein" Problem: Fraudsters stitch together real SSNs with fake names. Because the SSN is real, it passes credit bureau checks. Because the person doesn't exist, no one calls to complain. Traditional models fail to detect this because they treat rows independently, missing the shared device fingerprints or SSN collisions common in fraud rings.
Operational Bottlenecks: Human analysts often take 20-40 minutes to cross-reference databases to find hidden links.
The Business Goal: Build a system that optimizes Recall @ 1% False Positive Rate (FPR). This metric is crucial for minimizing customer friction while catching maximum fraud. The system aims to simulate a $2M annual reduction in fraud losses through improved detection of "bust-out" schemes.
3. Project Objectives
3.1 Data Science: Inductive Learning & Fairness
Topological Analysis: Profile graph statistics (Eigenvector Centrality, Clustering Coefficients) to distinguish "organic" social clusters from "synthetic" fraud rings.
Model Implementation: Implement GraphSAGE (Sample and Aggregate) rather than a standard GCN. This choice is critical for production environments (like Galileo/SoFi) because GraphSAGE is inductive—it can generate embeddings for new nodes (applicants) instantly without retraining the entire model.
Causal Validation & Compliance: Implement a "Counterfactual Testing" pipeline using dowhy. I will estimate the Average Treatment Effect (ATE) of high-degree nodes on fraud probability to prove that the model relies on behavioral structure (risk) rather than demographics (bias/redlining).
3.2 Engineering: The "Agentic" Architecture
GraphRAG Schema: Architect a Neptune schema optimized for LLM traversal, enabling the Agent to query "2-hop neighborhoods" dynamically.
Automated Reason Codes: Build an Amazon Bedrock Agent (Claude 3.5 Sonnet) that triggers post-inference. While the GNN provides a sub-second score, the Agent performs a deep-dive investigation to generate a natural language explanation (e.g., "High Risk: User shares SSN with 5 unrelated identities").
Hybrid MLOps: Implement a cost-optimized workflow using local Docker containers for dev/test and AWS Serverless for production.
4. Technical Architecture & Methodology
4.1 Data Strategy: The "GraphFaker" Pipeline
To move beyond generic graph generation and simulate realistic "Frankenstein" attacks, I will implement a custom data generation pipeline:
Background Traffic: Generate 50,000 legitimate users with unique PII using Faker and NetworkX.
SIF Injection: Inject specific "Fraud Rings" where multiple identities share hidden attributes (collisions).
The Shared Attribute Attack: 10 identities with different names but the same SSN.
The Device Farm: 50 identities connected to a single unique Device ID.
Temporal Logic: Transactions will be timestamped to simulate the "incubation" period of synthetic identities before the "bust-out."
4.2 The "Hybrid" Tech Stack
Optimized for zero-cost development and high-value cloud deployment.
Layer
Local Dev (Docker/Homelab)
Cloud Prod (AWS)
Role
Data
Gremlin Server
Amazon Neptune
Stores the Knowledge Graph.
Compute
PyTorch Geometric
Amazon SageMaker
Trains/Hosts the GraphSAGE inference endpoint.
Causality
DoWhy (Python)
Lambda Layers
Performs fairness/counterfactual tests.
Agent
Ollama (Llama 3)
Amazon Bedrock
Reasoning engine (Claude 3.5 Sonnet) for GraphRAG.
Ops
MLflow
CloudWatch
Tracks model Recall/Precision and system latency.

4.3 The "Analyst Cockpit" (Expo Deliverable)
To demonstrate the system's real-world utility at the CET Expo, the user interface will mimic a live Security Operations Center (SOC) across a multi-monitor setup:
The "Matrix" View (Real-Time Graph): A 3D, rotating visualization (using PyVis/Cosmograph) of the customer network. Normal users appear as isolated dots; fraud rings appear as glowing, pulsing "Star Clusters."
The "Kill Switch" (Streamlit Dashboard): A control panel allowing the user to simulate a live transaction.
Green State: "Transaction Approved."
Red State (The Attack): When a fraud ring is detected, the dashboard triggers a "LOCKDOWN" alert, flashing red and spiking the probability score to 99%.
The "Agent" Log: A scrolling terminal window displaying the Bedrock Agent's live reasoning (e.g., "Scanning neighbor nodes... Warning: Shared SSN detected... Generating SAR... Report Filed.").
4.4 Physical Artifacts (Theatrics)
To bridge the gap between software and physical engineering, this project includes physical deliverables for the final presentation:
The Red Button: A physical USB trigger on the demo table. When pressed by a judge, it executes the Python script to inject a "Synthetic Identity" into the live database, triggering the system alarms.
Printed SARs: High-quality, redacted "Confidential" folders containing the AI-generated Suspicious Activity Reports, proving the system creates regulatory-grade legal documents.
5. Strategic Justification & Market Analysis
5.1 Alignment with Utah "Silicon Slopes" Fintech
This project is reverse-engineered from the specific hiring requirements of the local market:
SoFi & Galileo (The Aggressors): These firms require "Inductive Learning" capabilities. By choosing GraphSAGE, this project demonstrates an understanding of low-latency transaction processing (sub-200ms) required by high-volume issuers.
Goldman Sachs & Mastercard (The Fortresses): These firms prioritize "Model Risk Management" (SR 11-7). The inclusion of Causal Inference (Counterfactual Testing) addresses their need for models that are robust, explainable, and compliant with Fair Lending laws.
MX Technologies (The Data Enablers): The focus on Entity Resolution and GraphRAG aligns with their roadmap for "Enhanced Data Context."
5.2 Academic Alignment
CS 4800 (Capstone): Full-stack system integration.
CS 4710 (Machine Learning II): Deep Learning (GNNs) and advanced evaluation.
STAT 4000 (Applied Stats): Application of Causal Inference and hypothesis testing.
6. Project Timeline (15 Weeks)
Weeks 1-4: Data Engineering ("The Foundry"). Develop the GraphFaker pipeline. Generate 1M nodes with injected SIF patterns (SSN collisions, address clusters). Clean data for Neptune ingestion.
Weeks 5-8: Model Training ("The Brain"). Train XGBoost baseline (tabular) and GraphSAGE (graph). Conduct comparative analysis to prove GraphSAGE's superiority in detecting fraud rings. Optimize for Recall @ 1% FPR.
Weeks 9-11: Agentic Build ("The Defense"). Connect Bedrock Agent to Neptune. Refine Prompt Engineering for SAR generation. Implement dowhy fairness checks.
Week 12: The "War Room" Build. Develop the multi-monitor signaling logic. Ensure the "Red Screen" flash is visceral and instant.
Week 13: Prop Fabrication. Print the fake FBI reports; configure the physical USB trigger button.
Week 14: Stress Testing. Run the "Live Attack" script 50+ times to ensure zero crashes during the demo.
Week 15: Expo Demo Day. Record a demo video focusing on the investigation workflow. Create a "Case Study PDF" tailored to recruiters, highlighting the $2M simulated ROI.
7. Career Leverage Strategy
7.1 The Resume Strategy
This project allows for specific, high-value bullet points optimized for Applicant Tracking Systems (ATS):
"Engineered a GraphRAG defense system on AWS, utilizing GraphSAGE to detect Synthetic Identity Fraud rings with sub-second inference."
"Optimized fraud detection for Recall @ 1% False Positive Rate, utilizing Causal Inference (DoWhy) to validate risk factors and ensure Fair Lending compliance."
"Deployed an Agentic AI workflow using Amazon Bedrock to automate Suspicious Activity Report (SAR) generation, simulating a 40% reduction in manual review time."
7.2 The Portfolio Pitch
Instead of a simple code repository, the final deliverable will include a "Case Study" white paper structured as: Business Problem (SIF) -> Solution (GraphRAG) -> Financial Impact (ROI).
8. Addendum: Interview Defense Cheat Sheet
(Self-Reference for Defense Preparation)
Q: Why Causal Inference? Isn't high accuracy enough?
A: "In banking, accuracy on historical data isn't enough because history contains bias. If my model flags a zip code because it historically had high fraud, that's a Fair Lending violation (Redlining). I used dowhy to perform Counterfactual Tests (e.g., 'If I change the zip code but keep the graph connections, does the score change?'), ensuring the model reacts to behavioral risk, not demographics."
Q: Why GraphSAGE over GCN?
A: "GCNs are transductive—they require the whole graph during training. Fraud is dynamic; new nodes appear every second. GraphSAGE is inductive; it learns how to aggregate neighbors, so it can predict on new nodes it has never seen before without retraining. That is critical for a live banking environment like Galileo or SoFi."
Q: How does this help with the 'Recall vs. Precision' paradox?
A: "Traditional models lower thresholds to catch more fraud, which spikes False Positives and insults legitimate customers. By using Graph topology, I can find the structural signature of a fraud ring. This allows me to keep high precision on normal transactions while maintaining high recall on the complex, hidden fraud rings that tabular models miss."
Q: Is this real-time?
A: "It operates on a split architecture. The GraphSAGE inference is Synchronous (<200ms) to block the transaction immediately. The Agentic Investigation is Asynchronous, triggered post-flag to help the analyst understand why it was blocked without digging through SQL tables."
Student Signature: ____________________ Date: ____________________

