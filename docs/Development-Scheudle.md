This is a **compressed "Crunch Mode" schedule**.

Your original proposal listed a 15-week timeline, but you only have **11 weeks** between today (Jan 7) and the deadline (Mar 25). I have accelerated the "Data Engineering" and "Model Training" phases to fit this deadline.

### **Phase 1: The Foundry (Data & Infrastructure)**

*Goal: Build the "Frankenstein" dataset and get the Local Graph DB running.*

* **Week 1: Jan 7 – Jan 11 (Short Week)**
* **Focus:** Environment Setup & Repo Initialization.
* **Tasks:**
* Set up local Docker environment (Gremlin Server, Jupyter, PyTorch).
* Initialize Git repository structure.
* Write basic `Faker` script to generate 1,000 legitimate "clean" user rows.


* **Deliverable:** A running local graph database accessible via Python.


* **Week 2: Jan 12 – Jan 18** ✅ Completed
* **Focus:** The "GraphFaker" Pipeline.
* **Tasks:**
* Code the `injector.py` script to create "Star Topologies" (Fraud Rings).
* Implement the "Shared Attribute Attack" (10 users, 1 SSN).
* Scale generation to 50,000 nodes/edges.
* Include demographic correlations in fraud patterns for later DoWhy bias testing.


* **Deliverable:** `graph_data.csv` containing hidden fraud patterns ready for ingestion.


* **Week 3: Jan 19 – Jan 25**
* **Focus:** Ingestion & Visualization.
* **Tasks:**
* Load CSV data into the local Gremlin Server.
* Use `NetworkX` or `PyVis` to visualize a specific fraud ring (sanity check).
* **Milestone:** Visually confirm you can see the "Frankenstein" structures in the graph.





---

### **Phase 2: The Brain (Model Engineering)**

*Goal: Train GraphSAGE and beat the XGBoost baseline.*

* **Week 4: Jan 26 – Feb 1**
* **Focus:** Baseline & Feature Engineering.
* **Tasks:**
* Train a standard XGBoost model on the tabular version of your data.
* Measure its failure (High False Negatives on linked fraud).
* Calculate Graph features (PageRank, Degree Centrality) to feed into GraphSAGE.


* **Deliverable:** A baseline report showing why tabular models fail.


* **Week 5: Feb 2 – Feb 8**
* **Focus:** GraphSAGE Implementation.
* **Tasks:**
* Implement `GraphSAGE` using PyTorch Geometric.
* Set up Inductive Learning (ensure it can predict on "new" nodes).
* Train on GPU (Local or Colab).


* **Deliverable:** A trained `.pt` model file.


* **Week 6: Feb 9 – Feb 15**
* **Focus:** Optimization & Fairness (DoWhy).
* **Tasks:**
* Tune hyperparameters to achieve **Recall @ 1% FPR**.
* Implement `dowhy` to test for bias by ensuring the model relies on graph structure, not demographics like zip codes.


* **Deliverable:** Model metrics report proving the "Lift" over XGBoost.



---

### **Phase 3: The Defense (Agent & UI)**

*Goal: Connect the AI Agent and build the "Matrix" Dashboard.*

* **Week 7: Feb 16 – Feb 22**
* **Focus:** The Agent (LLM).
* **Tasks:**
* Set up Ollama (Llama 3.2 3B) locally to prototype the Bedrock Agent.
* Write the Prompt Engineering templates for the "Suspicious Activity Report" (SAR).
* Feed graph neighbor data into the LLM context window.


* **Deliverable:** The system generates a text explanation for a specific Fraud Ring.


* **Week 8: Feb 23 – Mar 1**
* **Focus:** The "Analyst Cockpit" (UI).
* **Tasks:**
* Build the Streamlit Dashboard.
* Implement the 3D Graph visualization (Cosmograph/PyVis) inside the dashboard.
* Create the "Lockdown" red-screen state.


* **Deliverable:** A functional web interface running locally.



---

### **Phase 4: Integration & Theatrics**

*Goal: Physical props, Cloud deployment, and final polish.*

* **Week 9: Mar 2 – Mar 8**
* **Focus:** Hardware & Cloud Lift.
* **Tasks:**
* **Purchase/Program the USB "Red Button".**
* Write the Python script that triggers the "Bust-Out" injection when the button is pressed.
* *Optional:* Deploy to AWS Neptune/SageMaker (strictly for 1-2 days to test/record video).


* **Deliverable:** The physical button triggers the software alert.


* **Week 10: Mar 9 – Mar 15**
* **Focus:** Documentation & Artifacts.
* **Tasks:**
* Print the "Confidential" SAR reports.
* Write the "Case Study" PDF (ROI analysis).
* Record the Demo Video (Crucial backup in case live demo fails).


* **Deliverable:** Draft of Final Report and Demo Video.


* **Week 11: Mar 16 – Mar 22**
* **Focus:** Stress Testing & Final Polish.
* **Tasks:**
* Run the "Live Attack" sequence 50 times to ensure zero crashes.
* Final code cleanup and commenting.
* Pack the "Kit" (Laptop, Button, Papers).


* **Deliverable:** Project ready for turn-in.



### **Final Submission**

* **Due Date: Wednesday, March 25**
* Submit Code Repository.
* Submit Case Study PDF.
* Submit Demo Video link.