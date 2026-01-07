SAR_PROMPT = """
Generate a Suspicious Activity Report (SAR) based on the following graph data indicating potential Synthetic Identity Fraud:

Graph Data: {graph_data}

Include:
- Summary of suspicious patterns
- Involved entities
- Recommended actions
"""

EXPLANATION_PROMPT = """
Explain the following fraud detection result: {result}
"""