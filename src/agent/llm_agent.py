import ollama
import logging
from src.agent.prompts import SAR_PROMPT, EXPLANATION_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_llm(prompt, model='llama2'):
    """Query the local LLM (Ollama) with a prompt."""
    try:
        logger.info(f"Querying LLM with model: {model}")
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return f"Error: {e}"

def generate_sar(graph_data, model='llama2'):
    """Generate a Suspicious Activity Report (SAR) based on graph data."""
    logger.info("Generating Suspicious Activity Report...")
    
    prompt = SAR_PROMPT.format(graph_data=str(graph_data))
    sar_report = query_llm(prompt, model=model)
    
    logger.info("SAR generated successfully")
    return sar_report

def explain_fraud_detection(result, model='llama2'):
    """Explain a fraud detection result using the LLM."""
    logger.info("Generating fraud detection explanation...")
    
    prompt = EXPLANATION_PROMPT.format(result=str(result))
    explanation = query_llm(prompt, model=model)
    
    logger.info("Explanation generated successfully")
    return explanation

def analyze_fraud_ring(ring_data, model='llama2'):
    """Analyze a specific fraud ring and provide insights."""
    logger.info("Analyzing fraud ring...")
    
    analysis_prompt = f"""
    Analyze the following fraud ring data and provide:
    1. Summary of the fraud pattern
    2. Key indicators of fraud
    3. Recommended actions
    
    Fraud Ring Data:
    {ring_data}
    """
    
    analysis = query_llm(analysis_prompt, model=model)
    
    logger.info("Fraud ring analysis complete")
    return analysis

if __name__ == "__main__":
    # Test the LLM agent
    test_data = {
        'nodes': 10,
        'edges': 45,
        'fraud_nodes': 10,
        'pattern': 'Star topology with shared SSN'
    }
    
    logger.info("Testing LLM Agent...")
    sar = generate_sar(test_data)
    logger.info(f"Generated SAR:\n{sar}")