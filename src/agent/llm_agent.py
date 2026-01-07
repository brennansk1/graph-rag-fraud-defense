import ollama

def query_llm(prompt):
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def generate_sar(graph_data):
    prompt = f"Generate a Suspicious Activity Report for the following graph data: {graph_data}"
    return query_llm(prompt)