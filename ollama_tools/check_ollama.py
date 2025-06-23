import ollama 

OLLAMA_MODEL_NAME = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama API endpoint

def check_ollama_availability(model_name=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL):
    try:
        client = ollama.Client(host=base_url)
        client.list()
        print(f"Successfully connected to Ollama at {base_url}")
        print(f"{client.list()}")
        available_models = [m['model'] for m in client.list()['models']]
        if model_name in available_models:
            print(f"Ollama model '{model_name}' is available.")
            return True
        else:
            print(f"Error: Ollama model '{model_name}' not found. Available models: {available_models}")
            print(f"Please pull the model using: ollama pull {model_name}")
            return False
    except Exception as e:
        print(f"Error connecting to Ollama or listing models: {e}")
        print(f"Please ensure Ollama is running and accessible at {base_url}")
        return False
    

check_ollama_availability(OLLAMA_MODEL_NAME, OLLAMA_BASE_URL)