import os
import openai

def check_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"OpenAI API Key found in environment variables: {api_key[:5]}...{api_key[-4:]}")
        try:
            client = openai.OpenAI(api_key=api_key) # Explicitly pass for this test
            models = client.models.list()
            print(f"Successfully connected to OpenAI API. Found {len(models.data)} models.")
            return True
        except openai.AuthenticationError as e:
            print(f"OpenAI API Key Authentication Error: {e}")
            print("Please ensure your API key is correct, active, and has credit/quota.")
            return False
        except openai.APIConnectionError as e:
            print(f"OpenAI API Connection Error: {e}")
            print("Could not connect to OpenAI. Check your network or OpenAI's status page.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while testing OpenAI API: {e}")
            return False
    else:
        print("Error: OPENAI_API_KEY environment variable not found.")
        print("Please set it before running the script. Example: export OPENAI_API_KEY='your_key_here'")
        return False
