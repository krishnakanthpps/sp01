import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Load Azure OpenAI settings
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Azure OpenAI API URL
api_url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"

def get_design_tasks(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    # Define the request payload
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that generates a clear, structured list of web design tasks based on user prompts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
#        "max_tokens": 500,
        "temperature": 0.5
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        tasks = response.json()["choices"][0]["message"]["content"]
        return tasks
    else:
        raise Exception(f"Error calling Azure OpenAI API: {response.text}")

# Example usage:
if __name__ == "__main__":
    user_prompt = input("Describe the website you want to design: ")

    try:
        tasks = get_design_tasks(user_prompt)
        print("\n✅ Website Design Tasks:\n")
        print(tasks)
    except Exception as e:
        print(e)
