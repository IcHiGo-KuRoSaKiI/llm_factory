import requests
import json

# Define the JSON schema
json_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "component_info",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "component_name": {"type": "string"},
                "description": {"type": "string"},
                "tech_stack": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["component_name", "description", "tech_stack", "dependencies"]
        }
    }
}

# Define the messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Describe a microservice for user authentication."}
]

# Prepare the payload
payload = {
    # "model": "your-model-name",  # Replace with your model's name
    "messages": messages,
    "temperature": 0.7,
    "max_tokens": 500,
    "response_format": json_schema
}

# Send the request to LM Studio
response = requests.post("", json=payload)

# Parse and print the response
if response.ok:
    content = response.json()["choices"][0]["message"]["content"]
    try:
        structured_output = json.loads(content)
        print(json.dumps(structured_output, indent=2))
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw content:", content)
else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)
