import requests
import json

def test_tool_calling():
    url = "http://localhost:8000/v1/chat/completions"
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when necessary."},
        {"role": "user", "content": "Use the weather tool to find the weather in New York."}
    ]
    
    data = {
        "model": "RedHatAI/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto"
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            choice = result['choices'][0]
            if choice['message'].get('tool_calls'):
                print("SUCCESS: Model correctly identified the tool call!")
                print(json.dumps(choice['message']['tool_calls'], indent=2))
            else:
                print("FAILURE: Model did not use the tool. It said:")
                print(choice['message']['content'])
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_tool_calling()
