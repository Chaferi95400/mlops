import requests

BASE_URL = 'http://localhost:8082'

input_text = "Once upon a time"

payload = {"text": input_text}

response = requests.post(f'{BASE_URL}/generate', json=payload)

if response.status_code == 200:
    generated_text = response.json()['generated_text']
    print("Generated Text:")
    print(generated_text)
else:
    print("Error:", response.status_code)
    print(response.text)
