import requests

BASE_URL = 'https://http-pruthvi-adaptive-tutor-ai.hf.space'

print(f"Checking health of {BASE_URL}...")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    print(f"Status Code: {r.status_code}")
    print(f"Content-Type: {r.headers.get('Content-Type')}")
    print(f"Response Preview: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
