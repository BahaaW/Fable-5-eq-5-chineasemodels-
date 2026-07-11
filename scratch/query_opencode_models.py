import httpx
import json
import os

api_key = os.environ.get("OPENCODE_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
api_base = "https://opencode.ai/zen/go/v1"

headers = {
    "Authorization": f"Bearer {api_key}"
}

try:
    r = httpx.get(f"{api_base}/models", headers=headers, timeout=10.0)
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Error:", e)
