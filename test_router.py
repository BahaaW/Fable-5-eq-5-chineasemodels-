import httpx
import sys
import json
import os
import yaml

def load_port():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("routing", {}).get("port", 8000)
    except Exception:
        return 8000

PORT = load_port()
PROXY_URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# Define test cases for each routed category
TEST_CASES = [
    {
        "name": "Coding Task (Expected: deepseek/deepseek-v4-pro)",
        "prompt": "Write a high-performance Python function to compute the Fibonacci sequence using memoization and add type hints."
    },
    {
        "name": "Web Design Task (Expected: z-ai/glm-5.2)",
        "prompt": "Create a beautiful glassmorphic contact form in HTML and CSS with premium gradients and hover animations."
    },
    {
        "name": "Reasoning & Math Task (Expected: minimax/minimax-m3)",
        "prompt": "A box contains 3 red balls and 7 blue balls. If we draw two balls without replacement, what is the probability that both are red? Show step-by-step mathematical reasoning."
    }
]

def load_provider():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("provider", "openrouter").lower()
    except Exception:
        return "openrouter"

def check_proxy_running():
    try:
        httpx.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=2.0)
        return True
    except Exception:
        return False

def run_test_case(name: str, prompt: str, api_key: str):
    print(f"\n======================================================================")
    print(f" RUNNING TEST: {name}")
    print(f" Prompt: \"{prompt}\"")
    print(f"======================================================================")
    
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    payload = {
        "model": "custom-router",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 150 # Cap tokens to keep tests cheap and fast
    }
    
    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(PROXY_URL, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                actual_model = result.get("model", "unknown")
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                print(f"Actual Responding Model: {actual_model}")
                print(f"Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens}")
                
                if reasoning:
                    print("\n[Reasoning Thoughts Preview]:")
                    print("----------------------------------------------------------------------")
                    print(reasoning.strip()[:400] + ("..." if len(reasoning) > 400 else ""))
                    print("----------------------------------------------------------------------")
                    
                print("\nResponse Content Preview:")
                print("----------------------------------------------------------------------")
                if content:
                    print(content.strip()[:400] + ("..." if len(content) > 400 else ""))
                else:
                    print("(No final output content generated yet - reasoning model still thinking within max_tokens)")
                print("----------------------------------------------------------------------")
            else:
                print(f"Error {response.status_code}: {response.text}")
    except httpx.ConnectError:
        print(f"Connection Error: Could not connect to the proxy. Is it running on http://127.0.0.1:{PORT}?")
    except Exception as e:
        print(f"Unexpected error running test: {e}")

def main():
    if not check_proxy_running():
        print("Error: The local proxy server is not running.")
        print("Please start it first in another terminal: python proxy.py")
        sys.exit(1)
        
    provider = load_provider()
    
    api_key = ""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            api_key = cfg.get(provider, {}).get("api_key", "")
    except Exception:
        pass
        
    if not api_key:
        key_var = "OPENCODE_API_KEY" if provider == "opencode" else "OPENROUTER_API_KEY"
        api_key = os.environ.get(key_var, "")
        if provider == "opencode" and not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            
    if not api_key:
        print(f"Warning: API Key not found in config.yaml or environment variables.")
        print(f"You can run this test script, but calls to {provider.upper()} will fail unless key is configured.")
        user_key = input(f"Enter {provider.upper()} API Key (optional): ").strip()
        if user_key:
            api_key = user_key

    print(f"Starting router benchmark verification using provider: {provider.upper()}...")
    for tc in TEST_CASES:
        run_test_case(tc["name"], tc["prompt"], api_key)

if __name__ == "__main__":
    main()
