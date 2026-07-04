import os
import yaml
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Generator, Tuple

# In-memory lookup cache with sliding expiration
cache_store = {}
CACHE_TTL = 3600  # Cache duration: 1 hour
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import httpx
import uvicorn

# Configure logging to warning/error levels only
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ChineseRouter")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enable HTTP/2 and set 3-minute timeout limit for connection pool reuse
    app.state.client = httpx.AsyncClient(timeout=180.0, http2=True)
    yield
    await app.state.client.aclose()

app = FastAPI(title="Chinese LLM Router Proxy", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Gzip Middleware that automatically bypasses LLM API routes.
# Standard Gzip middleware buffers streaming responses (like SSE text/event-stream),
# which stops real-time token rendering. This subclass ensures static site pages
# still get compressed, but all API routes stream immediately.
class SafeGZipMiddleware(GZipMiddleware):
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            if "/v1/" in path or "/models" in path or "/api/" in path:
                # Bypass compression entirely
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)

app.add_middleware(SafeGZipMiddleware, minimum_size=1000)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        raise RuntimeError(f"Config load error: {e}")

config = load_config()

# Define dynamic PORT and LOCAL_URL based on configuration
CONFIG_PORT = config.get("routing", {}).get("port")
PORT = int(os.environ.get("PORT", CONFIG_PORT if CONFIG_PORT else 8000))
LOCAL_URL = f"http://localhost:{PORT}"

# Retrieve provider settings
PROVIDER = config.get("provider", "openrouter").lower()
logger.info(f"Loaded Router Provider: {PROVIDER}")

def map_model_for_provider(model_id: str) -> str:
    if PROVIDER == "opencode":
        if model_id.startswith("opencode-go/"):
            return model_id
        parts = model_id.split("/", 1)
        model_name = parts[-1]
        return f"opencode-go/{model_name}"
    return model_id


# Load default API keys from config or environment
def get_env_api_key() -> Optional[str]:
    provider_config = config.get(PROVIDER)
    if isinstance(provider_config, dict):
        key = provider_config.get("api_key")
        if key and isinstance(key, str):
            return key.strip()
            
    if PROVIDER == "opencode":
        key = os.environ.get("OPENCODE_API_KEY")
        if not key:
            key = os.environ.get("OPENROUTER_API_KEY")
        return key
    else:
        return os.environ.get("OPENROUTER_API_KEY")

ENV_API_KEY = get_env_api_key()
if not ENV_API_KEY:
    logger.warning(
        f"API Key is not set in config or environment. Provider is '{PROVIDER}'. "
        f"The proxy will run, but calls will fail unless configured in config.yaml, environment, or authorization header."
    )
else:
    masked_key = ENV_API_KEY[:10] + "..." + ENV_API_KEY[-4:] if len(ENV_API_KEY) > 10 else "invalid/too-short"
    logger.info(f"Loaded API Key: {masked_key} (length: {len(ENV_API_KEY)})")

def sanitize_json(obj: Any) -> Any:
    """Recursively converts any numeric 'id' field in a JSON object to a string."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, list):
        return [sanitize_json(item) for item in obj]
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k == 'id' and isinstance(v, (int, float)):
                new_dict[k] = str(v)
            else:
                new_dict[k] = sanitize_json(v)
        return new_dict
    return obj

class ToolCallIndexMapper:
    """Fixes the Quatarly streaming index mismatch bug by mapping incoming tool call indices to sequential ones."""
    def __init__(self):
        self.incoming_to_mapped = {}
        self.last_mapped = 0

    def map_index(self, incoming_index: int, has_id: bool) -> int:
        if incoming_index in self.incoming_to_mapped:
            return self.incoming_to_mapped[incoming_index]
        
        if has_id:
            mapped = len(self.incoming_to_mapped)
            self.incoming_to_mapped[incoming_index] = mapped
            self.last_mapped = mapped
            return mapped
        else:
            self.incoming_to_mapped[incoming_index] = self.last_mapped
            return self.last_mapped

def map_chunk_tool_calls(chunk: dict, mapper: ToolCallIndexMapper) -> dict:
    """Updates index on tool calls inside choice deltas using the mapper."""
    if not isinstance(chunk, dict):
        return chunk
    
    if "choices" in chunk and isinstance(chunk["choices"], list):
        for choice in chunk["choices"]:
            if "delta" in choice and isinstance(choice["delta"], dict):
                delta = choice["delta"]
                if "tool_calls" in delta and isinstance(delta["tool_calls"], list):
                    for tool_call in delta["tool_calls"]:
                        if isinstance(tool_call, dict) and "index" in tool_call:
                            incoming_idx = tool_call["index"]
                            has_id = "id" in tool_call and tool_call["id"] is not None
                            mapped_idx = mapper.map_index(incoming_idx, has_id)
                            tool_call["index"] = mapped_idx
    return chunk

def classify_by_keywords(text: str) -> Optional[str]:
    """Classifies a query based on keyword counts defined in config.yaml."""
    text_lower = text.lower()
    scores = {}
    
    for category, cat_data in config.get("categories", {}).items():
        if category == "general":
            continue
        score = 0
        for keyword in cat_data.get("keywords", []):
            pattern = r'\b' + re.escape(keyword) + r'\b'
            score += len(re.findall(pattern, text_lower))
        if score > 0:
            scores[category] = score
            
    if scores:
        best_cat = max(scores, key=scores.get)
        logger.info(f"Regex Keyword matches: {scores} -> Selected: {best_cat}")
        return best_cat
    return None

async def classify_semantically(query: str, api_key: str, client: httpx.AsyncClient) -> str:
    """Classifies a query using a cheap LLM via the active provider."""
    classifier_model = config.get("routing", {}).get("classifier_model", "qwen/qwen-2.5-7b-instruct")
    api_base = config.get(PROVIDER, {}).get("api_base")
    if not api_base:
        api_base = "https://openrouter.ai/api/v1" if PROVIDER == "openrouter" else "https://opencode.ai/zen/go/v1"
        
    prompt = (
        "You are a task classifier. Classify the user query into exactly one of these categories: "
        "'coding', 'design', 'agents', 'reasoning', or 'general'.\n"
        f"Query: \"{query}\"\n"
        "Category (respond with ONLY the single lowercase category word, no explanation, no markdown):"
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": LOCAL_URL,
        "X-Title": "Chinese LLM Router Classifier"
    }
    
    payload = {
        "model": map_model_for_provider(classifier_model),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    try:
        response = await client.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=5.0)
        if response.status_code == 200:
            result = response.json()
            cat = result["choices"][0]["message"]["content"].strip().lower()
            cat = re.sub(r'[^\w]', '', cat)
            if cat in config.get("categories", {}):
                logger.info(f"LLM Classifier selected: '{cat}'")
                return cat
            else:
                logger.warning(f"LLM Classifier returned invalid category: '{cat}'")
        else:
            logger.error(f"LLM Classifier request failed with code {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error calling semantic classifier: {e}")
        
    return "general"

async def determine_category(messages: List[Dict[str, Any]], api_key: Optional[str], client: httpx.AsyncClient) -> str:
    text_to_classify = ""
    for msg in messages[-3:]:
        if isinstance(msg.get("content"), str):
            text_to_classify += " " + msg["content"]
            
    # Regex check
    cat = classify_by_keywords(text_to_classify)
    if cat:
        return cat
        
    # LLM semantic check
    enable_semantic = config.get("routing", {}).get("enable_semantic_classification", True)
    if enable_semantic and api_key:
        return await classify_semantically(text_to_classify, api_key, client)
        
    return "general"

def calculate_estimated_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = config.get("pricing", {}).get(model)
    if not pricing:
        if model.startswith("opencode-go/"):
            model_name = model.split("/", 1)[-1]
            for key, val in config.get("pricing", {}).items():
                if key.endswith(f"/{model_name}"):
                    pricing = val
                    break
    if not pricing:
        pricing = {"prompt": 0.5, "completion": 0.5}
    
    p_cost = (prompt_tokens / 1_000_000) * pricing.get("prompt", 0.5)
    c_cost = (completion_tokens / 1_000_000) * pricing.get("completion", 0.5)
    return p_cost + c_cost

def print_transaction_summary(category: str, selected_model: str, actual_model: str, prompt_tokens: int, completion_tokens: int):
    cost = calculate_estimated_cost(actual_model, prompt_tokens, completion_tokens)
    total_tokens = prompt_tokens + completion_tokens
    cost_str = f"${cost:.6f}"
    
    summary = (
        f"\n"
        f"+-------------------------------------------------------------------------------------------------+\n"
        f"|                                     TRANSACTION SUMMARY                                         |\n"
        f"+----------------------+--------------------------------------------------------------------------+\n"
        f"| Provider             | {PROVIDER.upper():<72} |\n"
        f"| Category             | {category:<72} |\n"
        f"| Selected Model       | {selected_model:<72} |\n"
        f"| Actual Model         | {actual_model:<72} |\n"
        f"| Prompt Tokens        | {prompt_tokens:<72} |\n"
        f"| Completion Tokens    | {completion_tokens:<72} |\n"
        f"| Total Tokens         | {total_tokens:<72} |\n"
        f"| Estimated Cost       | {cost_str:<72} |\n"
        f"+----------------------+--------------------------------------------------------------------------+\n"
    )
    print(summary)

@app.get("/v1/models")
async def list_models(request: Request, authorization: Optional[str] = Header(None)):
    cache_key = ("GET", "v1/models", "")
    cached = cache_store.get(cache_key)
    if cached:
        cached_data, timestamp = cached
        if time.time() - timestamp < CACHE_TTL:
            # Update timestamp on hit (Sliding Expiration: 1 hour from last request)
            cache_store[cache_key] = (cached_data, time.time())
            return JSONResponse(content=cached_data)

    virtual_models = [
        {"id": "custom-router", "object": "model", "owned_by": "custom"},
        {"id": "qwen/qwen3.7-max", "object": "model", "owned_by": "alibaba"},
        {"id": "deepseek/deepseek-v4-pro", "object": "model", "owned_by": "deepseek"},
        {"id": "z-ai/glm-5.2", "object": "model", "owned_by": "z-ai"},
        {"id": "moonshotai/kimi-k2.7-code", "object": "model", "owned_by": "moonshotai"},
        {"id": "minimax/minimax-m3", "object": "model", "owned_by": "minimax"}
    ]

    api_base = config.get(PROVIDER, {}).get("api_base")
    if not api_base:
        api_base = "https://openrouter.ai/api/v1" if PROVIDER == "openrouter" else "https://opencode.ai/zen/go/v1"

    client = request.app.state.client
    headers = {}
    
    # Use global key or forwarded auth header if present
    api_key = ENV_API_KEY
    if not api_key and authorization:
        api_key = authorization.replace("Bearer ", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = await client.get(f"{api_base}/models", headers=headers, timeout=5.0)
        if response.status_code == 200:
            res_json = response.json()
            remote_models = res_json.get("data", [])
            
            # Merge remote models, filtering out duplicates of virtual ones
            virtual_ids = {m["id"] for m in virtual_models}
            merged = list(virtual_models)
            for m in remote_models:
                if m.get("id") not in virtual_ids:
                    merged.append(m)
            
            result = {"object": "list", "data": merged}
            cache_store[cache_key] = (result, time.time())
            return JSONResponse(content=result)
    except Exception as e:
        logger.warning(f"Failed to fetch dynamic models from provider: {e}. Falling back to virtual models.")
    
    # Fallback to local virtual models list
    result = {"object": "list", "data": virtual_models}
    return JSONResponse(content=result)

async def forward_request_with_failover(
    client: httpx.AsyncClient,
    api_base: str,
    headers: dict,
    body: dict,
    fallback_models: List[str],
    stream: bool
) -> Tuple[httpx.Response, str]:
    """Forwards the request using native server fallbacks (OpenRouter) or client-side retry loop (OpenCode)."""
    if PROVIDER == "openrouter":
        # OpenRouter supports native model list parameter
        body["model"] = fallback_models[0]
        body["models"] = fallback_models
        
        req = client.build_request("POST", f"{api_base}/chat/completions", headers=headers, json=body)
        r = await client.send(req, stream=stream)
        if r.status_code != 200:
            if not stream:
                # Read response text for detailed error output
                raise HTTPException(status_code=r.status_code, detail=f"OpenRouter API Error: {r.text}")
            else:
                await r.aread()
                raise HTTPException(status_code=r.status_code, detail=f"OpenRouter Streaming Error: {r.text}")
        return r, fallback_models[0]
        
    # OpenCode Go fallback loop (client-side failover)
    last_error = ""
    for model in fallback_models:
        payload = body.copy()
        payload["model"] = map_model_for_provider(model)
        # Strip OpenRouter custom parameters
        if "models" in payload:
            del payload["models"]
            
        logger.info(f"Forwarding call to {PROVIDER} for model: {model}")
        try:
            req = client.build_request("POST", f"{api_base}/chat/completions", headers=headers, json=payload)
            r = await client.send(req, stream=stream)
            if r.status_code == 200:
                return r, model
            else:
                if stream:
                    await r.aread()
                err_msg = r.text
                logger.warning(f"Model {model} failed on {PROVIDER} with code {r.status_code}: {err_msg}")
                last_error = f"Status {r.status_code}: {err_msg}"
        except Exception as e:
            logger.warning(f"Model {model} request failed with exception: {e}")
            last_error = str(e)
            
    raise HTTPException(
        status_code=502,
        detail=f"All models in the fallback chain failed on provider {PROVIDER}. Last error: {last_error}"
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    # 1. Use the server-configured key (from config.yaml or env) first
    api_key = ENV_API_KEY
    
    # 2. Fallback to client-provided key only if the server has no key configured
    if not api_key and authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        if token and token not in ["dummy-key-handled-by-proxy", "placeholder", "sk-or-"]:
            api_key = token
        
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=f"API key not found. Please configure 'api_key' in config.yaml, set environment variables, or provide a valid key in the Authorization header."
        )
        
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
        
    messages = body.get("messages", [])
    requested_model = body.get("model", "")
    stream = body.get("stream", False)
    
    # Check if request targets auto-routing
    should_route = requested_model in ["custom-router", "default", "qwen/qwen3.7-max", ""]
    
    # Retrieve the pre-initialized global connection client (reuses TCP/SSL pool with HTTP/2 support)
    client = request.app.state.client
    
    if should_route:
        category = await determine_category(messages, api_key, client)
        cat_config = config.get("categories", {}).get(category, {})
        primary_model = cat_config.get("primary", config["routing"]["default_model"])
        fallback_models = cat_config.get("fallbacks", [primary_model])
    else:
        category = "explicit"
        primary_model = requested_model
        matched_cat = next((cat for cat, data in config.get("categories", {}).items() if data.get("primary") == requested_model), None)
        if matched_cat:
            fallback_models = config["categories"][matched_cat].get("fallbacks", [primary_model])
        else:
            fallback_models = [primary_model, config["routing"]["default_model"]]
            
    api_base = config.get(PROVIDER, {}).get("api_base")
    if not api_base:
        api_base = "https://openrouter.ai/api/v1" if PROVIDER == "openrouter" else "https://opencode.ai/zen/go/v1"
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": LOCAL_URL,
        "X-Title": "Chinese LLM Router Gateway"
    }
    
    async def stream_generator(response: httpx.Response, selected_model: str) -> Generator[bytes, None, None]:
        actual_model = selected_model
        prompt_tokens = 0
        completion_tokens = 0
        mapper = ToolCallIndexMapper()
        has_thinking = False
        
        async for line in response.aiter_lines():
            if not line:
                continue
            
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    if has_thinking:
                        # Close the thinking tag before sending [DONE]
                        close_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": "\n</think>\n\n"
                                }
                            }]
                        }
                        yield f"data: {json.dumps(close_chunk)}\n\n".encode("utf-8")
                        has_thinking = False
                    yield f"{line}\n".encode("utf-8")
                    continue
                try:
                    chunk = json.loads(data_str)
                    chunk = sanitize_json(chunk)
                    chunk = map_chunk_tool_calls(chunk, mapper)
                    
                    # Convert reasoning/thinking tokens to standard content for clients to display
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        content = delta.get("content", "")
                        
                        if reasoning:
                            if not has_thinking:
                                has_thinking = True
                                delta["content"] = f"<think>\n{reasoning}"
                            else:
                                delta["content"] = reasoning
                        elif has_thinking and content:
                            has_thinking = False
                            delta["content"] = f"\n</think>\n\n{content}"
                            
                    if "model" in chunk:
                        actual_model = chunk["model"]
                    if "usage" in chunk and chunk["usage"]:
                        prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                        completion_tokens = chunk["usage"].get("completion_tokens", 0)
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                except Exception:
                    yield f"{line}\n".encode("utf-8")
            else:
                yield f"{line}\n".encode("utf-8")
                    
        if prompt_tokens > 0 or completion_tokens > 0:
            print_transaction_summary(category, selected_model, actual_model, prompt_tokens, completion_tokens)
        else:
            logger.info(f"Stream completed. Routed Category: {category} | Selected Model: {selected_model} -> Responded: {actual_model}")

    if stream:
        try:
            r, actual_primary = await forward_request_with_failover(
                client, api_base, headers, body, fallback_models, stream=True
            )
            return StreamingResponse(
                stream_generator(r, actual_primary),
                status_code=r.status_code,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        try:
            r, actual_primary = await forward_request_with_failover(
                client, api_base, headers, body, fallback_models, stream=False
            )
            response_json = r.json()
            response_json = sanitize_json(response_json)
            actual_model = response_json.get("model", actual_primary)
            
            # Map reasoning_content in non-streaming responses
            if "choices" in response_json and response_json["choices"]:
                message = response_json["choices"][0].get("message", {})
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                content = message.get("content", "")
                if reasoning and content:
                    message["content"] = f"<think>\n{reasoning}\n</think>\n\n{content}"
                elif reasoning:
                    message["content"] = f"<think>\n{reasoning}\n</think>\n\n"
                    
            usage = response_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            print_transaction_summary(category, actual_primary, actual_model, prompt_tokens, completion_tokens)
            
            return JSONResponse(content=response_json)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Non-streaming request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Auto-run Hermes Agent configuration on startup to ensure it uses the correct dynamic port/URL
    try:
        from configure_hermes import main as configure_hermes_main
        configure_hermes_main()
    except Exception as e:
        logger.warning(f"Could not automatically configure Hermes config: {e}")
        
    logger.info(f"Starting Chinese LLM Router Proxy on port {PORT}...")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
