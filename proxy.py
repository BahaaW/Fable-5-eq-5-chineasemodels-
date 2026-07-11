import os
import yaml
import json
import logging
import re
import time
import copy
import asyncio
import uuid
import html
from typing import Dict, List, Any, Optional, Generator, Tuple

# In-memory lookup cache with sliding expiration
cache_store = {}
CACHE_TTL = 3600  # Cache duration: 1 hour


class RateLimiter:
    """Simple sliding-window rate limiter for proxy endpoints."""
    def __init__(self, max_requests: int = 120, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: Dict[str, List[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        if client_ip not in self._clients:
            self._clients[client_ip] = []
        self._clients[client_ip] = [t for t in self._clients[client_ip] if t > window_start]
        if len(self._clients[client_ip]) >= self.max_requests:
            return False
        self._clients[client_ip].append(now)
        return True

    async def cleanup(self):
        """Periodically prune stale client entries."""
        while True:
            await asyncio.sleep(600)
            now = time.time()
            stale = [ip for ip, times in self._clients.items() if not times or times[-1] < now - self.window_seconds]
            for ip in stale:
                del self._clients[ip]


_rate_limiter = RateLimiter()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import httpx
import uvicorn

# Configure structured JSON logging for SIEM/aggregation compatibility
class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            entry["rid"] = record.request_id
        if record.exc_info and record.exc_info[1]:
            entry["exc"] = str(record.exc_info[1])
        return json.dumps(entry, default=str)


_handler = logging.StreamHandler()
_handler.setFormatter(_JSONFormatter())
logger = logging.getLogger("ChineseRouter")
logger.handlers.clear()
logger.addHandler(_handler)
logger.setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enable HTTP/2 and set 5-minute timeout limit for connection pool reuse
    app.state.client = httpx.AsyncClient(timeout=300.0, http2=True)
    # Background tasks: cache eviction and rate-limiter cleanup
    cleanup_tasks = [
        asyncio.create_task(_cache_eviction_loop()),
        asyncio.create_task(_rate_limiter.cleanup()),
    ]
    yield
    for task in cleanup_tasks:
        task.cancel()
    await app.state.client.aclose()


async def _cache_eviction_loop():
    """Evict expired cache entries every 5 minutes to prevent memory leaks."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        expired = [k for k, (_, ts) in list(cache_store.items()) if now - ts > CACHE_TTL]
        for k in expired:
            cache_store.pop(k, None)

app = FastAPI(title="Chinese LLM Router Proxy", lifespan=lifespan)

# Standard OpenAI-compliant error handlers so that client extensions decode error responses gracefully.
@app.exception_handler(HTTPException)
async def openai_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "param": None,
                "code": str(exc.status_code)
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def openai_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Validation Error: {exc.errors()}",
                "type": "invalid_request_error",
                "param": None,
                "code": "validation_error"
            }
        }
    )

@app.exception_handler(Exception)
async def openai_general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Internal Server Error: {str(exc)}",
                "type": "api_error",
                "param": None,
                "code": "internal_server_error"
            }
        }
    )

# CORS Middleware — restricted to localhost/127.0.0.1 origins only.
# allow_credentials=True is safe because origins are never wildcarded.
@app.middleware("http")
async def localhost_cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin", "")
    is_localhost = bool(re.match(r"https?://(localhost|127\.0\.0\.1)(:\d+)?$", origin))

    if request.method == "OPTIONS":
        # Handle CORS preflight directly
        resp = Response(status_code=200)
    else:
        resp = await call_next(request)

    if origin and is_localhost:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Max-Age"] = "86400"

    return resp


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Inject a short request ID into every response for audit trail."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

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
        # Strip any provider prefix (like moonshotai/, qwen/, or opencode-go/)
        model_name = model_id.split("/")[-1]
        
        # Specific model name translations to map OpenRouter names to OpenCode Go equivalents
        translation_map = {
            "qwen-2.5-72b-instruct": "qwen3.7-max",
            "qwen-2.5-7b-instruct": "qwen3.5-plus",
            "qwen-2.5-coder-32b-instruct": "qwen3.7-plus",
        }
        return translation_map.get(model_name, model_name)
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

class XMLToJSONStreamParser:
    """Parses streaming raw XML tool calls and repackages them as standard JSON tool_calls on the fly."""
    MAX_BUFFER = 100_000  # Safety cap to prevent memory exhaustion from unbounded XML

    def __init__(self):
        self.buffer = ""
        self.in_xml_mode = False
        self.tool_call_index = 0

    def feed(self, text: str) -> List[dict]:
        """Feeds a text chunk and returns a list of OpenAI delta chunks (either content or tool_calls)."""
        self.buffer += text
        if len(self.buffer) > self.MAX_BUFFER:
            raise ValueError(
                f"XMLToJSONStreamParser buffer overflow ({len(self.buffer)} > {self.MAX_BUFFER}). "
                "The model is streaming malformed or unterminated XML tool calls."
            )
        chunks = []

        while True:
            if not self.in_xml_mode:
                # Find if any XML tool call tags start
                idx_func = self.buffer.find("<function_calls>")
                idx_invoke = self.buffer.find("<invoke")
                
                # Determine which tag starts first
                indices = [i for i in [idx_func, idx_invoke] if i != -1]
                if not indices:
                    # No XML tag started, yield all current buffer
                    if self.buffer:
                        chunks.append({"content": self.buffer})
                        self.buffer = ""
                    break
                else:
                    start_idx = min(indices)
                    # Yield everything before the start of the XML tag as normal content
                    if start_idx > 0:
                        chunks.append({"content": self.buffer[:start_idx]})
                        self.buffer = self.buffer[start_idx:]
                    self.in_xml_mode = True
            else:
                # We are in XML mode. Look for closed invoke tags.
                # Find all complete invoke blocks
                match = re.search(r'<invoke\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</invoke>', self.buffer, re.DOTALL)
                if match:
                    tool_name = match.group(1)
                    inner_xml = match.group(2)
                    
                    # Extract parameters
                    params = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', inner_xml, re.DOTALL)
                    params_dict = {}
                    for param_name, param_val in params:
                        # Clean up value and unescape XML entities
                        clean_val = html.unescape(param_val.strip())
                        # Attempt to parse numbers/booleans/JSON arrays/dicts if applicable, otherwise keep as string
                        if clean_val.lower() == "true":
                            params_dict[param_name] = True
                        elif clean_val.lower() == "false":
                            params_dict[param_name] = False
                        else:
                            try:
                                # Try parsing as number
                                if "." in clean_val:
                                    params_dict[param_name] = float(clean_val)
                                else:
                                    params_dict[param_name] = int(clean_val)
                            except ValueError:
                                try:
                                    # Try parsing as JSON array/object
                                    params_dict[param_name] = json.loads(clean_val)
                                except Exception:
                                    params_dict[param_name] = clean_val
                    
                    # Create tool call dict
                    tool_call = {
                        "index": self.tool_call_index,
                        "id": f"call_xml_{self.tool_call_index}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(params_dict)
                        }
                    }
                    self.tool_call_index += 1
                    
                    chunks.append({"tool_calls": [tool_call]})
                    
                    # Remove the parsed invoke tag from the buffer
                    end_pos = match.end()
                    self.buffer = self.buffer[end_pos:]
                    continue
                
                # Check if we see the end of function_calls block
                idx_end_func = self.buffer.find("</function_calls>")
                if idx_end_func != -1:
                    # Remove the closing tag and exit XML mode
                    self.buffer = self.buffer[idx_end_func + len("</function_calls>"):]
                    self.in_xml_mode = False
                    continue
                
                # If we are in XML mode but have no complete invoke tag yet, do not yield anything (buffering)
                break

        return chunks

# Pre-compiled keyword patterns for fast regex classification (built once at startup)
_KEYWORD_PATTERNS: Dict[str, List[Tuple[str, re.Pattern]]] = {}

def _build_keyword_patterns():
    for category, cat_data in config.get("categories", {}).items():
        if category == "general":
            continue
        patterns = []
        for keyword in cat_data.get("keywords", []):
            patterns.append((keyword, re.compile(r"\b" + re.escape(keyword) + r"\b")))
        _KEYWORD_PATTERNS[category] = patterns

_build_keyword_patterns()


def classify_by_keywords(text: str) -> Optional[str]:
    """Classifies a query based on keyword counts defined in config.yaml."""
    text_lower = text.lower()
    scores: dict = {}
    for category, patterns in _KEYWORD_PATTERNS.items():
        score = sum(len(pattern.findall(text_lower)) for _, pattern in patterns)
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
    # Apply per-model parameter fixes (e.g. kimi temp/top_p restrictions)
    payload = sanitize_payload_for_model(payload, classifier_model)

    classifier_timeout = config.get("routing", {}).get("classifier_timeout", 120)
    try:
        response = await client.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=classifier_timeout)
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices") or []
            if not choices:
                logger.warning(f"LLM Classifier returned no choices: {json.dumps(result)[:300]}")
                return "general"
            message = choices[0].get("message", {}) or {}
            raw_content = message.get("content") or ""
            if not raw_content:
                logger.warning(f"LLM Classifier returned empty content: {json.dumps(choices[0])[:300]}")
                return "general"
            cat = raw_content.strip().lower()
            cat = re.sub(r'[^\w]', '', cat)
            if cat in config.get("categories", {}):
                logger.info(f"LLM Classifier selected: '{cat}'")
                return cat
            else:
                logger.warning(f"LLM Classifier returned invalid category: '{cat}'")
        else:
            logger.error(f"LLM Classifier request failed with code {response.status_code}: {response.text[:300]}")
    except Exception as e:
        logger.error(f"Error calling semantic classifier: {type(e).__name__}: {e}")
        
    return "general"

def messages_contain_images(messages: List[Dict[str, Any]]) -> bool:
    """Detect whether any message in the conversation contains an image content block."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype in ("image_url", "image"):
                        return True
    return False

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

def get_http_client(request: Request) -> httpx.AsyncClient:
    """Safely retrieves the connection pool client, falling back to a fresh one if lifespan was bypassed (e.g. in tests)."""
    try:
        return request.app.state.client
    except AttributeError:
        # For testing environments where lifespan events are not executed
        if not hasattr(request.app.state, "_fallback_client"):
            request.app.state._fallback_client = httpx.AsyncClient(timeout=300.0, http2=True)
        return request.app.state._fallback_client

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
        {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
        {"id": "gpt-4", "object": "model", "owned_by": "openai"},
        {"id": "claude-3-5-sonnet", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-3-5-haiku", "object": "model", "owned_by": "anthropic"},
        {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
        {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"},
        {"id": "qwen/qwen3.7-max", "object": "model", "owned_by": "alibaba"},
        {"id": "deepseek/deepseek-v4-pro", "object": "model", "owned_by": "deepseek"},
        {"id": "z-ai/glm-5.2", "object": "model", "owned_by": "z-ai"},
        {"id": "moonshotai/kimi-k2.7-code", "object": "model", "owned_by": "moonshotai"},
        {"id": "minimax/minimax-m3", "object": "model", "owned_by": "minimax"}
    ]

    api_base = config.get(PROVIDER, {}).get("api_base")
    if not api_base:
        api_base = "https://openrouter.ai/api/v1" if PROVIDER == "openrouter" else "https://opencode.ai/zen/go/v1"

    client = get_http_client(request)
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

def strip_unsupported_content_blocks(payload: dict, model_lower: str, allowed_types: set) -> dict:
    """Remove content blocks whose type is not in allowed_types for a given model.
    
    Some providers (e.g. DeepSeek) only accept `text` content blocks and reject
    `image_url` / `image` blocks with a 400. We strip unsupported blocks and
    insert a text placeholder so the model knows an image was present.
    """
    messages = payload.get("messages", [])
    if not messages:
        return payload

    changed = False
    new_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_blocks = []
        msg_changed = False
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            btype = block.get("type")
            if btype in allowed_types:
                new_blocks.append(block)
            else:
                msg_changed = True
                # Replace unsupported block with a text placeholder
                if btype == "image_url":
                    placeholder = "[image attached - not supported by this model]"
                elif btype == "image":
                    placeholder = "[image attached - not supported by this model]"
                elif btype == "input_audio":
                    placeholder = "[audio attached - not supported by this model]"
                else:
                    placeholder = f"[{btype} content - not supported by this model]"
                new_blocks.append({"type": "text", "text": placeholder})

        if msg_changed:
            changed = True
            new_msg = {k: v for k, v in msg.items()}
            new_msg["content"] = new_blocks
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)

    if not changed:
        # Always return a safe shallow copy — never leak the caller's original dict.
        return payload.copy()
    payload = payload.copy()
    payload["messages"] = new_messages
    return payload


def sanitize_payload_for_model(payload: dict, model: str) -> dict:
    """Apply per-model parameter fixes required by upstream providers.
    
    Some Chinese providers reject requests that look fine to OpenAI/OpenRouter clients:
    - Moonshot K2.7 code only accepts temperature=1 (any other value -> 400).
    - DeepSeek thinking models require reasoning_content to be echoed back in prior
      assistant turns; if the client stripped it, the API rejects the request with
      "The reasoning_content in the thinking mode must be passed back to the API."
      We work around this by disabling the thinking mode for DeepSeek-v4-pro when the
      conversation history lacks reasoning_content, so the request can still succeed.
    """
    model_lower = (model or "").lower()

    # Moonshot K2.7 code: only temperature=1 and top_p=0.95 are allowed
    if "kimi-k2.7" in model_lower or "kimi-k2-7" in model_lower:
        payload = payload.copy()
        payload["temperature"] = 1
        payload["top_p"] = 0.95

    # DeepSeek thinking models: if reasoning_content is missing from history, disable thinking
    if "deepseek-v4-pro" in model_lower or "deepseek-v4" in model_lower:
        # DeepSeek does not support image_url content blocks (only text).
        # Strip image blocks and replace with a text placeholder so the request is accepted.
        payload = strip_unsupported_content_blocks(payload, model_lower, allowed_types={"text"})

        # Deep-copy AFTER strip_unsupported_content_blocks (which mutates in-place)
        # so we never mutate the caller's original body dict.
        payload = copy.deepcopy(payload)

        messages = payload.get("messages", [])
        has_reasoning = any(
            isinstance(m, dict) and m.get("role") == "assistant" and m.get("reasoning_content")
            for m in messages
        )
        if not has_reasoning:
            # Disable thinking mode so the API does not require reasoning_content echo-back.
            # DeepSeek expects `thinking` to be a ThinkingOptions struct, not a boolean,
            # so we use the object form {"type": "disabled"} and also set enable_thinking=false.
            payload["enable_thinking"] = False
            payload["thinking"] = {"type": "disabled"}

    # MiniMax M3 is served via the Anthropic SDK, which enforces strict parameter
    # constraints that OpenAI/OpenRouter clients routinely violate:
    #   - temperature must be in [0, 1] (clients often send up to 2.0)
    #   - top_p must be in [0, 1]
    #   - max_tokens is required (OpenAI clients sometimes omit it)
    # We clamp/patch these so the request is accepted.
    if "minimax-m3" in model_lower or "minimax/m3" in model_lower:
        temp = payload.get("temperature")
        top_p = payload.get("top_p")
        needs_clamp = (
            (isinstance(temp, (int, float)) and (temp < 0 or temp > 1))
            or (isinstance(top_p, (int, float)) and (top_p < 0 or top_p > 1))
            or not payload.get("max_tokens")
        )
        if needs_clamp:
            payload = payload.copy()
            if isinstance(temp, (int, float)):
                payload["temperature"] = max(0.0, min(1.0, float(temp)))
            else:
                payload["temperature"] = 1.0
            if isinstance(top_p, (int, float)):
                payload["top_p"] = max(0.0, min(1.0, float(top_p)))
            if not payload.get("max_tokens"):
                payload["max_tokens"] = 4096

    return payload


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
        # OpenRouter supports native model list parameter and server-side failover,
        # but we add client-side retries for transient errors (5xx, rate limits, timeouts).
        active_model = fallback_models[0]
        if not ("claude" in active_model.lower() or "anthropic" in active_model.lower()):
            body = body.copy()
            body["messages"] = strip_prompt_caching(body.get("messages", []))

        body["model"] = active_model
        body["models"] = fallback_models
        body = sanitize_payload_for_model(body, active_model)

        max_retries = config.get("routing", {}).get("max_retries", 2)
        last_error = ""
        for attempt in range(max_retries):
            try:
                req = client.build_request("POST", f"{api_base}/chat/completions", headers=headers, json=body)
                r = await client.send(req, stream=stream)
                if r.status_code == 200:
                    return r, fallback_models[0]
                if stream:
                    await r.aread()
                err_msg = r.text
                is_transient = (
                    r.status_code >= 500
                    or r.status_code == 429
                    or "rate limit" in err_msg.lower()
                    or "overloaded" in err_msg.lower()
                )
                if is_transient and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "OpenRouter transient error %d (attempt %d/%d), retrying in %ds: %s",
                        r.status_code, attempt + 1, max_retries, wait, err_msg[:200],
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    "OpenRouter API Error %d: %s",
                    r.status_code, err_msg[:300],
                )
                raise HTTPException(status_code=r.status_code, detail=f"OpenRouter API Error: {err_msg[:200]}")
            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("OpenRouter timeout (attempt %d/%d), retrying in %ds", attempt + 1, max_retries, wait)
                    await asyncio.sleep(wait)
                    continue
                logger.error("OpenRouter timeout after %d attempts: %s", max_retries, e)
                raise HTTPException(status_code=504, detail=f"OpenRouter timeout: {e}")
        raise HTTPException(status_code=502, detail=f"OpenRouter all attempts failed: {last_error}")
        
    # OpenCode Go fallback loop (client-side failover)
    last_error = ""
    for model in fallback_models:
        payload = body.copy()
        # Always strip cache_control when calling OpenCode Go
        payload["messages"] = strip_prompt_caching(payload.get("messages", []))
        payload["model"] = map_model_for_provider(model)
        # Strip OpenRouter custom parameters
        if "models" in payload:
            del payload["models"]

        payload = sanitize_payload_for_model(payload, model)
            
        logger.info(f"Forwarding call to {PROVIDER} for model: {model}")
        max_retries = config.get("routing", {}).get("max_retries", 2)
        for attempt in range(max_retries):
            try:
                req = client.build_request("POST", f"{api_base}/chat/completions", headers=headers, json=payload)
                r = await client.send(req, stream=stream)
                if r.status_code == 200:
                    return r, model
                else:
                    if stream:
                        await r.aread()
                    err_msg = r.text
                    # Retry on transient upstream errors (5xx, rate limits, generic upstream failures)
                    is_transient = (
                        r.status_code >= 500 or
                        r.status_code == 429 or
                        "Upstream request failed" in err_msg or
                        "rate limit" in err_msg.lower() or
                        "overloaded" in err_msg.lower()
                    )
                    if is_transient and attempt < max_retries - 1:
                        wait = 2 ** attempt  # 1s, 2s backoff
                        logger.warning(
                            "Model %s transient error on %s (attempt %d/%d), retrying in %ds: %s",
                            model, PROVIDER, attempt + 1, max_retries, wait, err_msg[:200],
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error(
                        "Model %s failed on %s with code %d: %s",
                        model, PROVIDER, r.status_code, err_msg[:300],
                    )
                    last_error = f"Status {r.status_code}: {err_msg[:200]}"
                    break  # Non-transient or final attempt, move to next model
            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Model %s timed out on %s (attempt %d/%d), retrying in %ds",
                        model, PROVIDER, attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("Model %s timed out on %s: %s", model, PROVIDER, e)
                last_error = f"Timeout: {type(e).__name__}"
                break
            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Model %s connection failed on %s (attempt %d/%d), retrying in %ds",
                        model, PROVIDER, attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("Model %s connection failed on %s: %s", model, PROVIDER, e)
                last_error = f"Connection error: {type(e).__name__}"
                break
            except httpx.HTTPError as e:
                logger.error("Model %s HTTP error on %s: %s", model, PROVIDER, e)
                last_error = f"HTTP error: {type(e).__name__}"
                break
            except Exception as e:
                logger.error("Model %s request failed with exception: %s", model, e)
                last_error = f"{type(e).__name__}: {e}"
                break
            
    raise HTTPException(
        status_code=502,
        detail=f"All models in the fallback chain failed on provider {PROVIDER}. Last error: {last_error}"
    )

def inject_reminder(messages: List[dict]) -> List[dict]:
    """Appends a strict system constraint reminding the model to use tool calls instead of raw markdown text blocks for coding tasks."""
    if not messages:
        return messages

    reminder_prefix = "CRITICAL CONSTRAINT: You MUST execute coding changes"
    reminder_text = (
        "\n\nCRITICAL CONSTRAINT: You MUST execute coding changes, file modifications, "
        "and command executions using the designated tool calls. Do NOT simply display "
        "raw code inside markdown text blocks unless explicitly instructed. "
        "Always invoke the relevant tools to make changes."
    )

    new_messages = []
    system_updated = False

    for msg in messages:
        if msg.get("role") == "system" and not system_updated:
            content = msg.get("content", "")
            # Check if reminder is already present
            already_has = False
            if isinstance(content, str):
                already_has = reminder_prefix in content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and reminder_prefix in block.get("text", ""):
                        already_has = True
                        break

            if not already_has:
                new_msg = {k: v for k, v in msg.items()}
                if isinstance(content, str):
                    new_msg["content"] = content + reminder_text
                elif isinstance(content, list):
                    new_list = list(content)
                    if new_list and isinstance(new_list[-1], dict) and new_list[-1].get("type") == "text":
                        new_last = {k: v for k, v in new_list[-1].items()}
                        new_last["text"] = new_last.get("text", "") + reminder_text
                        new_list[-1] = new_last
                    else:
                        new_list.append({"type": "text", "text": reminder_text})
                    new_msg["content"] = new_list
                new_messages.append(new_msg)
                system_updated = True
                continue

        system_updated = system_updated or msg.get("role") == "system"
        new_messages.append(msg)  # Reuse original — no copy needed

    if not system_updated:
        new_messages.insert(0, {"role": "system", "content": reminder_text.strip()})

    return new_messages

def add_cache_control_to_message(message: dict) -> dict:
    """Helper to inject cache_control parameter into a message's content block."""
    content = message.get("content")
    if not content:
        return message
        
    new_msg = {k: v for k, v in message.items()}
    
    if isinstance(content, str):
        # Convert string content to content block list with cache_control
        new_msg["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    elif isinstance(content, list):
        # If it's already a list of blocks, add cache_control to the last block
        # only if it doesn't already have one (prevents double-wrap on re-injection).
        if content:
            new_list = []
            for block in content:
                if isinstance(block, dict):
                    new_list.append({k: v for k, v in block.items()})
                else:
                    new_list.append(block)
            # Only add cache_control if the last block doesn't already have it
            if isinstance(new_list[-1], dict) and "cache_control" not in new_list[-1]:
                new_list[-1]["cache_control"] = {"type": "ephemeral"}
            new_msg["content"] = new_list
            
    return new_msg

def inject_prompt_caching(messages: List[dict]) -> List[dict]:
    """Injects Anthropic-style cache_control breakpoints to maximize KV caching efficiency."""
    if not messages:
        return messages
        
    new_messages = []
    for msg in messages:
        new_msg = {k: v for k, v in msg.items()}
        new_messages.append(new_msg)
        
    # Set cache control breakpoints (up to 4 allowed by Anthropic):
    # 1. The system prompt (index 0 if role is 'system', otherwise the first user message)
    # 2. The message at index N-2 (usually the last assistant response, to cache the entire history prefix)
    # Always add cache breakpoint to message at index 0 (system or first user message).
    new_messages[0] = add_cache_control_to_message(new_messages[0])

    # Second breakpoint: message at index N-2.
    # Only apply when there are >= 3 messages so it points to a DISTINCT message from index 0.
    # With exactly 2 messages, N-2 == 0, so we'd re-apply to the same message (double-wrap bug).
    N = len(new_messages)
    if N >= 3:
        new_messages[N - 2] = add_cache_control_to_message(new_messages[N - 2])
        
    return new_messages

def strip_prompt_caching(messages: List[dict]) -> List[dict]:
    """Strips any cache_control annotations from message content blocks to avoid API errors on unsupported models."""
    if not messages:
        return messages

    # Fast path: bail out early if no message has cache_control blocks
    if not any(
        isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and "cache_control" in b for b in m["content"])
        for m in messages
        if isinstance(m, dict)
    ):
        return messages

    clean_messages = []
    for msg in messages:
        new_msg = {k: v for k, v in msg.items()}
        content = new_msg.get("content")
        if isinstance(content, list):
            new_list = []
            has_only_text = True
            text_acc = []
            for block in content:
                if isinstance(block, dict):
                    clean_block = {k: v for k, v in block.items() if k != "cache_control"}
                    new_list.append(clean_block)
                    if block.get("type") == "text" and len(block) <= 3: # type, text, cache_control
                        text_acc.append(block.get("text", ""))
                    else:
                        has_only_text = False
                else:
                    new_list.append(block)
                    has_only_text = False
            
            # If the block list was just text blocks and we stripped cache_control, convert back to a clean simple string
            if has_only_text and text_acc:
                new_msg["content"] = "".join(text_acc)
            else:
                new_msg["content"] = new_list
        clean_messages.append(new_msg)
    return clean_messages

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

    # Rate limiting — protect against runaway agent loops burning through budget
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Slow down — your API budget will thank you.",
        )
        
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
        
    messages = body.get("messages", [])
    requested_model = body.get("model", "")
    stream = body.get("stream", False)
    
    # Cap max_tokens to prevent "Response too long" errors in Copilot.
    # Chinese models (especially DeepSeek with thinking) can generate very long responses
    # that exceed Copilot's internal limits. Default cap: 16384 tokens.
    max_tokens_cap = config.get("routing", {}).get("max_tokens_cap", 16384)
    current_max_tokens = body.get("max_tokens")
    if current_max_tokens is None or current_max_tokens > max_tokens_cap:
        body["max_tokens"] = max_tokens_cap
    
    enable_system_reminder = config.get("routing", {}).get("enable_system_reminder", True)
    if enable_system_reminder:
        messages = inject_reminder(messages)
        body["messages"] = messages
    
    # Check if request targets auto-routing
    # Automatically routes custom-router, default, empty, or any standard model alias without a '/' slash (like gpt-4o, claude-3-5-sonnet, deepseek-chat)
    should_route = (
        requested_model in ["custom-router", "default", ""] or
        "/" not in requested_model
    )
    
    # Retrieve the pre-initialized global connection client (reuses TCP/SSL pool with HTTP/2 support)
    client = get_http_client(request)
    
    if should_route:
        # Image detection: if the conversation contains image content blocks,
        # route to a vision-capable model (qwen3.7-max) regardless of category,
        # since most of the fleet is text-only and would 400 on image_url.
        if messages_contain_images(messages):
            vision_model = config.get("routing", {}).get("vision_model", "qwen/qwen3.7-max")
            category = "vision"
            primary_model = vision_model
            fallback_models = [vision_model, config["routing"]["default_model"]]
            logger.info(f"Image content detected -> routing to vision model: {vision_model}")
        else:
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

    # OpenCode gateway does not pass through image content blocks for any model
    # (even Qwen-VL lineage). Strip image blocks and replace with a text placeholder
    # so the request is accepted instead of 400-ing with "Unexpected item type in content."
    if PROVIDER == "opencode" and messages_contain_images(body.get("messages", [])):
        stripped = strip_unsupported_content_blocks({"messages": body.get("messages", [])}, "all", {"text"})
        body = body.copy()
        body["messages"] = stripped["messages"]
        logger.info("Stripped image content blocks (OpenCode gateway does not support image passthrough).")
            
    api_base = config.get(PROVIDER, {}).get("api_base")
    if not api_base:
        api_base = "https://openrouter.ai/api/v1" if PROVIDER == "openrouter" else "https://opencode.ai/zen/go/v1"
        
    enable_response_caching = config.get("routing", {}).get("enable_response_caching", True)
    enable_prompt_caching = config.get("routing", {}).get("enable_prompt_caching", True)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": LOCAL_URL,
        "X-Title": "Chinese LLM Router Gateway"
    }
    
    if PROVIDER == "openrouter" and enable_response_caching:
        headers["X-OpenRouter-Cache"] = "true"

    if PROVIDER == "openrouter" and enable_prompt_caching:
        if "claude" in primary_model.lower() or "anthropic" in primary_model.lower():
            body["messages"] = inject_prompt_caching(messages)
    
    async def stream_generator(response: httpx.Response, selected_model: str) -> Generator[bytes, None, None]:
        actual_model = selected_model
        prompt_tokens = 0
        completion_tokens = 0
        mapper = ToolCallIndexMapper()
        xml_parser = XMLToJSONStreamParser()
        has_thinking = False
        
        async for line in response.aiter_lines():
            if not line:
                continue
            
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    if xml_parser.buffer:
                        flush_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": xml_parser.buffer
                                }
                            }]
                        }
                        yield f"data: {json.dumps(flush_chunk)}\n\n".encode("utf-8")
                        xml_parser.buffer = ""
                    if has_thinking:
                        # Close the thinking tag before sending [DONE]
                        close_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": "\n</thinking>\n\n"
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
                    
                    # Convert reasoning/thinking tokens to standard content for clients if enabled
                    enable_thinking_mapping = config.get("routing", {}).get("enable_thinking_mapping", False)
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        content = delta.get("content", "")
                        
                        if reasoning and enable_thinking_mapping:
                            if not has_thinking:
                                has_thinking = True
                                delta["content"] = "<thinking>\n" + reasoning
                            else:
                                delta["content"] = reasoning
                            # Remove the raw reasoning fields when mapping to content
                            delta.pop("reasoning_content", None)
                            delta.pop("reasoning", None)
                            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                        elif reasoning:
                            # Thinking mapping disabled: pass reasoning_content through unchanged.
                            # Clients like Copilot render it in a dedicated thinking UI.
                            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                        else:
                            if has_thinking and content and enable_thinking_mapping:
                                has_thinking = False
                                content = "\n</thinking>\n\n" + content
                            
                            if content:
                                parsed_items = xml_parser.feed(content)
                                for item in parsed_items:
                                    new_chunk = {
                                        "id": chunk.get("id", ""),
                                        "object": chunk.get("object", ""),
                                        "created": chunk.get("created", 0),
                                        "model": chunk.get("model", ""),
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {}
                                            }
                                        ]
                                    }
                                    if "content" in item:
                                        new_chunk["choices"][0]["delta"]["content"] = item["content"]
                                    elif "tool_calls" in item:
                                        new_chunk["choices"][0]["delta"]["tool_calls"] = item["tool_calls"]
                                        
                                    if "model" in chunk:
                                        new_chunk["model"] = chunk["model"]
                                    if "usage" in chunk:
                                        new_chunk["usage"] = chunk["usage"]
                                    yield f"data: {json.dumps(new_chunk)}\n\n".encode("utf-8")
                            else:
                                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    else:
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                        
                    if "model" in chunk:
                        actual_model = chunk["model"]
                    if "usage" in chunk and chunk["usage"]:
                        prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                        completion_tokens = chunk["usage"].get("completion_tokens", 0)
                except Exception:
                    yield f"{line}\n".encode("utf-8")
            else:
                yield f"{line}\n".encode("utf-8")
                    
        if prompt_tokens > 0 or completion_tokens > 0:
            logger.info(f"Stream completed. Category: {category} | Model: {selected_model} -> {actual_model} | Tokens: {prompt_tokens}+{completion_tokens}")
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
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"Upstream API error: {r.text[:300]}")
            try:
                response_json = r.json()
            except Exception:
                raise HTTPException(status_code=502, detail=f"Upstream returned non-JSON body: {r.text[:300]}")
            response_json = sanitize_json(response_json)
            actual_model = response_json.get("model", actual_primary)

            # Guard: never return a response with no choices to the client.
            # Some upstream providers return 200 with an error body or empty choices,
            # which crashes clients like Copilot ("Response contained no choices").
            if not response_json.get("choices"):
                err_msg = (response_json.get("error", {}) or {}).get("message") or "Upstream returned no choices"
                logger.error(f"Upstream {actual_primary} returned 200 with no choices: {json.dumps(response_json)[:300]}")
                raise HTTPException(status_code=502, detail=f"Upstream returned no choices: {err_msg}")
            
            # Map reasoning_content and XML tool calls in non-streaming responses
            if "choices" in response_json and response_json["choices"]:
                message = response_json["choices"][0].get("message", {})
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                content = message.get("content", "")
                enable_thinking_mapping = config.get("routing", {}).get("enable_thinking_mapping", False)
                if reasoning and enable_thinking_mapping:
                    if content:
                        content = "<thinking>\n" + reasoning + "\n</thinking>\n\n" + content
                    else:
                        content = "<thinking>\n" + reasoning + "\n</thinking>\n\n"
                # Always strip raw reasoning fields so clients don't loop on them
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
                
                # Check for XML tool calls in content
                if content and ("<invoke" in content or "<function_calls>" in content):
                    tool_calls = []
                    # Find all invoke blocks
                    invokes = re.finditer(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', content, re.DOTALL)
                    tool_call_index = 0
                    for match in invokes:
                        tool_name = match.group(1)
                        inner_xml = match.group(2)
                        
                        # Extract parameters
                        params = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', inner_xml, re.DOTALL)
                        params_dict = {}
                        for param_name, param_val in params:
                            clean_val = html.unescape(param_val.strip())
                            if clean_val.lower() == "true":
                                params_dict[param_name] = True
                            elif clean_val.lower() == "false":
                                params_dict[param_name] = False
                            else:
                                try:
                                    if "." in clean_val:
                                        params_dict[param_name] = float(clean_val)
                                    else:
                                        params_dict[param_name] = int(clean_val)
                                except ValueError:
                                    try:
                                        params_dict[param_name] = json.loads(clean_val)
                                    except Exception:
                                        params_dict[param_name] = clean_val
                        
                        tool_calls.append({
                            "id": f"call_xml_{tool_call_index}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(params_dict)
                            }
                        })
                        tool_call_index += 1
                    
                    if tool_calls:
                        message["tool_calls"] = tool_calls
                        # Remove the XML block from content
                        clean_content = re.sub(r'<function_calls>.*?</function_calls>', '', content, flags=re.DOTALL)
                        clean_content = re.sub(r'<invoke.*?</invoke>', '', clean_content, flags=re.DOTALL)
                        content = clean_content.strip()
                
                message["content"] = content or None
                    
            usage = response_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            logger.info(f"Request completed. Category: {category} | Model: {actual_primary} -> {actual_model} | Tokens: {prompt_tokens}+{completion_tokens}")

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
