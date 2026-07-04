# Chinease-routers — Project Rules

This folder houses the dynamic Chinese LLM routing proxy.

## Setup & Run Commands

- Install dependencies: `pip install -r requirements.txt`
- Run local routing proxy: `python proxy.py`
- Run test benchmark script: `python test_router.py`

## Core Outputs & Configuration

- `config.yaml`: Centralized model mappings, primary selection, and fallback lists.
- `proxy.py`: Local OpenAI-compatible HTTP server running at `http://localhost:8005/v1` for completions (keys configurable directly in `config.yaml` or loaded from environment).
- `test_router.py`: Command-line benchmark showing routing classification accuracy and exact transaction costs.

## Routing Logic Guidelines

- Intercept requests targeting OpenAI endpoint compatibility.
- Perform regex-based categorizations first. Fallback to cheap LLM classification if ambiguous.
- Map requests to the target Chinese frontier model:
  - **Coding**: `deepseek/deepseek-v4-pro`
  - **Web/Design**: `z-ai/glm-5.2`
  - **Agentic / Tool-use**: `moonshotai/kimi-k2.7-code`
  - **Reasoning / Math / Long-docs**: `minimax/minimax-m3`
  - **General**: `qwen/qwen3.7-max`
- Always inject the `models` array to OpenRouter containing the primary and fallback models. Never escalate to expensive US models unless requested explicitly.
