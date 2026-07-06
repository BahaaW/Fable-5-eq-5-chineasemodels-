# Chinease-routers Primer
- Current Phase: Production Refinement Completed
- Completed: Fixed OpenCode Go model prefix error and mapped OpenRouter fallback models dynamically. Refined XMLToJSONStreamParser, system reminder injection, and 5-min timeout limits. Added Fable 5 prompt. Added sanitize_payload_for_model() to fix kimi-k2.7 (temp=1 only) and deepseek-v4-pro (disable thinking when reasoning_content missing from history) 400 errors on opencode.
- Next Steps: Run live benchmarks.
- Blockers: None
