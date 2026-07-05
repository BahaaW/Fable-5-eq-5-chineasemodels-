import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import inject_prompt_caching, strip_prompt_caching

def test_prompt_caching_logic():
    print("Running Prompt Caching / KV Caching unit tests...")

    # Test Case 1: Minimal conversation (1 message)
    # Should only cache index 0
    messages = [
        {"role": "user", "content": "Hello, this is a long prompt."}
    ]
    injected = inject_prompt_caching(messages)
    assert len(injected) == 1
    assert isinstance(injected[0]["content"], list)
    assert injected[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    print("Test Case 1 (1 Message Injection) Passed!")

    # Test Case 2: Multi-turn conversation (3 messages)
    # Should cache index 0 (system prompt) and N-2 (assistant response)
    messages = [
        {"role": "system", "content": "You are a compiler assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."}
    ]
    injected = inject_prompt_caching(messages)
    assert len(injected) == 3
    # Index 0 (system prompt) should have cache control
    assert isinstance(injected[0]["content"], list)
    assert injected[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Index 1 (N-2, user message in this 3-msg list) should have cache control
    assert isinstance(injected[1]["content"], list)
    assert injected[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Index 2 (last message, N-1) should NOT have cache control (leave it to think on the new prompt)
    assert isinstance(injected[2]["content"], str) or "cache_control" not in injected[2]["content"]
    print("Test Case 2 (3 Messages Injection) Passed!")

    # Test Case 3: Stripping caching parameters for non-Claude fallbacks
    # Should return standard simple text strings and remove all cache_control blocks
    stripped = strip_prompt_caching(injected)
    assert len(stripped) == 3
    # Verify everything was stripped and converted back to standard strings
    assert isinstance(stripped[0]["content"], str)
    assert stripped[0]["content"] == "You are a compiler assistant."
    assert isinstance(stripped[1]["content"], str)
    assert stripped[1]["content"] == "What is 2+2?"
    assert isinstance(stripped[2]["content"], str)
    assert stripped[2]["content"] == "4."
    print("Test Case 3 (Cache Stripping) Passed!")

    print("\nALL PROMPT CACHING UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_prompt_caching_logic()
