import sys
import os
import json
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import app, strip_prompt_caching

client = TestClient(app)

def test_openai_compliant_errors():
    print("Testing OpenAI compliant error handlers...")
    
    # 1. Trigger a 401 Unauthorized by providing an empty API key
    # (Mock ENV_API_KEY so the proxy raises a local 401 instead of using the config key)
    import proxy
    old_key = proxy.ENV_API_KEY
    proxy.ENV_API_KEY = None
    try:
        response = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }, headers={"Authorization": "Bearer "}) # empty token
        
        assert response.status_code == 401
        err_json = response.json()
        assert "error" in err_json
        assert err_json["error"]["type"] == "invalid_request_error"
        assert err_json["error"]["code"] == "401"
        assert "message" in err_json["error"]
        print("Compliant 401 error handler passed!")
    finally:
        proxy.ENV_API_KEY = old_key

    # 2. Trigger a 400 Validation Error by sending bad request payload format
    response_bad = client.post("/v1/chat/completions", content="invalid json")
    assert response_bad.status_code == 400
    err_json_bad = response_bad.json()
    assert "error" in err_json_bad
    assert err_json_bad["error"]["type"] == "invalid_request_error"
    assert err_json_bad["error"]["code"] in ["validation_error", "400"]
    print("Compliant 400 validation error handler passed!")

def test_models_list_compliance():
    print("\nTesting models list compliance...")
    response = client.get("/v1/models")
    assert response.status_code == 200
    res_json = response.json()
    
    assert "data" in res_json
    model_ids = {m["id"] for m in res_json["data"]}
    
    # Standard aliases must be in models list
    assert "gpt-4o" in model_ids
    assert "claude-3-5-sonnet" in model_ids
    assert "deepseek-chat" in model_ids
    assert "custom-router" in model_ids
    print("Models list compliance passed!")

if __name__ == "__main__":
    test_openai_compliant_errors()
    test_models_list_compliance()
    print("\nALL SEAMLESS COMPLIANCE UNIT TESTS PASSED SUCCESSFULLY!")
