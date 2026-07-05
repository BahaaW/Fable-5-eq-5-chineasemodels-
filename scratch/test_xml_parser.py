import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxy import XMLToJSONStreamParser

def test_xml_to_json_parser():
    print("Running XMLToJSONStreamParser tests...")
    
    # Test Case 1: Simple text with no XML
    parser = XMLToJSONStreamParser()
    out = parser.feed("Hello, this is a plain text stream chunk.")
    assert len(out) == 1
    assert "content" in out[0]
    assert out[0]["content"] == "Hello, this is a plain text stream chunk."
    print("Test Case 1 Passed!")

    # Test Case 2: Standard XML tool call block
    parser = XMLToJSONStreamParser()
    chunks = []
    chunks.extend(parser.feed("<function_calls>\n  <invoke name=\"run_python_code\">\n"))
    chunks.extend(parser.feed("    <code>print('Hello World')</code>\n  </invoke>\n</function_calls>"))
    
    # We expect one tool_calls chunk
    tool_calls = [c for c in chunks if "tool_calls" in c]
    assert len(tool_calls) == 1
    tc = tool_calls[0]["tool_calls"][0]
    assert tc["function"]["name"] == "run_python_code"
    args = json.loads(tc["function"]["arguments"])
    assert args["code"] == "print('Hello World')"
    print("Test Case 2 Passed!")

    # Test Case 3: Mixed content and XML tool call
    parser = XMLToJSONStreamParser()
    chunks = []
    chunks.extend(parser.feed("Let me run that command for you:\n"))
    chunks.extend(parser.feed("<function_calls>\n  <invoke name=\"execute_command\">\n"))
    chunks.extend(parser.feed("    <cmd>git status</cmd>\n"))
    chunks.extend(parser.feed("  </invoke>\n</function_calls>\nDone!"))
    
    # Verify content chunks and tool calls
    contents = [c["content"] for c in chunks if "content" in c]
    tool_calls = [c["tool_calls"][0] for c in chunks if "tool_calls" in c]
    
    assert "Let me run that command for you:\n" in contents
    assert "\nDone!" in contents
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "execute_command"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["cmd"] == "git status"
    print("Test Case 3 Passed!")

    # Test Case 4: Multiple parameters with different types
    parser = XMLToJSONStreamParser()
    chunks = parser.feed(
        "<function_calls>\n"
        "  <invoke name=\"search_database\">\n"
        "    <query>SELECT * FROM users</query>\n"
        "    <limit>10</limit>\n"
        "    <active>true</active>\n"
        "  </invoke>\n"
        "</function_calls>"
    )
    tool_calls = [c["tool_calls"][0] for c in chunks if "tool_calls" in c]
    assert len(tool_calls) == 1
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["query"] == "SELECT * FROM users"
    assert args["limit"] == 10
    assert args["active"] is True
    print("Test Case 4 Passed!")

    # Test Case 5: Unfinished XML tags (should buffer and not output content)
    parser = XMLToJSONStreamParser()
    out = parser.feed("<function_calls>\n  <invoke name=\"test_tool\">\n    <arg>some value")
    assert len(out) == 0  # Should be buffering
    
    # Feed the rest
    out2 = parser.feed("</arg>\n  </invoke>\n</function_calls>")
    assert len(out2) == 1
    assert "tool_calls" in out2[0]
    args = json.loads(out2[0]["tool_calls"][0]["function"]["arguments"])
    assert args["arg"] == "some value"
    print("Test Case 5 Passed!")

    print("\nALL XML-TO-JSON PARSER UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_xml_to_json_parser()
