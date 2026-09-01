from __future__ import annotations

import json

from examples.codex.responses_proxy import chat_to_response, responses_to_chat


def test_responses_to_chat_lowers_messages_tools_and_outputs():
    result = responses_to_chat(
        {
            "model": "policy",
            "instructions": "system rule",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix"}]},
                {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ],
            "tools": [
                {"type": "function", "name": "exec_command", "description": "run", "parameters": {"type": "object"}},
                {"type": "web_search"},
            ],
            "max_output_tokens": 64,
        }
    )

    assert result["messages"][0] == {"role": "system", "content": "system rule"}
    assert result["messages"][1] == {"role": "user", "content": "fix"}
    assert result["messages"][2]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert result["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
    assert result["tools"] == [
        {
            "type": "function",
            "function": {"name": "exec_command", "description": "run", "parameters": {"type": "object"}},
        }
    ]
    assert result["max_tokens"] == 64


def test_chat_to_response_preserves_text_and_tool_calls():
    response = chat_to_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "exec_command", "arguments": json.dumps({"cmd": "pwd"})},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        "policy",
    )

    assert response["object"] == "response"
    assert response["output_text"] == "done"
    assert response["output"][0]["type"] == "function_call"
    assert response["output"][0]["call_id"] == "call_1"
    assert response["output"][1]["type"] == "message"
    assert response["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
