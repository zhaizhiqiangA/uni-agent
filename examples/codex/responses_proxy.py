#!/opt/codex/bin/python
"""Local Responses-to-Chat-Completions bridge for the sidecar runtime."""

from __future__ import annotations

import argparse
import http.client
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from uuid import uuid4


def _content_text(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ValueError("message content must be a string or list")
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text", "text", "summary_text"}:
            text = item.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def responses_to_chat(payload):
    messages = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    input_value = payload.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "message")
            if item_type == "message":
                role = item.get("role")
                if role == "developer":
                    role = "system"
                if role in {"system", "user", "assistant"}:
                    messages.append({"role": role, "content": _content_text(item.get("content", ""))})
            elif item_type == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or uuid4().hex)
                arguments = item.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": item.get("name", ""), "arguments": arguments},
                            }
                        ],
                    }
                )
            elif item_type in {"function_call_output", "custom_tool_call_output", "computer_call_output"}:
                output = item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                        "content": output,
                    }
                )
            elif item_type == "reasoning":
                text = _content_text(item.get("summary") or [])
                if text:
                    messages.append({"role": "assistant", "content": "", "reasoning_content": text})
    else:
        raise ValueError("input must be a string or list")

    if not messages:
        raise ValueError("input must contain at least one message")

    tools = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )

    request = {"model": payload.get("model", "default"), "messages": messages, "stream": False}
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
    ):
        if source in payload:
            request[target] = payload[source]
    return request


def chat_to_response(payload, model):
    choices = payload.get("choices") or []
    message = choices[0].get("message") if choices else {}
    message = message if isinstance(message, dict) else {}
    output = []

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        output.append(
            {
                "id": f"reasoning_{uuid4().hex}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "status": "completed",
            }
        )

    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        call_id = str(call.get("id") or uuid4().hex)
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        output.append(
            {
                "id": call_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": function.get("name", ""),
                "arguments": arguments,
            }
        )

    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )

    usage = payload.get("usage") or {}
    return {
        "id": f"resp_{uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": content if isinstance(content, str) else "",
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        "parallel_tool_calls": True,
    }


def response_events(response):
    yield "response.created", {"response": {**response, "status": "in_progress", "output": []}}
    yield "response.in_progress", {"response": {**response, "status": "in_progress", "output": []}}
    for index, item in enumerate(response["output"]):
        yield "response.output_item.added", {"output_index": index, "item": {**item, "status": "in_progress"}}
        if item["type"] == "message":
            content = item["content"][0]
            common = {"output_index": index, "content_index": 0, "item_id": item["id"]}
            yield "response.content_part.added", {**common, "part": {**content, "text": ""}}
            yield "response.output_text.delta", {**common, "delta": content["text"]}
            yield "response.output_text.done", {**common, "text": content["text"]}
            yield "response.content_part.done", {**common, "part": content}
        elif item["type"] == "function_call":
            common = {"output_index": index, "item_id": item["id"]}
            yield "response.function_call_arguments.delta", {**common, "delta": item["arguments"]}
            yield "response.function_call_arguments.done", {**common, "arguments": item["arguments"]}
        yield "response.output_item.done", {"output_index": index, "item": item}
    yield "response.completed", {"response": response}


class Handler(BaseHTTPRequestHandler):
    upstream = ""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_error(404)

    def do_POST(self):
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            request = responses_to_chat(payload)
            upstream = urlsplit(self.upstream)
            path = upstream.path.rstrip("/") + "/chat/completions"
            connection_cls = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
            connection = connection_cls(upstream.hostname, upstream.port, timeout=7200)
            headers = {"content-type": "application/json"}
            authorization = self.headers.get("authorization")
            if authorization:
                headers["authorization"] = authorization
            connection.request("POST", path, json.dumps(request).encode(), headers)
            upstream_response = connection.getresponse()
            body = upstream_response.read()
            if upstream_response.status >= 400:
                self.send_response(upstream_response.status)
                self.send_header("content-type", upstream_response.getheader("content-type") or "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            response = chat_to_response(json.loads(body), str(payload.get("model") or "default"))
            if payload.get("stream") is True:
                chunks = []
                for event_type, fields in response_events(response):
                    data = json.dumps({"type": event_type, **fields}, ensure_ascii=False)
                    chunks.append(f"event: {event_type}\ndata: {data}\n\n".encode())
                encoded = b"".join(chunks)
                content_type = "text/event-stream"
            else:
                encoded = json.dumps(response, ensure_ascii=False).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:
            encoded = json.dumps({"error": {"message": str(exc), "type": "proxy_error"}}).encode()
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    def log_message(self, *_args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1:18999")
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    host, port = args.listen.rsplit(":", 1)
    Handler.upstream = args.upstream.rstrip("/")
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
