#!/usr/bin/env python3
"""Opt-in synthetic Responses reproducer: size once, replay the same JSON before/after a rebuild."""
import argparse
import copy
import json
from pathlib import Path
import urllib.request

TOOL = {
    "type": "function", "name": "check_code", "description": "Check a synthetic code file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"], "additionalProperties": False},
}


def fixture(model, padding):
    return {
        "model": model,
        "instructions": "You are a coding assistant testing synthetic fixtures. Follow the latest user instruction.",
        "input": [
            {"role": "user", "content": "Synthetic repository context. Padding is inert code-comment material:\n#"
             + " x" * padding + "\ndef add(a, b): return a + b\n"},
            {"type": "function_call", "call_id": "call_fixture", "name": "check_code",
             "arguments": '{"path":"synthetic.py"}'},
            {"type": "function_call_output", "call_id": "call_fixture", "output": '{"ok":true,"synthetic":true}'},
            {"role": "user", "content": "Call check_code with path synthetic.py now. "
             "After its result, reply exactly ALIGNMENT_OK."},
        ],
        "tools": [TOOL], "stream": True, "temperature": 0, "max_output_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False}, "store": False,
    }


def chat(body):
    # Restricted translation for this fixture, matching responses_translate.zig; verify final usage too.
    inp = body["input"]
    return {
        "messages": [
            {"role": "system", "content": body["instructions"]}, inp[0],
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": inp[1]["call_id"], "type": "function", "function": {
                    "name": inp[1]["name"], "arguments": inp[1]["arguments"]}}]},
            {"role": "tool", "tool_call_id": inp[2]["call_id"], "content": inp[2]["output"]}, inp[3],
        ],
        "tools": [{"type": "function", "function": {k: v for k, v in TOOL.items() if k != "type"}}],
        "chat_template_kwargs": body["chat_template_kwargs"],
    }


def post(url, path, payload):
    req = urllib.request.Request(url.rstrip("/") + path, payload, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return response.read()


def completed(raw):
    events = [json.loads(line[6:]) for line in raw.decode().splitlines()
              if line.startswith("data: ") and line[6:] != "[DONE]"]
    failures = [e for e in events if e.get("type") in ("error", "response.failed")]
    if failures:
        raise RuntimeError(json.dumps(failures))
    responses = [e["response"] for e in events if e.get("type") == "response.completed"]
    if not responses or responses[-1]["status"] != "completed":
        raise RuntimeError("No completed response in SSE stream")
    return responses[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Native server base URL, without /v1")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--tokens", type=int, required=True, help="Rendered input tokens (not text length)")
    parser.add_argument("--request", type=Path, required=True, help="Synthetic request JSON to create or replay")
    parser.add_argument("--output", type=Path, help="New directory for SSE evidence (required with --run)")
    parser.add_argument("--run", action="store_true", help="Send inference and a tool continuation; otherwise only size")
    args = parser.parse_args()
    if args.tokens < 1 or (args.run and args.output is None):
        parser.error("positive --tokens and --output with --run are required")
    if not args.request.exists():
        padding = 0
        for _ in range(5):
            body = fixture(args.model, padding)
            count = json.loads(post(args.url, "/tokenize", json.dumps(chat(body)).encode()))["count"]
            if count == args.tokens:
                with args.request.open("x") as output:
                    json.dump(body, output, indent=2)
                break
            padding += args.tokens - count
            if padding < 0:
                raise ValueError("Token target is smaller than the fixture")
        else:
            raise RuntimeError("Could not size fixture exactly")
    if not args.run:
        print("Request ready; inference was not sent")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    payload = args.request.read_bytes()
    body = json.loads(payload)
    count = json.loads(post(args.url, "/tokenize", json.dumps(chat(body)).encode()))["count"]
    if count != args.tokens:
        raise ValueError(f"Saved request renders to {count}, expected {args.tokens}")
    raw = post(args.url, "/v1/responses", payload)
    (args.output / "first.sse").write_bytes(raw)
    response = completed(raw)
    if response["usage"]["input_tokens"] != args.tokens:
        raise ValueError("Responses usage differs from the tokenizer target")
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    if len(calls) != 1 or calls[0]["name"] != "check_code" or json.loads(calls[0]["arguments"]) != {"path": "synthetic.py"}:
        raise ValueError("Unexpected tool call")
    continuation = copy.deepcopy(body)
    continuation["input"] += response["output"] + [{"type": "function_call_output",
        "call_id": calls[0]["call_id"], "output": '{"ok":true,"synthetic":true}'}]
    raw = post(args.url, "/v1/responses", json.dumps(continuation).encode())
    (args.output / "continuation.sse").write_bytes(raw)
    result = completed(raw)
    text = "".join(part.get("text", "") for item in result["output"] for part in item.get("content", []))
    if text.strip() != "ALIGNMENT_OK":
        raise ValueError("Unexpected continuation text")
    print(json.dumps({"completed": True, "tool_continuation_completed": True, "usage": response["usage"]}))


if __name__ == "__main__":
    main()
