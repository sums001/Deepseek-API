"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.

Because DeepSeek's model doesn't speak OpenAI's native tool-call format, we
implement a *shim*: incoming `tools` become a prompt-level instruction that
asks the model to emit JSON like {"tool_calls":[...]}; we then parse that JSON
out of the model's text and translate it into OpenAI's `tool_calls` shape in
the response.
"""

from __future__ import annotations

import ast
import json
import time
import uuid
from typing import Iterable, List, Optional, Union

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}

_BACKSLASH = chr(92)
_DOUBLE_QUOTE = chr(34)
_SINGLE_QUOTE = chr(39)


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def _format_tools(tools: List[dict],
                  tool_choice: Optional[Union[str, dict]] = None) -> str:
    """Turn OpenAI tool definitions into a prompt-level instruction.

    DeepSeek has no native tool schema, so we describe the functions in prose
    and ask the model to emit a strict JSON block. The sample is written with
    *double quotes* on purpose: the model tends to mirror the example's style,
    and a single-quoted sample makes it emit non-JSON output.
    """
    if not tools:
        return ""

    tool_descs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        tool_descs.append(
            f"- {name}: {desc}\n"
            f"  Parameters: {json.dumps(params, ensure_ascii=False, indent=2)}"
        )

    if not tool_descs:
        return ""

    instruction = (
        "You have access to the following functions. To call a function, "
        "respond with ONLY a single valid JSON object (double quotes, no "
        "markdown, no extra text) in exactly this format:\n"
        '{"tool_calls": [{"name": "function_name", "arguments": {"arg": "value"}}]}\n'
        "If you do not need to call a function, respond normally in plain text.\n\n"
        "Available functions:\n" + "\n".join(tool_descs)
    )

    if tool_choice == "none":
        instruction += "\n\nDo not call any functions."
    elif tool_choice == "required":
        instruction += "\n\nYou must call one of the functions."
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        forced = tool_choice.get("function", {}).get("name", "")
        instruction += "\n\nYou must call the function " + forced + "."

    return instruction


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} block that contains a tool_calls key.

    Finds the tool_calls key, walks back to the enclosing open brace, then
    brace-matches — string-aware, so braces inside string literals are ignored.
    """
    idx = text.find("tool_calls")
    if idx == -1:
        return None
    start = text.rfind("{", 0, idx)
    if start == -1:
        return None

    depth = 0
    in_str = False
    quote = ""
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == _BACKSLASH:
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch == _DOUBLE_QUOTE or ch == _SINGLE_QUOTE:
                in_str = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _loads_tolerant(blob: str):
    """Parse JSON, falling back to Python-literal syntax (single quotes)."""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(blob)
    except Exception:
        return None


def parse_tool_calls(text: str) -> Optional[List[dict]]:
    """Extract tool calls from the model's raw text output.

    Returns OpenAI-style `tool_calls` objects, or `None` when no function-call
    block is found. Tolerates the model emitting single-quoted / Python-literal
    output instead of strict JSON.
    """
    if not text:
        return None

    blob = _extract_json_object(text)
    if blob is None:
        return None

    data = _loads_tolerant(blob)
    if not isinstance(data, dict):
        return None

    raw_calls = data.get("tool_calls")
    if not isinstance(raw_calls, list):
        return None

    result = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name")
        args = tc.get("arguments", {})
        if not name:
            continue
        if isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps(args, ensure_ascii=False)
        result.append({
            "id": "call_" + uuid.uuid4().hex[:8],
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return result if result else None


def messages_to_prompt(
    messages: List[ChatMessage],
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message with no tools is sent verbatim. Multi-turn /
    system-prompted conversations are serialised with role labels and a
    trailing 'Assistant:' cue so the model continues in the right voice.

    `tools` (when present) is rendered as an instruction block at the top.
    Tool-result and prior assistant-with-tool-calls messages are rendered so
    the model sees the call/answer pair.
    """
    tool_instruction = _format_tools(tools, tool_choice) if tools else ""

    if len(messages) == 1 and messages[0].role == "user" and not tools:
        return _text_of(messages[0].content)

    lines = []
    if tool_instruction:
        lines.append(f"System: {tool_instruction}")

    for m in messages:
        if m.role == "tool":
            lines.append(f"Tool result: {_text_of(m.content)}")
        elif m.role == "assistant" and m.tool_calls:
            calls = []
            for tc in m.tool_calls:
                func = tc.get("function", {})
                calls.append({
                    "name": func.get("name"),
                    "arguments": func.get("arguments"),
                })
            lines.append(
                f"Assistant: {json.dumps({'tool_calls': calls}, ensure_ascii=False)}"
            )
        else:
            label = _ROLE_LABELS.get(m.role, m.role.capitalize())
            lines.append(f"{label}: {_text_of(m.content)}")

    lines.append("Assistant:")
    return "\n\n".join(lines)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(
    model: str,
    content: str,
    prompt: str,
    conversation_id: str = None,
    tool_calls: Optional[List[dict]] = None,
) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    When `tool_calls` is present the message is shaped to OpenAI's spec:
    `content` becomes `null` and `finish_reason` is `"tool_calls"`.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content)

    message = {"role": "assistant", "content": content}
    finish_reason = "stop"

    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
        finish_reason = "tool_calls"

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(
    model: str,
    stream: Iterable[str],
    prompt: str = "",
) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}` frames) for a streamed completion.

    Because the model may emit a JSON tool-call block mid-stream, we buffer the
    full output first, then decide whether the response is a tool call or plain
    text. Text responses still stream token by token.
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    full_text = "".join(stream)
    tool_calls = parse_tool_calls(full_text)

    if tool_calls:
        yield frame(
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            finish="tool_calls",
        )
        yield frame({}, finish="stop")
        yield "data: [DONE]\n\n"
        return

    conversation_id = getattr(stream, "conversation_id", None)

    yield frame({"role": "assistant", "content": ""})
    for d in full_text:
        if d:
            yield frame({"content": d})
    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
