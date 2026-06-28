"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


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


def messages_to_prompt(messages: List[ChatMessage], tools: Optional[List[dict]] = None) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer."""
    # If there are no tools and only a single user message, send it verbatim to keep it fast
    if not tools and len(messages) == 1 and messages[0].role == "user":
        return _text_of(messages[0].content)

    lines = []
    
    # Inject tool instructions if tools are present
    if tools:
        tool_desc = []
        for t in tools:
            if t.get("type") == "function":
                f = t["function"]
                tool_desc.append(
                    f"- Name: {f['name']}\n"
                    f"  Description: {f.get('description', '')}\n"
                    f"  Parameters Schema: {json.dumps(f.get('parameters', {}))}"
                )
        
        system_instruction = (
            "SYSTEM INSTRUCTION: You have access to the following tools/functions:\n\n"
            + "\n\n".join(tool_desc)
            + "\n\nIf you need to use a tool to answer the user's question, you MUST reply with a JSON code block matching the following schema and NOTHING else:\n"
            "```json\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "name": "TOOL_NAME",\n'
            '      "arguments": {}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "where 'arguments' is a JSON object containing parameter values. Do not output any conversational text before or after this JSON block when calling a tool. If you do not need any tools, reply normally in plain text."
        )
        lines.append(system_instruction)

    for m in messages:
        if m.role == "tool":
            lines.append(f"Tool Result ({m.name or m.tool_call_id}): {_text_of(m.content)}")
        elif m.role == "assistant" and m.tool_calls:
            tc_data = {"tool_calls": [{"name": tc.get("function", {}).get("name", tc.get("name")), "arguments": tc.get("function", {}).get("arguments", tc.get("arguments"))} for tc in m.tool_calls]}
            lines.append(f"Assistant: ```json\n{json.dumps(tc_data, indent=2)}\n```")
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


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None,
                        tool_calls: List[dict] = None,
                        finish_reason: str = "stop") -> dict:
    """A full (non-streaming) OpenAI chat.completion object."""
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    
    message = {
        "role": "assistant",
        "content": content if not tool_calls else None
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
        
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


def stream_chunks(model: str, stream: Iterable[str]) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    `stream` is the client's stream object; after it's consumed we read its
    `.conversation_id` and attach it to the final chunk.
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
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})
    for d in stream:
        if d:
            yield frame({"content": d})
    conversation_id = getattr(stream, "conversation_id", None)
    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
