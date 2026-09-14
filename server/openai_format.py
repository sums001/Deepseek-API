"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable, List, Optional

from .schemas import ChatMessage

_ROLE_LABELS = {
    "system": "System",
    "developer": "Developer",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "function": "Function",
}


def _text_of(content: Any) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # OpenAI standard part format: {"type": "text", "text": "..."}
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    # DeepSeek web does not yet ingest raw image URLs directly via text prompt
                    parts.append("[Image attached]")
        return "\n".join(parts)
    return str(content)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message is sent verbatim. Multi-turn / system-prompted
    conversations are serialised with role labels and a trailing 'Assistant:'
    cue so the model continues in the right voice.
    """
    if len(messages) == 1 and messages[0].role in ("user",):
        return _text_of(messages[0].content)

    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(m.role.lower(), m.role.capitalize())
        content = _text_of(m.content)
        # Include tool/function details if present
        if m.tool_calls:
            try:
                tc_str = json.dumps(m.tool_calls)
                content = f"{content}\n[Tool Calls]: {tc_str}".strip()
            except Exception:
                pass
        lines.append(f"{label}: {content}")

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
    conversation_id: Optional[str] = None,
) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    Includes all standard OpenAI fields: system_fingerprint, logprobs, usage,
    and finishes with conversation_id for seamless thread resumption.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    created = _now()
    cid = _id()

    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": "fp_deepseek_web",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": None,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "prompt_tokens_details": {
                "cached_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
            },
        },
        "conversation_id": conversation_id,
    }


def stream_chunks(
    model: str,
    stream: Iterable[str],
    prompt: str = "",
    include_usage: bool = False,
) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    Follows the official OpenAI streaming protocol:
    1. Initial chunk with role='assistant'
    2. Delta chunks with content='...'
    3. Final choice chunk with finish_reason='stop'
    4. Optional usage chunk if include_usage is requested
    5. 'data: [DONE]\\n\\n'
    """
    cid, created = _id(), _now()
    total_text = []

    def frame(delta: dict, finish=None, extra: Optional[dict] = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "fp_deepseek_web",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish,
                }
            ],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role with empty content
    yield frame({"role": "assistant", "content": ""})

    for d in stream:
        if d:
            total_text.append(d)
            yield frame({"content": d})

    conversation_id = getattr(stream, "conversation_id", None)
    # Stop frame
    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})

    # If stream_options: {"include_usage": true} was requested, emit an empty choices chunk with usage
    if include_usage:
        full_content = "".join(total_text)
        pt, ct = _est_tokens(prompt), _est_tokens(full_content)
        usage_obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "fp_deepseek_web",
            "choices": [],
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            },
        }
        yield f"data: {json.dumps(usage_obj, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
