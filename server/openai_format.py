"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Iterable, List, Optional, Tuple, Dict, Any

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


def _format_tools_system_prompt(tools: List[dict]) -> str:
    """Format tools list into system instructions for tool calls."""
    tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
    return (
        "# Tools\n\n"
        "You have access to the following tools:\n\n"
        f"```json\n{tools_json}\n```\n\n"
        "When calling a tool, respond with a tool call using the following XML format:\n"
        '<tool_call name="TOOL_NAME">\n'
        '<parameter name="PARAM_NAME" string="true">PARAM_VALUE</parameter>\n'
        '</tool_call>\n'
    )


def messages_to_prompt(messages: List[ChatMessage], tools: Optional[List[dict]] = None) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message is sent verbatim unless tools are provided. Multi-turn /
    system-prompted conversations are serialised with role labels and a trailing
    'Assistant:' cue so the model continues in the right voice.
    """
    lines = []

    if tools:
        lines.append(f"System: {_format_tools_system_prompt(tools)}")

    if len(messages) == 1 and messages[0].role == "user" and not tools:
        return _text_of(messages[0].content)

    for m in messages:
        if m.role == "system":
            lines.append(f"System: {_text_of(m.content)}")
        elif m.role == "user":
            lines.append(f"User: {_text_of(m.content)}")
        elif m.role == "assistant":
            parts = []
            txt = _text_of(m.content)
            if txt:
                parts.append(txt)
            if m.tool_calls:
                for tc in m.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass

                    tc_xml = f'<tool_call name="{name}">\n'
                    if isinstance(args, dict):
                        for k, v in args.items():
                            tc_xml += f'<parameter name="{k}" string="true">{v}</parameter>\n'
                    else:
                        tc_xml += f'<parameter name="input" string="true">{args}</parameter>\n'
                    tc_xml += '</tool_call>'
                    parts.append(tc_xml)
            lines.append(f"Assistant: {' '.join(parts)}")
        elif m.role in ("tool", "function"):
            tool_id = m.tool_call_id or m.name or "tool"
            lines.append(f"Tool Output ({tool_id}): {_text_of(m.content)}")
        else:
            label = _ROLE_LABELS.get(m.role, m.role.capitalize())
            lines.append(f"{label}: {_text_of(m.content)}")

    lines.append("Assistant:")
    return "\n\n".join(lines)


def _parse_xml_tool_call(full_match_str: str, name_attr: Optional[str], inner_content: str) -> Optional[Dict[str, Any]]:
    tool_name = name_attr
    if not tool_name:
        name_match = re.search(r'<name>(.*?)</name>', inner_content, re.DOTALL)
        if name_match:
            tool_name = name_match.group(1).strip()

    if not tool_name:
        try:
            data = json.loads(inner_content.strip())
            if isinstance(data, dict) and ("name" in data or "tool" in data or "function" in data):
                name = data.get("name") or data.get("tool") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or data.get("input") or {}
                args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                return {"name": name, "arguments": args_str}
        except Exception:
            pass
        return None

    args: Dict[str, Any] = {}
    param_matches = re.findall(r'<parameter\s+name=["\']([^"\']+)["\'][^>]*>(.*?)</parameter>', inner_content, re.DOTALL)
    if param_matches:
        for p_name, p_val in param_matches:
            args[p_name] = p_val.strip()
    else:
        args_match = re.search(r'<arguments>(.*?)</arguments>', inner_content, re.DOTALL)
        if args_match:
            args_str = args_match.group(1).strip()
            try:
                parsed = json.loads(args_str)
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    args = {"input": args_str}
            except Exception:
                args = {"input": args_str}
        else:
            tags = re.findall(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', inner_content, re.DOTALL)
            for t_name, t_val in tags:
                if t_name not in ("name", "arguments"):
                    args[t_name] = t_val.strip()

    args_json = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
    return {"name": tool_name, "arguments": args_json}


def extract_tool_calls_and_clean_text(content: str) -> Tuple[Optional[str], Optional[List[dict]]]:
    """Extract tool calls (XML or JSON format) from model output and return cleaned content and tool calls."""
    if not content:
        return None, None

    tool_calls: List[dict] = []
    cleaned_content = content

    xml_pattern = r'<tool_call(?:\s+name=["\']([^"\']+)["\'])?\s*>(.*?)</tool_call>'
    matches = list(re.finditer(xml_pattern, content, re.DOTALL))

    if matches:
        for m in matches:
            full_match_str = m.group(0)
            name_attr = m.group(1)
            inner_content = m.group(2)

            parsed = _parse_xml_tool_call(full_match_str, name_attr, inner_content)
            if parsed:
                call_id = "call_" + uuid.uuid4().hex[:12]
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": parsed
                })
                cleaned_content = cleaned_content.replace(full_match_str, "")
    else:
        json_pattern = r'```(?:json)?\s*(\{\s*"name"\s*:.*?\})\s*```'
        json_matches = list(re.finditer(json_pattern, content, re.DOTALL))
        for m in json_matches:
            full_match_str = m.group(0)
            json_str = m.group(1)
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "name" in data:
                    name = data["name"]
                    args = data.get("arguments", data.get("parameters", {}))
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    call_id = "call_" + uuid.uuid4().hex[:12]
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args_str}
                    })
                    cleaned_content = cleaned_content.replace(full_match_str, "")
            except Exception:
                pass

    cleaned_content = cleaned_content.strip()
    final_content = cleaned_content if cleaned_content else None
    final_tool_calls = tool_calls if tool_calls else None
    return final_content, final_tool_calls


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None, reasoning_content: str = None) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content or "")

    cleaned_content, tool_calls = extract_tool_calls_and_clean_text(content)

    message: Dict[str, Any] = {"role": "assistant", "content": cleaned_content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

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

    buffer = ""
    has_tool_calls = False

    for d in stream:
        if d:
            chunk_type = getattr(d, "chunk_type", "RESPONSE")
            if chunk_type == "THINK":
                yield frame({"reasoning_content": str(d)})
            else:
                text_chunk = str(d)
                buffer += text_chunk

                # Check if buffer contains completed tool calls
                if "</tool_call>" in buffer:
                    # Extract up to </tool_call>
                    end_idx = buffer.find("</tool_call>") + len("</tool_call>")
                    tool_block = buffer[:end_idx]
                    buffer = buffer[end_idx:]

                    cleaned_text, tool_calls = extract_tool_calls_and_clean_text(tool_block)
                    if cleaned_text:
                        yield frame({"content": cleaned_text})
                    if tool_calls:
                        has_tool_calls = True
                        yield frame({"tool_calls": tool_calls})
                elif "<tool_call" not in buffer:
                    # Buffer has no tool call tags, safe to flush
                    # Keep last 10 chars in buffer in case <tool_call tag is split across chunks
                    if len(buffer) > 15:
                        flush_text = buffer[:-15]
                        buffer = buffer[-15:]
                        yield frame({"content": flush_text})

    # Flush remaining buffer at end of stream
    if buffer:
        cleaned_text, tool_calls = extract_tool_calls_and_clean_text(buffer)
        if cleaned_text:
            yield frame({"content": cleaned_text})
        if tool_calls:
            has_tool_calls = True
            yield frame({"tool_calls": tool_calls})

    finish_reason = "tool_calls" if has_tool_calls else "stop"
    conversation_id = getattr(stream, "conversation_id", None)
    yield frame({}, finish=finish_reason, extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"

