"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    role: str
    # content is a plain string, or a list of parts (OpenAI vision-style). We only
    # read text parts; non-text parts are ignored.
    content: Union[str, List[dict], None] = None
    # New: for assistant messages that contain tool calls (from an OpenAI client).
    tool_calls: Optional[List[dict]] = None
    # New: for tool-result messages (the id of the call being answered).
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # Tools to enable for this request, independent of the model. OpenAI clients
    # pass these via extra_body: `thinking` (DeepThink), `search` (web).
    thinking: bool = False
    search: bool = False
    # New: OpenAI-style function definitions. When provided, the model is
    # instructed to emit a JSON block {"tool_calls":[...]} in its response, which
    # we translate back into OpenAI's tool_calls shape.
    tools: Optional[List[dict]] = None
    # New: how to handle tool_choice, forwarded as a prompt-level instruction:
    # "none" -> never call a function, "required" -> must call one, or a dict
    # {"type": "function", "function": {"name": ...}} to force a specific one.
    tool_choice: Optional[Union[str, dict]] = None
    # Accepted for compatibility but not all are forwarded to DeepSeek.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
