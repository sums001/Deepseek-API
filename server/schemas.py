"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    # `extra="allow"` keeps fields this project does not model itself - tool
    # calls, names, and the OpenAI vision content parts - so a request bound
    # for OrcaRouter is forwarded with its structure intact instead of being
    # silently narrowed to the DeepSeek bridge's own message shape.
    model_config = ConfigDict(extra="allow")

    role: str
    # content is a plain string, or a list of parts (OpenAI vision-style). The
    # DeepSeek bridge only reads text parts; OrcaRouter receives them verbatim.
    content: Union[str, List[dict], None] = None


class ChatCompletionRequest(BaseModel):
    # Extra fields are preserved so the OrcaRouter path can forward standard
    # OpenAI parameters (tools, response_format, penalties, ...) it supports,
    # while the DeepSeek bridge continues to ignore the ones it cannot use.
    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # Tools to enable for this request, independent of the model. OpenAI clients
    # pass these via extra_body: `thinking` (DeepThink), `search` (web).
    thinking: bool = False
    search: bool = False
    # Accepted for compatibility but not all are forwarded to DeepSeek.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
