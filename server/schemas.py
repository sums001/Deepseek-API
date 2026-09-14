"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_MODEL


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: Optional[bool] = False


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = "text"  # "text" or "json_object"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, List[Any], None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False

    # Standard OpenAI Chat Completion parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    stream_options: Optional[StreamOptions] = None
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None

    # DeepSeek / custom extension parameters
    conversation_id: Optional[str] = None
    thinking: Optional[bool] = None
    search: Optional[bool] = None


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: Union[str, List[str], List[int], List[List[int]]]
    model: Optional[str] = "text-embedding-ada-002"
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None
