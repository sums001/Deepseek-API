"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at either:
    - http://localhost:8000/v1
    - http://localhost:8000

Example:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints supported (both with and without /v1 prefix):
    POST /v1/chat/completions or /chat/completions (stream=true supported)
    GET  /v1/models or /models
    GET  /v1/models/{model} or /models/{model}
    POST /v1/embeddings or /embeddings
    GET  /healthz or /
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient

from .config import (
    API_KEY,
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    resolve_model_type,
    should_enable_thinking,
)
from .openai_format import completion_response, messages_to_prompt, stream_chunks
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest, EmbeddingRequest

load_dotenv()

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.2.0")

# Enable CORS for browser frontends (NextChat, OpenWebUI, LibreChat, n8n, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.

    This touches Playwright's sync API, so callers must invoke it OFF the event
    loop (via run_in_threadpool); calling it inside the asyncio loop raises
    "Playwright Sync API inside the asyncio loop"."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _error(message: str, status: int = 500, err_type: str = "server_error", code: Optional[str] = None):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code or str(status),
            }
        },
    )


def _verify_auth(authorization: Optional[str] = None):
    """If API_KEY is configured in the environment, verify incoming Bearer token."""
    if not API_KEY:
        return True
    if not authorization:
        return False
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == API_KEY
    return authorization == API_KEY


@app.get("/")
@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "deepseek-openai-api"}


@app.get("/models")
@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
                "permission": [],
                "root": name,
                "parent": None,
            }
            for name in MODEL_MAP
        ],
    }


@app.get("/models/{model}")
@app.get("/v1/models/{model}")
def retrieve_model(model: str, authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not is_known_model(model):
        return _error(f"The model `{model}` does not exist", status=404, err_type="model_not_found")

    return {
        "id": model,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "deepseek",
        "permission": [],
        "root": model,
        "parent": None,
    }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    thinking_enabled = should_enable_thinking(req.model, req.thinking)
    search_enabled = bool(req.search)
    prompt = messages_to_prompt(req.messages)

    try:
        # Off the event loop: get_client() uses Playwright's sync API, which
        # errors if run inside the asyncio loop.
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    include_usage = bool(req.stream_options and req.stream_options.include_usage)

    if req.stream:
        def gen():
            stream = client.stream(
                prompt,
                conversation_id=req.conversation_id,
                model=model_type,
                thinking=thinking_enabled,
                search=search_enabled,
            )
            yield from stream_chunks(
                req.model,
                stream,
                prompt=prompt,
                include_usage=include_usage,
            )

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    try:
        reply = await run_in_threadpool(
            client.chat,
            prompt,
            req.conversation_id,
            model_type,
            thinking_enabled,
            search_enabled,
        )
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    return completion_response(req.model, reply.text, prompt, reply.conversation_id)


@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(
    req: EmbeddingRequest,
    authorization: Optional[str] = Header(None),
):
    """Fallback embeddings endpoint so OpenAI clients / agents checking embeddings do not break."""
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    data = []
    for idx, _ in enumerate(inputs):
        # Provide standard 1536-dim normalized dummy embedding
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": [0.0] * 1536,
        })

    return {
        "object": "list",
        "data": data,
        "model": req.model or "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": len(inputs) * 5,
            "total_tokens": len(inputs) * 5,
        },
    }
