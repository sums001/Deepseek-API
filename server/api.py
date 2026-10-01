"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.
"""

from __future__ import annotations

import json
import threading
import time
from functools import partial
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient

from .config import (
    MODEL_MAP,
    ORCAROUTER_PROVIDER,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    is_orcarouter_model,
    resolve_model_type,
    strip_orcarouter_prefix,
)
from .openai_format import completion_response, messages_to_prompt, stream_chunks
from .orcarouter import runtime as orcarouter_runtime
from .orcarouter.provider import OrcaUpstreamError, content_modalities
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest

load_dotenv()

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.1.0")
_rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0)
install_rate_limit(app, _rate_limiter)
# OrcaRouter's own control surface (status, catalog, connect lifecycle).
orcarouter_runtime.install_orcarouter(app)

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


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    """The catalog clients can call.

    DeepSeek's own ids always come first, so an existing client is unaffected.
    When OrcaRouter is configured, its live catalog is merged in - each entry
    keeps its ``vendor/model`` namespace and carries the capability metadata
    the catalog advertised, so a picker can filter without guessing.
    """
    created = int(time.time())
    data = [
        {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
        for name in MODEL_MAP
    ]

    credential = orcarouter_runtime.current_credential()
    if credential is not None:
        try:
            catalog = orcarouter_runtime.get_catalog().fetch("chat", credential=credential)
        except Exception:  # a catalog outage must not break /v1/models
            catalog = None
        if catalog is not None:
            known = set(MODEL_MAP)
            for model in catalog.models:
                if model.id in known:
                    continue
                entry = model.to_public_dict()
                entry.update({"object": "model", "created": created,
                              "owned_by": model.owned_by or ORCAROUTER_PROVIDER})
                data.append(entry)

    return {"object": "list", "data": data}


# DeepSeek's chat body carries DeepThink / web-search / thread extras that the
# OpenAI wire format (and therefore the gateway) has no place for.
_DEEPSEEK_ONLY_FIELDS = ("conversation_id", "thinking", "search")


def _to_orcarouter_messages(req: ChatCompletionRequest) -> List[dict]:
    """Serialize the request back to the OpenAI message shape OrcaRouter expects."""
    messages = []
    for message in req.messages:
        entry = message.model_dump(exclude_none=True)
        messages.append(entry)
    return messages


def _orcarouter_options(req: ChatCompletionRequest) -> dict:
    """Standard OpenAI parameters to forward; provider-specific ones are dropped."""
    dumped = req.model_dump(exclude_none=True)
    for field in _DEEPSEEK_ONLY_FIELDS:
        dumped.pop(field, None)
    for field in ("model", "messages", "stream"):
        dumped.pop(field, None)
    return dumped


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # --- routing ---------------------------------------------------------
    # Which provider answers is a property of the requested model, not of a
    # global switch: a DeepSeek id keeps using the local bridge, an OrcaRouter
    # id goes to the gateway, and an unconfigured OrcaRouter request says so.
    target = "deepseek"
    live_models = None
    if is_orcarouter_model(req.model):
        target = ORCAROUTER_PROVIDER
    else:
        provider = orcarouter_runtime.get_provider()
        if provider.configured:
            try:
                live_models = provider.list_models("chat")
            except Exception:
                live_models = None
            if live_models is not None and req.model in set(live_models.ids):
                target = ORCAROUTER_PROVIDER

    if target == ORCAROUTER_PROVIDER:
        return await _orcarouter_chat(req)

    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    prompt = messages_to_prompt(req.messages)

    try:
        # Off the event loop: get_client() uses Playwright's sync API, which
        # errors if run inside the asyncio loop.
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    if req.stream:
        def gen():
            stream = client.stream(
                prompt, conversation_id=req.conversation_id,
                model=model_type, thinking=req.thinking, search=req.search,
            )
            yield from stream_chunks(req.model, stream)

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat, prompt, req.conversation_id,
            model_type, req.thinking, req.search,
        )
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    return completion_response(req.model, reply.text, prompt, reply.conversation_id)


async def _orcarouter_chat(req: ChatCompletionRequest):
    """Route a chat completion through OrcaRouter.

    Credentials are resolved on this path only through the provider, which
    takes whichever credential the adapters produced - a pasted API key and a
    PKCE login are indistinguishable here, on purpose.
    """
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    provider = orcarouter_runtime.get_provider()
    model = strip_orcarouter_prefix(req.model)
    messages = _to_orcarouter_messages(req)

    # Second-layer guard: the picker is already bound to a filtered list, but a
    # hand-written request must not route a non-chat model through this
    # endpoint, nor attach a modality the selected model never declared.
    try:
        await run_in_threadpool(provider.ensure_model_compatible, model, "chat")
        attachments = content_modalities(messages)
        if attachments:
            await run_in_threadpool(
                provider.ensure_modalities_supported, model, sorted(attachments)
            )
    except OrcaUpstreamError as exc:
        return _orca_error(exc)

    if req.stream:
        def orca_gen():
            try:
                for line in provider.chat_completion(
                    messages, model, stream=True, options=_orcarouter_options(req)
                ):
                    yield line + "\n\n" if line and not line.startswith("data:") else line + "\n\n"
            except OrcaUpstreamError as exc:
                # A stream cannot change its status code mid-flight; emit a
                # final SSE error frame the client can act on instead of
                # truncating silently.
                yield f"data: {json.dumps(_orca_error_body(exc))}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(orca_gen(), media_type="text/event-stream")

    try:
        result = await run_in_threadpool(
            partial(
                provider.chat_completion,
                messages,
                model,
                stream=False,
                options=_orcarouter_options(req),
            )
        )
    except OrcaUpstreamError as exc:
        return _orca_error(exc)
    return JSONResponse(content=result)


def _orca_error_body(exc: OrcaUpstreamError) -> dict:
    return {"error": {"message": str(exc), "type": exc.err_type}}


def _orca_error(exc: OrcaUpstreamError):
    response = JSONResponse(
        status_code=exc.status,
        content=_orca_error_body(exc),
    )
    if exc.retry_after:
        response.headers["Retry-After"] = str(int(exc.retry_after) + 1)
    if exc.needs_reauth or exc.status == 401:
        # Machine-readable so a client can prompt for a (re)login rather than
        # retrying a credential that will never work again.
        response.headers["X-OrcaRouter-Reauth"] = "required"
    return response
