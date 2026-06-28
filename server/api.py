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

import asyncio
import json
import logging
import re
import threading
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired, Session, login, DEFAULT_PROFILE_DIR
from deepseek.client import DeepSeekClient

from .config import (
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    PROXY_ADMIN_TOKEN,
    DEEPSEEK_PROXY_API_KEY,
    SESSION_ALERT_WEBHOOK_URL,
    is_known_model,
    resolve_model_type,
    parse_virtual_model,
    get_all_model_names,
)
from .openai_format import completion_response, messages_to_prompt, stream_chunks
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_proxy")


proxy_key_scheme = HTTPBearer(scheme_name="ProxyApiKeyAuth", auto_error=False)
admin_token_scheme = HTTPBearer(scheme_name="AdminTokenAuth", auto_error=False)


def verify_proxy_api_key(credentials: HTTPAuthorizationCredentials = Depends(proxy_key_scheme)) -> str | None:
    if not DEEPSEEK_PROXY_API_KEY:
        if credentials and credentials.credentials:
            token = credentials.credentials
            if ":" in token:
                return token.split(":", 1)[1]
            return token
        return None

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key in Authorization header."
        )

    token = credentials.credentials
    if token == DEEPSEEK_PROXY_API_KEY:
        return None

    prefix = f"{DEEPSEEK_PROXY_API_KEY}:"
    if token.startswith(prefix):
        return token[len(prefix):]

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key in Authorization header."
    )



def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(admin_token_scheme)):
    if not PROXY_ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session import endpoint is disabled because PROXY_ADMIN_TOKEN is not configured."
        )
    if not credentials or credentials.credentials != PROXY_ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Admin token in Authorization header."
        )


from pathlib import Path

THREADS_MAPPING_FILE = Path(__file__).resolve().parent.parent / "session" / "user_threads.json"
_threads_lock = threading.Lock()


def get_conversation_id_for_user(user_id: str) -> str | None:
    if not THREADS_MAPPING_FILE.exists():
        return None
    try:
        with _threads_lock:
            data = json.loads(THREADS_MAPPING_FILE.read_text(encoding="utf-8"))
            return data.get(user_id)
    except Exception as e:
        logger.error(f"Failed to read user threads mapping: {e}")
        return None


def save_conversation_id_for_user(user_id: str, conversation_id: str) -> None:
    try:
        with _threads_lock:
            data = {}
            if THREADS_MAPPING_FILE.exists():
                try:
                    data = json.loads(THREADS_MAPPING_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[user_id] = conversation_id
            THREADS_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
            THREADS_MAPPING_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info(f"Saved thread mapping for user '{user_id}': {conversation_id}")
    except Exception as e:
        logger.error(f"Failed to save user threads mapping: {e}")


load_dotenv()

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.1.0")
install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))


class SessionImportRequest(BaseModel):
    token: str
    cookies: dict
    user_agent: str
    captured_at: float


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


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(session_keep_alive_loop())


@app.post("/v1/session/import", dependencies=[Depends(verify_admin_token)])
async def import_session(req: SessionImportRequest):
    
    try:
        session_obj = Session(
            token=req.token,
            cookies=req.cookies,
            user_agent=req.user_agent,
            captured_at=req.captured_at
        )
        
        await run_in_threadpool(session_obj.save)
        
        global _client
        with _client_lock:
            if _client is not None:
                try:
                    await run_in_threadpool(_client.close)
                except Exception:
                    pass
            _client = DeepSeekClient(session=session_obj, allow_interactive=SERVER_INTERACTIVE_LOGIN)
            
        logger.info("Session imported successfully and client reloaded.")
        return {"status": "ok", "message": "Session imported successfully and client reloaded."}
    except Exception as e:
        logger.error(f"Failed to import session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import session: {str(e)}"
        )


async def send_webhook_alert(message: str):
    if not SESSION_ALERT_WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "content": f"🚨 **[DeepSeek Proxy Alert]** {message}"
            }
            resp = await client.post(SESSION_ALERT_WEBHOOK_URL, json=payload, timeout=10.0)
            if resp.status_code >= 400:
                logger.error(f"Webhook alert failed with status {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Failed to send webhook alert: {e}")


def verify_and_refresh_session():
    """Verify if the current session is valid; if expired or older than 4 hours, refresh it."""
    global _client
    client = get_client()
    
    session = client.session
    age_seconds = time.time() - session.captured_at
    if age_seconds > 4 * 60 * 60:
        logger.info(f"Session age ({age_seconds / 3600:.1f} hours) exceeds 4 hours. Proactively refreshing headlessly...")
        new_session = login(profile_dir=DEFAULT_PROFILE_DIR, headless=True)
        with _client_lock:
            client.session = new_session
            client._http.cookies.update(new_session.cookies)
            client._http.headers["authorization"] = f"Bearer {new_session.token}"
            client._http.headers["user-agent"] = new_session.user_agent
        logger.info("Session refreshed headlessly successfully (proactive).")
        return

    try:
        # Check WAF/token validity
        client._pow_header()
        logger.debug("Session verified successfully (active WAF and token).")
    except Exception as e:
        logger.warning(f"Session validation ping failed: {e}. Attempting headless recovery...")
        try:
            new_session = login(profile_dir=DEFAULT_PROFILE_DIR, headless=True)
            with _client_lock:
                client.session = new_session
                client._http.cookies.update(new_session.cookies)
                client._http.headers["authorization"] = f"Bearer {new_session.token}"
                client._http.headers["user-agent"] = new_session.user_agent
            logger.info("Session refreshed headlessly successfully after validation failure.")
        except Exception as refresh_error:
            logger.error(f"Headless session recovery failed: {refresh_error}")
            raise LoginRequired(
                f"Headless session recovery failed. Interactive login or manual import required. Details: {refresh_error}"
            )


async def session_keep_alive_loop():
    logger.info("Starting DeepSeek session keep-alive loop.")
    await asyncio.sleep(5)
    while True:
        try:
            await run_in_threadpool(verify_and_refresh_session)
        except LoginRequired as e:
            alert_msg = "DeepSeek session has expired critically. Headless refresh failed. Please run `python -m deepseek.auth` or use `/v1/session/import` to restore access."
            logger.critical(alert_msg)
            await send_webhook_alert(alert_msg)
        except Exception as e:
            logger.error(f"Session keep-alive loop check failed: {e}")
        await asyncio.sleep(1800)


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(verify_proxy_api_key)])
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
            for name in get_all_model_names()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    auth_user_id: str | None = Depends(verify_proxy_api_key)
):
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    # Parse virtual model name to extract thinking/search suffixes and handle fallback for unknown models
    base_model, virtual_thinking, virtual_search, virtual_user_id = parse_virtual_model(req.model)
    thinking = req.thinking or virtual_thinking
    search = req.search or virtual_search
    user_id = req.user or auth_user_id or virtual_user_id

    # Determine conversation_id:
    # 1. From request body
    # 2. From user session mapping if conversation_id is not passed
    conversation_id = req.conversation_id
    if not conversation_id and user_id:
        conversation_id = get_conversation_id_for_user(user_id)

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if conversation_id else resolve_model_type(base_model)

    # Optimize prompt: if resuming, send only the last user message to avoid duplicate history in the DeepSeek thread
    if conversation_id:
        user_msgs = [m for m in req.messages if m.role == "user"]
        if user_msgs:
            from .openai_format import _text_of
            prompt = _text_of(user_msgs[-1].content)
        else:
            prompt = messages_to_prompt(req.messages, tools=req.tools)
    else:
        prompt = messages_to_prompt(req.messages, tools=req.tools)

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
                prompt, conversation_id=conversation_id,
                model=model_type, thinking=thinking, search=search,
            )
            yield from stream_chunks(req.model, stream)
            
            # Once stream is consumed, save mapping if user was specified
            new_cid = getattr(stream, "conversation_id", None)
            if user_id and new_cid:
                save_conversation_id_for_user(user_id, new_cid)

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat, prompt, conversation_id,
            model_type, thinking, search,
        )
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    # Check if the model decided to make a tool call
    tool_calls = None
    finish_reason = "stop"

    if req.tools:
        # Extract the outermost JSON object by finding the first '{' and last '}'
        start = reply.text.find("{")
        end = reply.text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(reply.text[start:end+1])
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    tool_calls = []
                    for raw_tc in data["tool_calls"]:
                        args = raw_tc.get("arguments", {})
                        if isinstance(args, dict):
                            args_str = json.dumps(args)
                        else:
                            args_str = str(args)
                        
                        tc = {
                            "id": "call_" + uuid.uuid4().hex[:8],
                            "type": "function",
                            "function": {
                                "name": raw_tc.get("name"),
                                "arguments": args_str
                            }
                        }
                        tool_calls.append(tc)
                    finish_reason = "tool_calls"
            except Exception:
                pass

    if user_id and reply.conversation_id:
        await run_in_threadpool(save_conversation_id_for_user, user_id, reply.conversation_id)

    return completion_response(
        req.model, reply.text, prompt, reply.conversation_id,
        tool_calls=tool_calls, finish_reason=finish_reason
    )



@app.delete("/v1/chat/session/{user_id}", dependencies=[Depends(verify_proxy_api_key)])
async def delete_user_session(user_id: str):
    conversation_id = get_conversation_id_for_user(user_id)
    if not conversation_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active session found for user '{user_id}'"
        )
    
    # 1. Delete on DeepSeek
    try:
        client = await run_in_threadpool(get_client)
        await run_in_threadpool(client.delete_chat_session, conversation_id)
    except Exception as e:
        logger.error(f"Failed to delete session on DeepSeek for user '{user_id}': {e}")
        # We proceed to clear local mapping even if DeepSeek delete request fails (e.g. if already deleted)
    
    # 2. Delete locally from JSON mapping
    try:
        with _threads_lock:
            if THREADS_MAPPING_FILE.exists():
                data = json.loads(THREADS_MAPPING_FILE.read_text(encoding="utf-8"))
                if user_id in data:
                    del data[user_id]
                    THREADS_MAPPING_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    logger.info(f"Cleared thread mapping for user '{user_id}'")
    except Exception as e:
        logger.error(f"Failed to clear local thread mapping for user '{user_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear local thread mapping: {str(e)}"
        )
    
    return {"status": "ok", "message": f"Session for user '{user_id}' deleted successfully from DeepSeek and local mapping."}


@app.get("/v1/chat/sessions", dependencies=[Depends(verify_proxy_api_key)])
async def list_chat_sessions(days: int = None, page_size: int = 100):
    try:
        client = await run_in_threadpool(get_client)
        sessions = await run_in_threadpool(client.fetch_chat_sessions, page_size)
    except Exception as e:
        logger.error(f"Failed to fetch sessions from DeepSeek: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch sessions: {str(e)}"
        )
    
    # Load reverse mapping: session_id -> user_id
    reverse_mapping = {}
    try:
        if THREADS_MAPPING_FILE.exists():
            with _threads_lock:
                data = json.loads(THREADS_MAPPING_FILE.read_text(encoding="utf-8"))
                for uid, cid in data.items():
                    sid = cid.split(":")[0]
                    reverse_mapping[sid] = uid
    except Exception as e:
        logger.error(f"Failed to load reverse mapping: {e}")

    # Calculate time cutoff
    cutoff = None
    if days is not None:
        cutoff = time.time() - (days * 86400)

    filtered_sessions = []
    for s in sessions:
        updated_at = s.get("updated_at")
        if cutoff and (updated_at is None or updated_at < cutoff):
            continue
            
        sid = s.get("id")
        filtered_sessions.append({
            "id": sid,
            "title": s.get("title"),
            "pinned": s.get("pinned", False),
            "model_type": s.get("model_type"),
            "updated_at": updated_at,
            "user_id": reverse_mapping.get(sid)
        })

    return filtered_sessions
