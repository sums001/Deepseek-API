"""
Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into text

    from deepseek.auth import get_session
    from deepseek.client import DeepSeekClient

    client = DeepSeekClient(get_session())
    print(client.chat("Hello!"))                 # full reply
    for chunk in client.stream("Tell a joke"):   # streamed
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from .auth import Session, get_session
from .pow import DeepSeekPow

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"

# DeepSeek's mode pill, sent as `model_type` in the completion body. "default" is
# Instant (the fast model); "expert" is the stronger, slower model. Omitting the
# field lets the backend pick, so we always send one explicitly.
DEFAULT_MODEL_TYPE = "default"

# A conversation_id is an opaque "<chat_session_id>:<last_message_id>" token. It
# carries everything needed to resume a thread, so the client stays stateless.
_CID_SEP = ":"


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation."""

    text: str
    conversation_id: str
    reasoning_text: str = ""
    citations: list[dict] = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")
    biz = data.get("data", {}).get("biz_data")
    if biz is None:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
    ):
        # `allow_interactive=False` makes session resolution non-blocking: it
        # uses a cached/headless session and raises LoginRequired instead of
        # opening a browser window. The server passes False (see server/api.py).
        self.session = session or get_session(allow_interactive=allow_interactive)
        self._pow = DeepSeekPow()
        # The wasmtime Store behind the PoW solver is not reentrant; serialise
        # access so concurrent server requests don't corrupt it.
        self._pow_lock = threading.Lock()
        self._http = httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(120.0, read=300.0),
        )

    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        r = self._http.post("/api/v0/chat_session/create", json={})
        r.raise_for_status()
        return _biz(r.json())["chat_session"]["id"]

    def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        r = self._http.post(
            "/api/v0/chat/create_pow_challenge", json={"target_path": target_path}
        )
        r.raise_for_status()
        challenge = _biz(r.json())["challenge"]
        with self._pow_lock:
            return self._pow.make_header(challenge)

    # --- public API ---------------------------------------------------------

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value: "default" (Instant) or
        "expert"; it defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
        """
        if conversation_id and model is not None:
            raise ValueError(
                "`model` cannot be set together with `conversation_id`; a thread's "
                "model is fixed when it is created. Pass `model` only on the first turn."
            )
        session_id, parent_id = _decode_cid(conversation_id)
        if session_id is None:
            # New thread: select the model (default when unspecified).
            session_id = self.create_chat_session()
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            # Resuming: let the existing thread's model stand (send no model_type).
            model_type = None
        return _Stream(self, prompt, session_id, parent_id, model_type, thinking, search)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> Reply:
        """Return the complete reply (`.text`) plus its `.conversation_id`."""
        s = self.stream(prompt, conversation_id=conversation_id,
                        model=model, thinking=thinking, search=search)
        content_parts: list[str] = []
        for chunk_type, text in s:
            if chunk_type == "content":
                content_parts.append(text)
        return Reply(text="".join(content_parts),
                     reasoning_text=s.reasoning_text,
                     conversation_id=s.conversation_id,
                     citations=s.citations)

    def close(self) -> None:
        self._http.close()


class _Stream:
    """Iterable of (type, text) chunks. After it's consumed, `.conversation_id`
    holds the token for resuming the conversation and `.reasoning_text` contains
    the model's internal thinking (if any)."""

    def __init__(self, client: "DeepSeekClient", prompt: str, session_id: str,
                 parent_id: Optional[int], model: str,
                 thinking: bool, search: bool):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._message_id: Optional[int] = None
        self._thinking_parts: list[str] = []

    def __iter__(self) -> Iterator[tuple[str, str]]:
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": [],
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        # Only select a model on a new thread; on resume the thread keeps its own.
        if self._model is not None:
            body["model_type"] = self._model
        # PoW challenges are short-lived, so solve right before the request.
        headers = {"x-ds-pow-response": self._client._pow_header()}
        meta: dict = {}
        self._thinking_parts = []
        self._citations = []
        with self._client._http.stream(
            "POST", COMPLETION_PATH, json=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            for chunk_type, text in _parse_sse(resp.iter_lines(), meta):
                if chunk_type == "thinking":
                    self._thinking_parts.append(text)
                yield (chunk_type, text)
        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]
        self._citations = meta.get("citations", [])

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)

    @property
    def reasoning_text(self) -> str:
        """The model's internal reasoning (only available after iteration)."""
        return "".join(self._thinking_parts)

    @property
    def citations(self) -> list:
        """Search result citations collected during parsing (only available after iteration)."""
        return getattr(self, "_citations", [])


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[tuple[str, str]]:
    """Turn DeepSeek's SSE completion stream into ``(type, text)`` chunks.

    Yields ``("thinking", text)`` for THINK-fragment content and
    ``("content", text)`` for RESPONSE-fragment content.  TOOL_* fragments
    (search, open) are tracked but not yielded directly — their result URLs
    and titles are accumulated into ``meta["citations"]`` when *meta* is given.

    The stream sends an initial snapshot frame whose ``v`` is the full response
    object, then path-addressed append / batch frames that modify fragments.
    We maintain an ordered list of fragment types so we know which kind of
    content is arriving at ``fragments/-1``.

    If *meta* is given, the assistant's ``message_id`` is recorded into it,
    and search-result citations are collected into ``meta["citations"]``.
    """
    active_path: Optional[str] = None
    emitted_initial = False
    fragments: list[Optional[str]] = []  # ordered fragment types by index

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        v = obj.get("v")

        # ── snapshot frame ──────────────────────────────────────────────
        if isinstance(v, dict) and "response" in v:
            if meta is not None:
                _capture_message_id(meta, v)
            fragments.clear()
            for frag in v["response"].get("fragments", []):
                ft = frag.get("type") if isinstance(frag, dict) else None
                fragments.append(ft)
                if isinstance(frag, dict):
                    if ft in ("TOOL_SEARCH", "SEARCH"):
                        _capture_search_results(meta, frag.get("results", []))
                    elif ft == "TOOL_OPEN":
                        _capture_source(meta, frag.get("result"))
                    elif ft in ("RESPONSE", "THINK") and frag.get("content"):
                        active_path = "response/fragments/-1/content"
                        if not emitted_initial:
                            emitted_initial = True
                            yield ("thinking" if ft == "THINK" else "content",
                                   frag["content"])
            continue

        # ── BATCH operation (nested mutations) ──────────────────────────
        if "p" in obj and obj.get("o") == "BATCH" and isinstance(v, list):
            active_path = obj["p"]
            _scan_for_sources(meta, v)
            yield from _process_batch(v, active_path, fragments, meta)
            continue

        # ── bare BATCH (list of ops without outer wrapper) ──────────────
        if "p" not in obj and "o" not in obj and isinstance(v, list):
            _scan_for_sources(meta, v)
            yield from _process_batch(v, active_path or "response", fragments, meta)
            continue

        # ── path-setting frame ──────────────────────────────────────────
        if "p" in obj:
            active_path = obj["p"]
            if meta is not None and active_path.endswith("message_id") \
                    and isinstance(v, int):
                meta["message_id"] = v
            op = obj.get("o")
            if op == "SET" and meta is not None and isinstance(v, (dict, list)):
                _scan_for_sources(meta, v)
                if active_path.endswith("results") and isinstance(v, list):
                    _capture_search_results(meta, v)
            elif op == "APPEND":
                if isinstance(v, str) and active_path.endswith("content"):
                    ft = _last_fragment_type(fragments)
                    yield ("thinking" if ft == "THINK" else "content", v)
                elif active_path.endswith("fragments"):
                    items = v if isinstance(v, list) else [v]
                    for frag in items:
                        ft = frag.get("type") if isinstance(frag, dict) else None
                        fragments.append(ft)
                        if isinstance(frag, dict):
                            if ft in ("TOOL_SEARCH", "SEARCH"):
                                _capture_search_results(meta, frag.get("results", []))
                            elif ft == "TOOL_OPEN":
                                _capture_source(meta, frag.get("result"))
                            elif ft in ("RESPONSE", "THINK") \
                                    and frag.get("content"):
                                yield ("thinking" if ft == "THINK" else "content",
                                       frag["content"])
            elif op is None:
                if isinstance(v, str) and active_path.endswith("content"):
                    ft = _last_fragment_type(fragments)
                    yield ("thinking" if ft == "THINK" else "content", v)
                elif isinstance(v, list) and meta is not None \
                        and active_path.endswith("results"):
                    _scan_for_sources(meta, v)
                    _capture_search_results(meta, v)
            continue

        # ── bare append (path kept from previous frame) ─────────────────
        if isinstance(v, str) and active_path and active_path.endswith("content"):
            ft = _last_fragment_type(fragments)
            yield ("thinking" if ft == "THINK" else "content", v)


def _last_fragment_type(fragments: list) -> Optional[str]:
    """Type of the fragment at index -1 (the most recently added one)."""
    return fragments[-1] if fragments else None


def _capture_source(meta: Optional[dict], result) -> None:
    """Add a single search/open result to ``meta["citations"]``, deduplicating by URL."""
    if meta is None or not isinstance(result, dict):
        return
    url = result.get("url", "")
    if not url:
        return
    citations = meta.setdefault("citations", [])
    if not any(c.get("url") == url for c in citations):
        citations.append({"url": url, "title": result.get("title", "")})


def _capture_search_results(meta: Optional[dict], results: list) -> None:
    """Add all results from a TOOL_SEARCH fragment to the citations list."""
    if meta is None:
        return
    for r in results:
        _capture_source(meta, r)


def _scan_for_sources(meta: Optional[dict], obj, seen_urls: set = None) -> None:
    """Recursively scan *obj* for dicts carrying ``url`` + ``title`` keys.

    This catches search results regardless of the SSE path or fragment type
    they arrive in, making citation extraction resilient to API changes.
    """
    if meta is None:
        return
    if seen_urls is None:
        seen_urls = set()
    if isinstance(obj, dict):
        url = obj.get("url")
        if url and isinstance(url, str) and url.startswith("http") \
                and url not in seen_urls:
            seen_urls.add(url)
            _capture_source(meta, obj)
        for v in obj.values():
            _scan_for_sources(meta, v, seen_urls)
    elif isinstance(obj, list):
        for item in obj:
            _scan_for_sources(meta, item, seen_urls)


def _process_batch(ops: list, base_path: str, fragments: list,
                   meta: Optional[dict] = None) -> Iterator[tuple[str, str]]:
    """Recursively process BATCH operations, tracking fragment types and
    yielding any content appends found inside."""
    for op in ops:
        if not isinstance(op, dict):
            continue
        p = op.get("p", "")
        o = op.get("o", "")
        v = op.get("v")

        resolved = f"{base_path}/{p}" if p else base_path

        if p == "fragments" and o == "APPEND":
            items = v if isinstance(v, list) else [v]
            for frag in items:
                ft = frag.get("type") if isinstance(frag, dict) else None
                fragments.append(ft)
                if isinstance(frag, dict):
                    if ft in ("TOOL_SEARCH", "SEARCH"):
                        _capture_search_results(meta, frag.get("results", []))
                    elif ft == "TOOL_OPEN":
                        _capture_source(meta, frag.get("result"))
                    elif ft in ("RESPONSE", "THINK") and frag.get("content"):
                        yield ("thinking" if ft == "THINK" else "content",
                               frag["content"])
        elif resolved.endswith("content") and o == "APPEND" and isinstance(v, str):
            ft = _last_fragment_type(fragments)
            yield ("thinking" if ft == "THINK" else "content", v)
        elif o == "BATCH" and isinstance(v, list):
            yield from _process_batch(v, resolved, fragments, meta)
        # SET ops (references, status, elapsed_secs) are intentionally ignored.



def _capture_message_id(meta: dict, snapshot: dict) -> None:
    """Best-effort: pull the assistant message_id out of a snapshot frame.

    DeepSeek nests the assistant message under `response`; we check there first,
    then the snapshot root, accepting `message_id` or `id`.
    """
    for container in (snapshot.get("response"), snapshot):
        if isinstance(container, dict):
            mid = container.get("message_id", container.get("id"))
            if isinstance(mid, int):
                meta["message_id"] = mid
                return
