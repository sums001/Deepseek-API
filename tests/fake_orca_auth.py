"""A local stand-in for OrcaRouter's consent screen and code exchange.

It implements just enough of the documented protocol to drive the client's
PKCE flow end to end over real loopback TCP: ``GET /auth`` behaves like the
consent screen (redirecting to the callback with a code, or with an error),
and ``POST /api/v1/auth/keys`` validates the code, its single-use/10-minute
lifetime, the challenge method, and the SHA-256 challenge before returning a
fake key.

No request in this module ever leaves the machine, and the "keys" it mints are
obviously fake.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlsplit

FAKE_KEY = "sk-orca-fake-0000000000000000000000"
FAKE_USER_ID = "12345"
CODE_TTL = 600.0


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class FakeOrcaAuth:
    """A configurable local OrcaRouter auth service."""

    def __init__(
        self,
        *,
        behavior: str = "approve",
        granted_scope: str = "api",
        key: str = FAKE_KEY,
        exchange_status: Optional[int] = None,
    ):
        self.behavior = behavior
        self.granted_scope = granted_scope
        self.key = key
        self.exchange_status = exchange_status

        self.codes: Dict[str, dict] = {}
        self.authorize_requests: List[dict] = []
        self.exchange_requests: List[dict] = []
        self._seq = 0
        self._lock = threading.Lock()

        holder = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                return

            def _send(self, status: int, body: bytes, content_type: str,
                      extra_headers: Optional[dict] = None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra_headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, payload: dict):
                self._send(status, json.dumps(payload).encode(), "application/json")

            def do_GET(self):  # noqa: N802
                url = urlsplit(self.path)
                if url.path != "/auth":
                    self._json(404, {"error": "not found"})
                    return
                params = {k: v[0] for k, v in parse_qs(url.query).items()}
                holder.authorize_requests.append(params)

                if params.get("scope") not in (None, "", "api", "connector"):
                    self._json(400, {"error": "invalid_scope"})
                    return

                callback = params.get("callback_url", "")
                if holder.behavior == "deny":
                    self._redirect(callback, params, error="access_denied")
                    return
                if holder.behavior == "bad_state":
                    self._redirect(callback, params, state="not-the-state")
                    return

                code = holder._mint(params)
                if callback == "oob":
                    body = f"<html><body>code: {code}</body></html>".encode()
                    self._send(200, body, "text/html; charset=utf-8")
                    return
                self._redirect(callback, params, code=code)

            def _redirect(self, callback: str, params: dict, *, code=None, error=None,
                          state=None):
                if not callback or callback == "oob":
                    self._json(400, {"error": "no redirect target"})
                    return
                query = {"state": state if state is not None else params.get("state", "")}
                if code:
                    query["code"] = code
                if error:
                    query["error"] = error
                location = f"{callback}?{urlencode(query)}"
                self._send(302, b"", "text/plain", {"Location": location})

            def do_POST(self):  # noqa: N802
                url = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode() or "{}")
                except Exception:
                    body = {}

                if url.path == "/api/v1/auth/keys":
                    holder.exchange_requests.append(body)
                    self._json(*holder._exchange(body))
                    return
                self._json(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    # --- lifecycle --------------------------------------------------------

    def __enter__(self) -> "FakeOrcaAuth":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass

    @property
    def auth_base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    # --- behaviour --------------------------------------------------------

    def _mint(self, params: dict) -> str:
        with self._lock:
            self._seq += 1
            code = f"fake-code-{self._seq}"
            self.codes[code] = {
                "challenge": params.get("code_challenge", ""),
                "method": params.get("code_challenge_method", ""),
                "state": params.get("state", ""),
                "created_at": time.time(),
                "used": False,
            }
            return code

    def expire(self, code: str) -> None:
        self.codes[code]["created_at"] = time.time() - CODE_TTL - 1

    def mark_used(self, code: str) -> None:
        self.codes[code]["used"] = True

    def _exchange(self, body: dict) -> tuple:
        if self.exchange_status == 429:
            return 429, {"error": "rate_limited"}
        if self.exchange_status is not None:
            return self.exchange_status, {"error": "forced"}

        code = body.get("code")
        verifier = body.get("code_verifier")
        method = body.get("code_challenge_method")

        record = self.codes.get(code or "")
        if record is None:
            return 403, {"error": "invalid_grant", "error_description": "unknown code"}
        if record["used"]:
            return 403, {"error": "invalid_grant", "error_description": "code already used"}
        if time.time() - record["created_at"] > CODE_TTL:
            return 403, {"error": "invalid_grant", "error_description": "code expired"}
        if method and record["method"] and method != record["method"]:
            return 400, {"error": "invalid_request", "error_description": "method mismatch"}
        expected = b64url(hashlib.sha256((verifier or "").encode()).digest())
        if not record["challenge"] or expected != record["challenge"]:
            return 403, {"error": "invalid_grant", "error_description": "verifier mismatch"}

        record["used"] = True
        return 200, {"key": self.key, "user_id": FAKE_USER_ID, "scope": self.granted_scope}
