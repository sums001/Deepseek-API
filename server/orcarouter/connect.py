"""The OrcaRouter connect flow: OAuth 2.0 + PKCE, no client secret.

Two flows are implemented, both ending in the same artifact - a normal,
durable ``sk-orca-...`` API key owned by the user:

* **Flow A - loopback redirect.** The client binds ``127.0.0.1:<random port>``
  *before* opening the browser, so the callback address is known and nothing
  races. The code arrives automatically.
* **Flow B - out-of-band code.** ``callback_url=oob``; the consent screen shows
  the code and the user pastes it back. Needed because this project is often
  run on a headless box or a VPS, where there is no browser on the same machine.

Flow C (RFC 8628 device grant) is implemented as an additional convenience but
is deliberately *not* a substitute for PKCE - it is offered only when asked for
explicitly.

Every flow sends ``code_challenge_method=S256``. That is not optional even on
Flow A: the consent screen lets the user pick "show me a code", which puts the
code into human hands, and under ``plain`` the challenge *is* the verifier.

Failure handling is terminal-by-default: a denial, a state mismatch, an expired
or already-used code, a scope downgrade, and a 429 all end the attempt with an
actionable message instead of retrying in a hot loop.
"""

from __future__ import annotations

import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlsplit

import httpx

from .endpoints import OrcaEndpoints, resolve_endpoints
from .pkce import PkceAttempt, build_authorize_url, states_match

APP_NAME = "DeepSeek API bridge"
DEFAULT_SCOPE = "api"
DEFAULT_LOGIN_TIMEOUT = 300.0
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

# Granted-vs-requested scope ranking. "connector" is the wider grant and
# subsumes "api"; the reverse is a downgrade we must not silently accept.
_SCOPE_RANK = {"api": 1, "connector": 2}

_CALLBACK_PATH = "/cb"

_SUCCESS_PAGE = (
    "<!doctype html><meta charset='utf-8'><title>Connected</title>"
    "<style>body{font:16px/1.5 system-ui,sans-serif;margin:4rem auto;max-width:32rem}"
    "</style><h1>Connected</h1>"
    "<p>OrcaRouter authorized this app. You can close this tab and return to "
    "your terminal.</p>"
)

_FAILURE_PAGE = (
    "<!doctype html><meta charset='utf-8'><title>Not connected</title>"
    "<style>body{font:16px/1.5 system-ui,sans-serif;margin:4rem auto;max-width:32rem}"
    "</style><h1>Not connected</h1>"
    "<p>Authorization was not completed. You can close this tab and return to "
    "your terminal for the reason.</p>"
)


class OrcaAuthError(RuntimeError):
    """Base class for a failed authorization attempt.

    Carries a user-facing ``hint``; never carries a key, a verifier, or a raw
    upstream body.
    """

    terminal = True

    def __init__(self, message: str, *, hint: str = ""):
        super().__init__(message)
        self.hint = hint

    @property
    def full_message(self) -> str:
        return f"{self} {self.hint}".strip()


class OrcaAuthDenied(OrcaAuthError):
    """The user declined on the consent screen (``error=access_denied``)."""


class OrcaAuthTimeout(OrcaAuthError):
    """No callback arrived before the deadline."""


class OrcaAuthCancelled(OrcaAuthError):
    """The caller cancelled (explicit Cancel, sign-out, or server shutdown)."""


class OrcaStateMismatch(OrcaAuthError):
    """The callback's ``state`` did not match the one we sent.

    Nothing else in the callback is trusted after this.
    """


class OrcaExchangeError(OrcaAuthError):
    """The code could not be exchanged for a key."""

    def __init__(self, message: str, *, status: int = 0, hint: str = ""):
        super().__init__(message, hint=hint)
        self.status = status


class OrcaScopeDowngrade(OrcaAuthError):
    """The exchange succeeded but granted less than this app asked for."""


@dataclass(frozen=True)
class ConnectResult:
    """A successful authorization: the durable key plus what was granted."""

    key: str
    user_id: str
    scope: str
    flow: str
    expires_at: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover - asserted in tests
        return (
            f"ConnectResult(key=<redacted>, user_id={self.user_id!r}, "
            f"scope={self.scope!r}, flow={self.flow!r})"
        )


def scope_satisfies(granted: str, required: str = DEFAULT_SCOPE) -> bool:
    """Whether ``granted`` covers ``required``. Unknown/empty fails closed."""
    got = _SCOPE_RANK.get((granted or "").strip())
    need = _SCOPE_RANK.get((required or "").strip())
    if need is None:
        raise ValueError(f"unsupported required scope: {required!r}")
    if got is None:
        return False
    return got >= need


def _sanitize_detail(text: str, limit: int = 240) -> str:
    """Bound and scrub an upstream error body before it reaches the user.

    Never returns anything that looks like a credential, and never returns an
    unbounded body.
    """
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    for marker in ("sk-orca-", "sk-"):
        idx = cleaned.find(marker)
        if idx != -1:
            end = idx
            while end < len(cleaned) and not cleaned[end].isspace():
                end += 1
            cleaned = cleaned[:idx] + "<redacted>" + cleaned[end:]
    return cleaned[:limit]


class _CallbackServer:
    """A single-shot loopback listener for Flow A.

    Binds on construction so the port is known before the browser opens.
    """

    def __init__(self, timeout: float = DEFAULT_LOGIN_TIMEOUT):
        self._timeout = timeout
        self._event = threading.Event()
        self._params: dict = {}
        self._error: Optional[str] = None
        self._served_ok = threading.Event()

        params_holder = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # noqa: D102 - silence the default logger
                return

            def do_GET(self):  # noqa: N802 - stdlib naming
                url = urlsplit(self.path)
                if url.path != _CALLBACK_PATH:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                params_holder._params = query
                ok = not query.get("error") and bool(query.get("code"))
                body = (_SUCCESS_PAGE if ok else _FAILURE_PAGE).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                params_holder._event.set()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
            name="orca-oauth-callback", daemon=True,
        )

    @property
    def callback_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{_CALLBACK_PATH}"

    def __enter__(self) -> "_CallbackServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def wait(self, *, cancel: Optional[threading.Event] = None) -> dict:
        """Block until the browser hits the callback, cancels, or times out."""
        deadline = time.monotonic() + self._timeout
        while True:
            if self._event.wait(0.05):
                return self._params
            if cancel is not None and cancel.is_set():
                raise OrcaAuthCancelled(
                    "OrcaRouter authorization cancelled.",
                    hint="Start the login again when you are ready.",
                )
            if time.monotonic() >= deadline:
                raise OrcaAuthTimeout(
                    "Timed out waiting for the browser to return to "
                    f"127.0.0.1:{self.port}.",
                    hint="Check that the consent tab finished loading, or retry "
                         "with --flow oob to paste the code instead.",
                )

    def close(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass


class OrcaOAuthClient:
    """Drives a PKCE authorization attempt against the configured auth origin."""

    def __init__(
        self,
        endpoints: Optional[OrcaEndpoints] = None,
        *,
        app_name: str = APP_NAME,
        scope: str = DEFAULT_SCOPE,
        timeout: float = DEFAULT_LOGIN_TIMEOUT,
        http_timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        open_browser: Optional[Callable[[str], bool]] = None,
    ):
        self.endpoints = endpoints or resolve_endpoints()
        self.app_name = app_name
        self.scope = scope
        self.timeout = timeout
        self.http_timeout = http_timeout
        self._transport = transport
        self._open_browser = open_browser or webbrowser.open

    # --- HTTP -------------------------------------------------------------

    def _client(self) -> httpx.Client:
        # The auth origin only: this client can never reach the inference base.
        return httpx.Client(
            timeout=httpx.Timeout(self.http_timeout, connect=10.0),
            transport=self._transport,
            follow_redirects=False,
        )

    def exchange(self, code: str, attempt: PkceAttempt) -> ConnectResult:
        """Trade an auth code for a durable key. Uses the *auth* origin."""
        body = attempt.exchange_body(code)
        url = self.endpoints.exchange_url()
        try:
            with self._client() as client:
                resp = client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise OrcaExchangeError(
                f"Could not reach the OrcaRouter auth service ({type(exc).__name__}).",
                hint="Check your network connection and retry.",
            ) from None

        status = resp.status_code
        if status == 200:
            payload = _safe_json(resp)
            key = (payload or {}).get("key") or ""
            if not key:
                raise OrcaExchangeError(
                    "OrcaRouter returned success without a key.",
                    status=status,
                    hint="Retry the login; if it persists, contact OrcaRouter.",
                )
            granted = str((payload or {}).get("scope") or "")
            if not scope_satisfies(granted, self.scope):
                raise OrcaScopeDowngrade(
                    f"OrcaRouter granted scope {granted!r}, which does not cover "
                    f"the requested {self.scope!r}.",
                    hint="Ask a workspace admin for a role that permits this scope, "
                         "then sign in again.",
                )
            return ConnectResult(
                key=key,
                user_id=str((payload or {}).get("user_id") or ""),
                scope=granted,
                flow=attempt.flow,
            )

        detail = _sanitize_detail(_safe_text(resp))
        if status == 400:
            raise OrcaExchangeError(
                "OrcaRouter rejected the exchange request"
                + (f": {detail}" if detail else "."),
                status=status,
                hint="This usually means the PKCE challenge method did not match "
                     "the one sent at authorize time. Start the login again.",
            )
        if status == 403:
            raise OrcaExchangeError(
                "The authorization code was rejected (unknown, expired, already "
                "used, or the verifier did not match).",
                status=status,
                hint="Auth codes are single-use with a 10 minute lifetime. "
                     "Start the login again.",
            )
        if status == 429:
            raise OrcaExchangeError(
                "OrcaRouter is rate limiting new authorizations for this account.",
                status=status,
                hint="At most 10 PKCE keys may be issued per user per 24 hours. "
                     "Reuse the stored key, or wait before signing in again.",
            )
        raise OrcaExchangeError(
            f"OrcaRouter exchange failed with HTTP {status}"
            + (f": {detail}" if detail else "."),
            status=status,
            hint="Retry later; if it persists, contact OrcaRouter.",
        )

    # --- Flow A -----------------------------------------------------------

    def connect_loopback(
        self,
        *,
        on_url: Optional[Callable[[str], None]] = None,
        cancel: Optional[threading.Event] = None,
        login_hint: Optional[str] = None,
    ) -> ConnectResult:
        """Browser + loopback listener. The one-click path."""
        attempt = PkceAttempt.create("loopback")
        with _CallbackServer(timeout=self.timeout) as server:
            url = build_authorize_url(
                self.endpoints.authorize_url(),
                callback_url=server.callback_url,
                attempt=attempt,
                app_name=self.app_name,
                scope=self.scope,
                login_hint=login_hint,
            )
            if on_url:
                on_url(url)
            self._open_browser(url)
            try:
                params = server.wait(cancel=cancel)
            except OrcaAuthError:
                raise

        if not states_match(attempt.state, params.get("state")):
            raise OrcaStateMismatch(
                "The browser callback did not carry the expected state value.",
                hint="The response was discarded. Start the login again.",
            )
        error = params.get("error")
        if error:
            if error == "access_denied":
                raise OrcaAuthDenied(
                    "Authorization was denied.",
                    hint="Approve the request on the consent screen to connect.",
                )
            raise OrcaAuthError(
                f"Authorization failed: {error}",
                hint="Start the login again.",
            )
        code = params.get("code")
        if not code:
            raise OrcaAuthError(
                "The callback carried no authorization code.",
                hint="Start the login again.",
            )
        return self.exchange(code, attempt)

    # --- Flow B -----------------------------------------------------------

    def connect_oob(
        self,
        *,
        on_url: Optional[Callable[[str], None]] = None,
        prompt_code: Optional[Callable[[], str]] = None,
        cancel: Optional[threading.Event] = None,
    ) -> ConnectResult:
        """Out-of-band code. For headless hosts with no reachable loopback.

        ``S256`` is mandatory here: the code is displayed to a human, so it must
        be redeemable only by the process holding the verifier.
        """
        attempt = PkceAttempt.create("oob")
        url = build_authorize_url(
            self.endpoints.authorize_url(),
            callback_url="oob",
            attempt=attempt,
            app_name=self.app_name,
            scope=self.scope,
        )
        if on_url:
            on_url(url)
        self._open_browser(url)

        reader = prompt_code or _read_code_from_stdin
        if cancel is not None and cancel.is_set():
            raise OrcaAuthCancelled("OrcaRouter authorization cancelled.")
        code = (reader() or "").strip()
        if not code:
            raise OrcaAuthCancelled(
                "No code was entered.",
                hint="Start the login again and paste the code shown on the "
                     "consent screen.",
            )
        return self.exchange(code, attempt)

    # --- Flow C (extra, never a replacement for PKCE) ----------------------

    def connect_device(
        self,
        *,
        on_prompt: Optional[Callable[[dict], None]] = None,
        cancel: Optional[threading.Event] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> ConnectResult:
        """RFC 8628 device grant. Extra convenience for keyboard-only hosts."""
        with self._client() as client:
            try:
                start = client.post(
                    self.endpoints.device_code_url(),
                    json={"app_name": self.app_name, "scope": self.scope},
                )
            except httpx.HTTPError as exc:
                raise OrcaAuthError(
                    f"Could not start device authorization ({type(exc).__name__}).",
                    hint="Check your network connection and retry.",
                ) from None
            if start.status_code != 200:
                raise OrcaAuthError(
                    f"Device authorization start failed with HTTP {start.status_code}.",
                    hint="Retry later; if it persists, contact OrcaRouter.",
                )
            payload = _safe_json(start) or {}
            device_code = payload.get("device_code") or ""
            if not device_code:
                raise OrcaAuthError("Device authorization returned no device code.")
            if on_prompt:
                on_prompt(payload)

            interval = float(payload.get("interval") or 5)
            expires_in = float(payload.get("expires_in") or 600)
            deadline = time.monotonic() + min(expires_in, self.timeout)

            while time.monotonic() < deadline:
                sleep(interval)
                if cancel is not None and cancel.is_set():
                    raise OrcaAuthCancelled("OrcaRouter device authorization cancelled.")
                try:
                    resp = client.post(
                        self.endpoints.device_token_url(),
                        json={"device_code": device_code, "grant_type": DEVICE_GRANT_TYPE},
                    )
                except httpx.HTTPError as exc:
                    raise OrcaAuthError(
                        f"Device polling failed ({type(exc).__name__}).",
                        hint="Retry the login.",
                    ) from None
                body = _safe_json(resp) or {}
                err = body.get("error")
                if not err and resp.status_code == 200:
                    key = body.get("key") or ""
                    granted = str(body.get("scope") or "")
                    if not key:
                        raise OrcaAuthError("Device authorization returned no key.")
                    if not scope_satisfies(granted, self.scope):
                        raise OrcaScopeDowngrade(
                            f"OrcaRouter granted scope {granted!r}, which does not "
                            f"cover the requested {self.scope!r}.",
                        )
                    return ConnectResult(
                        key=key,
                        user_id=str(body.get("user_id") or ""),
                        scope=granted,
                        flow="device",
                    )
                if err == "authorization_pending":
                    continue
                if err == "slow_down":
                    interval += 5
                    continue
                if err == "access_denied":
                    raise OrcaAuthDenied(
                        "Device authorization was denied.",
                        hint="Approve the request on the verification page.",
                    )
                if err == "expired_token":
                    raise OrcaAuthTimeout(
                        "The device authorization window closed.",
                        hint="Start the login again for a fresh code.",
                    )
                raise OrcaAuthError(
                    f"Device authorization failed: {err or resp.status_code}",
                    hint="Start the login again.",
                )
        raise OrcaAuthTimeout(
            "Device authorization timed out.",
            hint="Start the login again for a fresh code.",
        )


def _read_code_from_stdin() -> str:  # pragma: no cover - interactive
    try:
        return input("Code: ")
    except EOFError:
        return ""


def _safe_json(resp: httpx.Response) -> Optional[dict]:
    try:
        data = resp.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_text(resp: httpx.Response) -> str:
    try:
        return resp.text
    except Exception:
        return ""


# --- login-lock lifecycle -------------------------------------------------


class OrcaLoginManager:
    """Server-side "a login is already in progress" guard.

    A local service can receive several "connect" requests, and the browser
    side can be closed, reloaded, or navigated away at any moment. Every
    terminal path must release the lock - success, denial, exchange error,
    timeout, explicit cancel, switching authentication method, and the
    page-hide/unmount case - and a late response from an abandoned attempt must
    never overwrite the credentials of a newer one.

    That is what the monotonically increasing ``generation`` is for: an async
    result carries the generation it was started with, and only the current
    generation is allowed to publish credentials or touch UI state.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._busy = False
        self._hint = ""
        self._cancel: Optional[threading.Event] = None
        self._method: Optional[str] = None
        self._abandoned = 0

    # --- state ------------------------------------------------------------

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def hint(self) -> str:
        with self._lock:
            return self._hint

    @property
    def method(self) -> Optional[str]:
        with self._lock:
            return self._method

    @property
    def abandoned(self) -> int:
        """How many attempts were abandoned before finishing."""
        with self._lock:
            return self._abandoned

    # --- lifecycle --------------------------------------------------------

    def begin(self, method: str) -> tuple[int, threading.Event]:
        """Start an attempt. Returns its generation and cancellation event."""
        with self._lock:
            self._generation += 1
            self._busy = True
            self._method = method
            self._hint = ""
            self._cancel = threading.Event()
            return self._generation, self._cancel

    def is_current(self, generation: int) -> bool:
        """Whether ``generation`` is still the live attempt."""
        with self._lock:
            return self._busy and generation == self._generation

    def publish(self, generation: int, credential) -> bool:
        """Record credentials if the attempt is still current.

        Returns False when the attempt was superseded - the caller must then
        discard its result rather than overwrite newer credentials.
        """
        with self._lock:
            if generation != self._generation:
                return False
            self._busy = False
            self._hint = ""
            self._cancel = None
            return True

    def fail(self, generation: int, message: str) -> bool:
        """Mark the attempt failed and release the lock."""
        with self._lock:
            if generation != self._generation:
                return False
            self._busy = False
            self._hint = message
            self._cancel = None
            return True

    def cancel(self, reason: str = "") -> bool:
        """Explicit user cancel (Cancel button, sign-out, method switch).

        Clears the UI/lock state **synchronously** and signals the waiter; it
        does not rely on the request's ``finally`` block, which would refuse to
        touch state once the generation moved on.
        """
        with self._lock:
            if not self._busy:
                return False
            self._generation += 1
            self._busy = False
            self._hint = reason or "Authorization cancelled."
            self._abandoned += 1
            ev, self._cancel = self._cancel, None
        if ev is not None:
            ev.set()
        return True

    def abandon(self, reason: str = "") -> bool:
        """Page-hide / unmount path.

        Invalidate the generation, clear busy and the hint synchronously, and
        signal cancellation so the server-side work stops. Unlike :meth:`cancel`
        this is safe to call when the UI is going away - it performs no
        UI-bound work of its own.
        """
        return self.cancel(reason or "Authorization window closed.")

    def reset(self) -> None:
        """Full teardown, e.g. process shutdown."""
        with self._lock:
            self._generation += 1
            self._busy = False
            self._hint = ""
            ev, self._cancel = self._cancel, None
        if ev is not None:
            ev.set()

    def run(
        self,
        method: str,
        work: Callable[[int, threading.Event], ConnectResult],
        *,
        on_credential: Optional[Callable[[int, ConnectResult], None]] = None,
        on_error: Optional[Callable[[int, str], None]] = None,
    ) -> Optional[ConnectResult]:
        """Run ``work`` under the lock with generation-safe publication."""
        generation, cancel = self.begin(method)
        try:
            result = work(generation, cancel)
        except OrcaAuthError as exc:
            if self.fail(generation, exc.full_message) and on_error:
                on_error(generation, exc.full_message)
            return None
        except Exception as exc:  # unexpected: still release the lock
            message = f"Authorization failed: {type(exc).__name__}"
            if self.fail(generation, message) and on_error:
                on_error(generation, message)
            return None
        if self.publish(generation, result):
            if on_credential:
                on_credential(generation, result)
            return result
        return None
