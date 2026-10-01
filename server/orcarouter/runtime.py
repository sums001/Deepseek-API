"""Runtime glue: one credential store, one catalog cache, one login lock.

The server needs exactly one of each, so they live here rather than being
rebuilt per request. Nothing in this module stores or returns a raw key: the
HTTP status payload carries a redacted form, and the model list carries
metadata only.

The connect endpoint exists because this project is also run as a local
service. A local login can be abandoned in many ways - the user denies, the
tab is closed, the page is hidden, the method is switched - so the login lock
is released on every terminal path and a late response from an abandoned
attempt cannot overwrite a newer credential.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .catalog import CAPABILITIES, OrcaCatalogClient
from .connect import OrcaOAuthClient, OrcaLoginManager
from .credentials import (
    API_KEY_METHOD,
    CredentialStore,
    OrcaCredential,
    redact_key,
    resolve_credential,
)
from .endpoints import OrcaEndpoints, resolve_endpoints
from .provider import OrcaRouterProvider

_lock = threading.Lock()
_store: Optional[CredentialStore] = None
_catalog: Optional[OrcaCatalogClient] = None
_endpoints: Optional[OrcaEndpoints] = None
_login = OrcaLoginManager()
_pending_url: Dict[int, str] = {}
_pending_code: Dict[int, str] = {}


# The singletons are built outside the lock. The lock only guards assignment,
# because none of the constructors is re-entrant on it: `get_catalog()` needs
# the endpoints, and calling back into a locked `get_endpoints()` would
# deadlock. Building a second copy in a race is harmless and cheap.


def get_endpoints() -> OrcaEndpoints:
    global _endpoints
    if _endpoints is None:
        with _lock:
            if _endpoints is None:
                _endpoints = resolve_endpoints()
    return _endpoints


def get_store() -> CredentialStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = CredentialStore()
    return _store


def get_catalog() -> OrcaCatalogClient:
    global _catalog
    if _catalog is None:
        endpoints = get_endpoints()
        with _lock:
            if _catalog is None:
                _catalog = OrcaCatalogClient(endpoints)
    return _catalog


def get_login_manager() -> OrcaLoginManager:
    return _login


def current_credential() -> Optional[OrcaCredential]:
    return resolve_credential(get_store())


def get_provider(credential: Optional[OrcaCredential] = None) -> OrcaRouterProvider:
    """Build a provider bound to a credential snapshot.

    A fresh provider per request is intentional: the snapshot carries the
    ``generation`` that a later 401 must report, so a late failure from an old
    request cannot mark a newly reauthorized credential as revoked.
    """
    return OrcaRouterProvider(
        credential if credential is not None else current_credential(),
        endpoints=get_endpoints(),
        store=get_store(),
        catalog=get_catalog(),
    )


def reset_runtime() -> None:
    """Test/teardown hook: forget cached singletons and release the login lock."""
    global _store, _catalog, _endpoints
    _login.reset()
    with _lock:
        _store = None
        _catalog = None
        _endpoints = None
        _pending_url.clear()
        _pending_code.clear()


def _status_payload() -> dict:
    credential = current_credential()
    return {
        "auth_base": get_endpoints().auth_base,
        "api_base": get_endpoints().api_base,
        "key_console_url": get_endpoints().key_console_url(),
        "configured": credential is not None,
        "api_key_available": bool(
            _store_has_key() or (credential is not None and credential.method == API_KEY_METHOD)
        ),
        "pkce_available": _store_has_pkce(),
        "method": credential.method if credential else None,
        "source": credential.source if credential else None,
        "masked": redact_key(credential.api_key) if credential else None,
        "scope": credential.scope if credential else None,
        "user_id": credential.user_id if credential else None,
        "status": credential.status if credential else None,
        "needs_reauth": bool(credential and credential.needs_reauth),
        "generation": credential.generation if credential else None,
        "busy": _login.busy,
        "login_generation": _login.generation,
        "hint": _login.hint,
        "login_method": _login.method,
        "authorize_url": _pending_url.get(_login.generation, ""),
    }


def _store_has_key() -> bool:
    record = get_store().load()
    return bool(record and record.method == API_KEY_METHOD)


def _store_has_pkce() -> bool:
    record = get_store().load()
    return bool(record and record.method == "pkce")


def _error(message: str, status: int = 400, err_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": err_type}}
    )


def _run_pkce_login(flow: str):
    """Background worker for a PKCE login started over HTTP."""

    def work(generation: int, cancel: threading.Event):
        client = OrcaOAuthClient(endpoints=get_endpoints())
        on_url = lambda url: _pending_url.__setitem__(generation, url)  # noqa: E731

        if flow == "oob":
            return client.connect_oob(
                on_url=on_url,
                prompt_code=lambda: _pending_code.get(generation, ""),
                cancel=cancel,
            )
        if flow == "device":
            return client.connect_device(on_prompt=lambda _p: None, cancel=cancel)
        return client.connect_loopback(on_url=on_url, cancel=cancel)

    def on_credential(generation: int, result) -> None:
        previous = get_store().load()
        previous_generation = previous.generation if previous else 0
        credential = OrcaCredential(
            api_key=result.key,
            method="pkce",
            user_id=result.user_id,
            scope=result.scope,
            source=f"pkce:{result.flow}",
            generation=previous_generation + 1,
        )
        get_store().save(credential)
        _pending_url.pop(generation, None)
        _pending_code.pop(generation, None)

    thread = threading.Thread(
        target=_login.run,
        args=("pkce", work),
        kwargs={"on_credential": on_credential},
        name="orca-pkce-login",
        daemon=True,
    )
    thread.start()


router = APIRouter(prefix="/v1/orcarouter", tags=["orcarouter"])


@router.get("/status")
def orcarouter_status():
    """Both authentication methods, the active credential (redacted), and the
    live login-lock state."""
    return _status_payload()


@router.get("/models")
def orcarouter_models(
    capability: str = "chat",
    modality: Optional[str] = None,
    refresh: bool = False,
):
    """The catalog for one capability, already filtered.

    The credential stays in this process; the response carries model metadata
    only, so a browser never holds an OrcaRouter key.
    """
    if capability not in CAPABILITIES:
        return _error(
            f"`{capability}` is not a capability. Known: {', '.join(sorted(CAPABILITIES))}."
        )
    modalities = [m for m in (modality or "").split(",") if m]
    result = get_catalog().fetch(
        capability,
        credential=current_credential(),
        required_modalities=modalities or None,
        allow_cache=not refresh,
    )
    return result.public()


@router.post("/connect/start")
def orcarouter_connect_start(body: Optional[dict] = None):
    """Begin a login.

    ``method=api_key`` saves a pasted key synchronously. ``method=pkce`` starts
    the OAuth 2.0 + PKCE flow in the background and returns the consent URL;
    poll ``/status`` for the result. Only the redacted credential is ever
    returned.
    """
    body = body or {}
    method = (body.get("method") or "pkce").strip()

    if method == API_KEY_METHOD:
        from .credentials import looks_like_orca_key

        value = (body.get("api_key") or "").strip()
        if not looks_like_orca_key(value):
            return _error(
                "That does not look like an OrcaRouter key (expected an "
                "`sk-orca-...` value). Create one at "
                f"{get_endpoints().key_console_url()}."
            )
        previous = get_store().load()
        credential = OrcaCredential(
            api_key=value,
            method=API_KEY_METHOD,
            source="api",
            generation=(previous.generation + 1) if previous else 1,
        )
        get_store().save(credential)
        return {
            "ok": True,
            "method": API_KEY_METHOD,
            "masked": credential.masked,
            "status": credential.status,
        }

    if method != "pkce":
        return _error("`method` must be `api_key` or `pkce`.")

    flow = (body.get("flow") or "loopback").strip()
    if flow not in ("loopback", "oob", "device"):
        return _error("`flow` must be `loopback`, `oob`, or `device`.")

    if _login.busy:
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "message": "A login is already in progress.",
                    "type": "login_in_progress",
                },
                "generation": _login.generation,
                "authorize_url": _pending_url.get(_login.generation, ""),
            },
        )

    _run_pkce_login(flow)
    return {
        "ok": True,
        "method": "pkce",
        "flow": flow,
        "generation": _login.generation,
        "status": "pending",
    }


@router.post("/connect/code")
def orcarouter_connect_code(body: Optional[dict] = None):
    """Deliver the out-of-band code to a running OOB login."""
    body = body or {}
    code = (body.get("code") or "").strip()
    if not code:
        return _error("`code` is required.")
    if not _login.busy:
        return _error("No login is in progress.", status=409, err_type="no_login")
    _pending_code[_login.generation] = code
    return {"ok": True, "generation": _login.generation}


@router.post("/connect/cancel")
def orcarouter_connect_cancel(body: Optional[dict] = None):
    """Explicit cancel: releases the lock synchronously."""
    cancelled = _login.cancel("Authorization cancelled.")
    return {"ok": True, "cancelled": cancelled, "generation": _login.generation}


@router.post("/connect/pagehide")
def orcarouter_connect_pagehide(body: Optional[dict] = None):
    """The page is going away - possibly into the back-forward cache.

    Clears the busy flag and hint synchronously and signals the server-side
    work to stop. The response tells the caller to start again without
    remounting.
    """
    cancelled = _login.abandon("Authorization window closed.")
    return {
        "ok": True,
        "abandoned": cancelled,
        "busy": _login.busy,
        "hint": _login.hint,
        "generation": _login.generation,
        "can_start_again": not _login.busy,
    }


@router.post("/logout")
def orcarouter_logout():
    """Forget the stored credential. An exported ORCAROUTER_API_KEY still wins."""
    return {"ok": True, "cleared": get_store().clear()}


def install_orcarouter(app) -> None:
    """Attach the OrcaRouter control surface to the app."""
    app.include_router(router)
