"""The two OrcaRouter origins, and how they may be overridden.

OrcaRouter deliberately splits authentication from inference:

    authentication  https://www.orcarouter.ai      (paths: /auth, /api/v1/auth/...)
    inference       https://api.orcarouter.ai/v1

The single most common integration mistake is assuming one origin can be
derived from the other. It cannot: ``https://api.orcarouter.ai/v1/auth/keys``
is a 404. This module is the only place that knows either origin, so no other
module has an opportunity to guess one from the other.

Override policy (explicit always wins):

    ORCA_AUTH_BASE_URL -> auth origin
    ORCA_API_BASE_URL  -> inference base (including its ``/v1`` suffix)
    ORCA_BASE_URL      -> shared self-hosted base, used for both when an
                          explicit per-origin value is absent

Remote origins must be HTTPS; plain HTTP is accepted only for loopback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

DEFAULT_AUTH_BASE = "https://www.orcarouter.ai"
DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"

AUTHORIZE_PATH = "/auth"
EXCHANGE_PATH = "/api/v1/auth/keys"
DEVICE_CODE_PATH = "/api/v1/auth/device/code"
DEVICE_TOKEN_PATH = "/api/v1/auth/device/token"
MODELS_PATH = "/models"
CHAT_COMPLETIONS_PATH = "/chat/completions"

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


class OrcaConfigError(ValueError):
    """An OrcaRouter origin is unusable (bad scheme, not HTTPS, empty)."""


def _is_loopback(host: str) -> bool:
    return (host or "").lower() in LOOPBACK_HOSTS


def validate_base_url(url: str, *, what: str) -> str:
    """Normalise a configured base URL and enforce the transport policy.

    Raises :class:`OrcaConfigError` rather than silently falling back to a
    default - a typo in a self-hosted override should be loud, not routed to
    the public service.
    """
    if not url or not url.strip():
        raise OrcaConfigError(f"{what} is empty")
    candidate = url.strip().rstrip("/")
    parts = urlsplit(candidate)
    if parts.scheme not in ("http", "https"):
        raise OrcaConfigError(
            f"{what} must be an absolute http(s) URL, got scheme "
            f"{parts.scheme or '(none)'!r}"
        )
    if not parts.netloc:
        raise OrcaConfigError(f"{what} has no host")
    if parts.scheme == "http" and not _is_loopback(parts.hostname or ""):
        raise OrcaConfigError(
            f"{what} must use https for a non-loopback host "
            f"({parts.hostname!r} is remote)"
        )
    if parts.query or parts.fragment:
        raise OrcaConfigError(f"{what} must not carry a query or fragment")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _join_v1(base: str) -> str:
    """Append the inference ``/v1`` suffix to a shared base, at most once."""
    return base if base.endswith("/v1") else base + "/v1"


def resolve_endpoints(env: Optional[Mapping[str, str]] = None) -> "OrcaEndpoints":
    """Build the origin pair from the environment (explicit overrides first)."""
    env = os.environ if env is None else env
    auth_override = (env.get("ORCA_AUTH_BASE_URL") or "").strip()
    api_override = (env.get("ORCA_API_BASE_URL") or "").strip()
    shared = (env.get("ORCA_BASE_URL") or "").strip()

    if auth_override:
        auth_base = validate_base_url(auth_override, what="ORCA_AUTH_BASE_URL")
    elif shared:
        auth_base = validate_base_url(shared, what="ORCA_BASE_URL")
    else:
        auth_base = DEFAULT_AUTH_BASE

    if api_override:
        api_base = validate_base_url(api_override, what="ORCA_API_BASE_URL")
    elif shared:
        api_base = _join_v1(validate_base_url(shared, what="ORCA_BASE_URL"))
    else:
        api_base = DEFAULT_API_BASE

    return OrcaEndpoints(auth_base=auth_base, api_base=api_base)


@dataclass(frozen=True)
class OrcaEndpoints:
    """The resolved authentication and inference origins."""

    auth_base: str = DEFAULT_AUTH_BASE
    api_base: str = DEFAULT_API_BASE

    # --- authentication origin -------------------------------------------

    def authorize_url(self) -> str:
        """Consent screen. Always the auth origin, never the inference one."""
        return self.auth_base + AUTHORIZE_PATH

    def exchange_url(self) -> str:
        """Code -> key exchange. ``/api/v1/auth/keys`` on the auth origin."""
        return self.auth_base + EXCHANGE_PATH

    def device_code_url(self) -> str:
        return self.auth_base + DEVICE_CODE_PATH

    def device_token_url(self) -> str:
        return self.auth_base + DEVICE_TOKEN_PATH

    # --- inference origin ------------------------------------------------

    def models_url(self, capability: Optional[str] = None) -> str:
        url = self.api_base + MODELS_PATH
        if capability:
            url = f"{url}?capability={capability}"
        return url

    def chat_completions_url(self) -> str:
        return self.api_base + CHAT_COMPLETIONS_PATH

    def key_console_url(self) -> str:
        """Where the user creates or revokes keys."""
        return self.auth_base + "/console/authorized-apps"

    def assert_distinct_purposes(self) -> None:
        """Guard used by tests and startup checks.

        Refuses the exact misconfiguration this module exists to prevent: an
        inference base carrying the authentication path.
        """
        if self.api_base.endswith("/auth") or "/api/v1/auth" in self.api_base:
            raise OrcaConfigError(
                "the inference base must not contain the auth path; "
                "exchange codes against the auth origin"
            )
