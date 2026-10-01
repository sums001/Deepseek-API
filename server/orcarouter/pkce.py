"""PKCE primitives: verifier, S256 challenge, and state.

Everything here is standard-library only - ``hashlib`` and ``secrets`` already
do SHA-256 and a cryptographically secure RNG, so no dependency is added.

Two rules drive the implementation:

* the **verifier never leaves the process** until the code exchange; it must
  not appear in a URL, a log line, an exception message, or a UI payload.
* the verifier is **fresh per attempt** and comes from a cryptographic RNG.
  Deriving it from a timestamp, a username, or a fixed salt would make the
  whole flow decorative.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass

# RFC 7636 section 4.1 requires 43..128 characters of the unreserved set.
VERIFIER_BYTES = 32
STATE_BYTES = 16
MIN_VERIFIER_LENGTH = 43


def b64url(raw: bytes) -> str:
    """base64url without padding, as RFC 7636 specifies for the challenge."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def new_verifier(nbytes: int = VERIFIER_BYTES) -> str:
    """A fresh high-entropy code verifier."""
    return b64url(secrets.token_bytes(nbytes))


def new_state(nbytes: int = STATE_BYTES) -> str:
    """A fresh opaque CSRF state value, echoed back by the consent screen."""
    return b64url(secrets.token_bytes(nbytes))


def challenge_for(verifier: str) -> str:
    """``base64url(sha256(verifier))`` - the S256 challenge."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return b64url(digest)


def states_match(expected: str, received: str | None) -> bool:
    """Constant-time state comparison.

    ``hmac.compare_digest`` keeps the comparison from leaking where the first
    differing byte is, and it tolerates a missing parameter (``None``) rather
    than raising mid-callback.
    """
    import hmac

    if received is None:
        return False
    return hmac.compare_digest(str(expected), str(received))


@dataclass(frozen=True)
class PkceAttempt:
    """One authorization attempt's secret material.

    ``verifier`` is deliberately excluded from ``__repr__`` so an accidental
    ``print(attempt)`` - or a debugger traceback that renders locals - cannot
    leak it.
    """

    verifier: str
    challenge: str
    state: str
    flow: str

    @classmethod
    def create(cls, flow: str = "loopback") -> "PkceAttempt":
        verifier = new_verifier()
        return cls(
            verifier=verifier,
            challenge=challenge_for(verifier),
            state=new_state(),
            flow=flow,
        )

    def __repr__(self) -> str:  # pragma: no cover - shape asserted in tests
        return (
            f"PkceAttempt(challenge={self.challenge!r}, state={self.state!r}, "
            f"flow={self.flow!r}, verifier=<redacted>)"
        )

    def exchange_body(self, code: str) -> dict:
        """The JSON body for ``POST /api/v1/auth/keys``."""
        return {
            "code": code,
            "code_verifier": self.verifier,
            "code_challenge_method": "S256",
        }


def build_authorize_url(
    authorize_url: str,
    *,
    callback_url: str,
    attempt: PkceAttempt,
    app_name: str,
    scope: str = "api",
    login_hint: str | None = None,
    workspace_hint: str | None = None,
) -> str:
    """Compose the consent URL. Carries the challenge and state, never the verifier."""
    from urllib.parse import urlencode

    params = {
        "callback_url": callback_url,
        "code_challenge": attempt.challenge,
        "code_challenge_method": "S256",
        "state": attempt.state,
        "app_name": app_name,
        "scope": scope,
    }
    if login_hint:
        params["login_hint"] = login_hint
    if workspace_hint:
        params["workspace_hint"] = workspace_hint
    return f"{authorize_url}?{urlencode(params)}"
