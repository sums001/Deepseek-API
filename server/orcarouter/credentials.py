"""Credentials: one interface, two adapters, one durable key.

Both user-facing ways in end in exactly the same thing - a normal OrcaRouter
API key - so the rest of the program never learns which one was used:

    API key  (paste an existing ``sk-orca-...``)  -> \\
                                                    >-- OrcaCredential
    PKCE     (browser/OOB login, no secret)       -> /

Everything downstream - the provider, the model catalog, the server - consumes
``OrcaCredential`` and nothing else. There is deliberately no place where a
request can tell whether the key came from a paste or from a login.

Storage follows the project's existing secret mechanism: a git-ignored JSON
file under ``session/``, the same directory the DeepSeek session already uses,
with owner-only permissions. No new secret store is introduced, and the key is
never written to a log, an error, or a URL.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

API_KEY_METHOD = "api_key"
PKCE_METHOD = "pkce"
AUTH_METHODS = (API_KEY_METHOD, PKCE_METHOD)

# Where a credential is in its life. ``needs_reauth`` is what a relay 401 sets:
# the key is dead and must be replaced by a new login, not retried and not
# silently refreshed (there is no refresh grant).
ACTIVE = "active"
NEEDS_REAUTH = "needs_reauth"

KEY_PREFIX = "sk-orca-"
_ENV_VAR = "ORCAROUTER_API_KEY"

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CREDENTIAL_FILE = ROOT / "session" / "orcarouter.json"


def redact_key(key: Optional[str]) -> str:
    """A displayable form: enough to tell two keys apart, useless as a secret.

    Never returns the key itself, and never returns a suffix long enough to be
    a meaningful hint about a 51-character credential.
    """
    if not key:
        return "(no key)"
    if len(key) <= 12:
        return KEY_PREFIX + "…"
    return f"{key[:11]}…{key[-4:]}"


def looks_like_orca_key(value: str) -> bool:
    """Lightweight shape check only.

    An ``sk-orca-`` prefix is *not* proof that a credential is valid - there is
    no non-billing validation request to make, so validity is established by
    the first real call. This only catches an obvious paste mistake.
    """
    return bool(value) and value.startswith(KEY_PREFIX) and len(value) > len(KEY_PREFIX) + 8


@dataclass
class OrcaCredential:
    """The one credential type produced by both adapters.

    ``generation`` is what makes terminal 401 handling safe for multi-account
    and multi-attempt use: a rejected request records the generation it ran
    with, so a late failure from an old request can never mark a freshly
    reauthorized credential as broken.
    """

    api_key: str
    method: str = API_KEY_METHOD
    user_id: str = ""
    scope: str = "api"
    status: str = ACTIVE
    created_at: float = field(default_factory=time.time)
    generation: int = 1
    account_id: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.method not in AUTH_METHODS:
            raise ValueError(f"unknown auth method: {self.method!r}")
        if not self.account_id:
            self.account_id = self.user_id or f"local-{uuid.uuid4().hex[:8]}"

    # --- display ----------------------------------------------------------

    @property
    def masked(self) -> str:
        return redact_key(self.api_key)

    @property
    def usable(self) -> bool:
        return self.status == ACTIVE and bool(self.api_key)

    @property
    def needs_reauth(self) -> bool:
        return self.status == NEEDS_REAUTH

    def __repr__(self) -> str:  # pragma: no cover - asserted in tests
        return (
            f"OrcaCredential(masked={self.masked!r}, method={self.method!r}, "
            f"user_id={self.user_id!r}, scope={self.scope!r}, "
            f"status={self.status!r}, generation={self.generation})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    # --- serialisation ----------------------------------------------------

    def to_dict(self, *, include_secret: bool = True) -> Dict:
        data = {
            "method": self.method,
            "user_id": self.user_id,
            "scope": self.scope,
            "status": self.status,
            "created_at": self.created_at,
            "generation": self.generation,
            "account_id": self.account_id,
            "source": self.source,
            "masked": self.masked,
        }
        if include_secret:
            data["api_key"] = self.api_key
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "OrcaCredential":
        if not isinstance(data, dict):
            raise ValueError("credential record must be an object")
        record = data.get("orcarouter") if "orcarouter" in data else data
        key = record.get("api_key")
        if not key or not isinstance(key, str):
            raise ValueError("credential record has no api_key")
        return cls(
            api_key=key,
            method=record.get("method") or API_KEY_METHOD,
            user_id=str(record.get("user_id") or ""),
            scope=str(record.get("scope") or "api"),
            status=record.get("status") or ACTIVE,
            created_at=float(record.get("created_at") or time.time()),
            generation=int(record.get("generation") or 1),
            account_id=str(record.get("account_id") or ""),
            source=str(record.get("source") or ""),
        )


class CredentialStore:
    """Reads and writes the OrcaRouter credential in the project's secret file."""

    def __init__(self, path: Path = DEFAULT_CREDENTIAL_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: Optional[OrcaCredential] = None

    # --- read -------------------------------------------------------------

    def load(self) -> Optional[OrcaCredential]:
        """Return the stored credential, or None when there is not one.

        A corrupt file is reported as *no credential* plus a hint rather than
        raised at import time - an unreadable side file must not stop the
        server from answering API-key requests.
        """
        with self._lock:
            if self._cache is not None:
                return self._cache
            if not self.path.exists():
                return None
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._cache = OrcaCredential.from_dict(raw)
            except Exception:
                self._cache = None
                self._corrupt = True
            return self._cache

    @property
    def corrupt(self) -> bool:
        """Whether the last load found an unreadable record."""
        return bool(getattr(self, "_corrupt", False))

    def peek(self) -> Optional[OrcaCredential]:
        """Read without touching the cache - used to re-validate on startup."""
        if not self.path.exists():
            return None
        try:
            return OrcaCredential.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return None

    # --- write ------------------------------------------------------------

    def save(self, credential: OrcaCredential) -> OrcaCredential:
        """Persist a credential with owner-only permissions.

        Only ever replaces the stored record after the new one exists, so a
        transient failure cannot destroy a working credential.
        """
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"orcarouter": credential.to_dict()}, indent=2),
                encoding="utf-8",
            )
            try:
                os.chmod(tmp, 0o600)
            except OSError:  # pragma: no cover - platform dependent
                pass
            os.replace(tmp, self.path)
            self._cache = credential
            self._corrupt = False
            return credential

    def clear(self) -> bool:
        """Forget the credential. Returns whether anything was removed."""
        with self._lock:
            self._cache = None
            self._corrupt = False
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    return False
                return True
            return False

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    # --- lifecycle --------------------------------------------------------

    @staticmethod
    def next_generation(previous: Optional[OrcaCredential]) -> int:
        return (previous.generation + 1) if previous else 1

    def mark_needs_reauth(
        self,
        *,
        account_id: str,
        generation: int,
    ) -> Optional[OrcaCredential]:
        """Terminal 401 handling, scoped to one account *and* one generation.

        Returns the updated credential when the rejection applied, or None when
        it was stale - the request was made with an older credential generation
        than the one now stored, so marking it would corrupt a fresh login.
        """
        with self._lock:
            current = self._cache
            if current is None:
                current = self.peek()
                if current is None:
                    return None
            if current.account_id != account_id:
                return None
            if generation != current.generation:
                return None
            if current.status == NEEDS_REAUTH:
                return current
            current.status = NEEDS_REAUTH
            self._persist_locked(current)
            self._cache = current
            return current

    def _persist_locked(self, credential: OrcaCredential) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"orcarouter": credential.to_dict()}, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(tmp, 0o600)
        except OSError:  # pragma: no cover
            pass
        os.replace(tmp, self.path)


# --- the seam: two adapters over one result type --------------------------


@dataclass
class CredentialResult:
    """What both adapters return. The caller cannot tell them apart."""

    credential: Optional[OrcaCredential]
    ok: bool
    message: str = ""


class CredentialAdapter:
    """A way to obtain an OrcaRouter credential."""

    method = ""

    def acquire(self, **kwargs) -> CredentialResult:  # pragma: no cover - interface
        raise NotImplementedError


class ApiKeyAdapter(CredentialAdapter):
    """Adapter 1: an existing ``sk-orca-...`` key from the environment or a paste.

    Takes precedence over a stored PKCE login when ``ORCAROUTER_API_KEY`` is
    set, because an operator who exports a key has said which key they want
    used. Clearing the variable falls back to the PKCE credential.
    """

    method = API_KEY_METHOD

    def __init__(self, env_var: str = _ENV_VAR, store: Optional[CredentialStore] = None):
        self.env_var = env_var
        self.store = store

    def acquire(
        self,
        value: Optional[str] = None,
        *,
        persist: bool = True,
    ) -> CredentialResult:
        raw = (value if value is not None else os.environ.get(self.env_var) or "").strip()
        if not raw:
            return CredentialResult(None, False, "no API key provided")
        if not looks_like_orca_key(raw):
            return CredentialResult(
                None,
                False,
                f"that does not look like an OrcaRouter key (expected a "
                f"{KEY_PREFIX}... value)",
            )
        credential = OrcaCredential(api_key=raw, method=API_KEY_METHOD, source=self.env_var)
        if persist and self.store is not None:
            previous = self.store.load()
            credential.generation = self.store.next_generation(previous)
            self.store.save(credential)
        return CredentialResult(credential, True, f"using the key from {self.env_var}")

    def clear(self) -> bool:
        """Remove the stored credential (an environment variable is not ours to unset)."""
        if self.store is None:
            return False
        return self.store.clear()


class PkceAdapter(CredentialAdapter):
    """Adapter 2: sign in with an OrcaRouter account.

    Produces an ordinary API key, which means it produces an ordinary
    ``OrcaCredential``. Calling it "an auth credential" anywhere downstream
    would be a bug.
    """

    method = PKCE_METHOD

    def __init__(self, client, store: CredentialStore):
        self.client = client
        self.store = store

    def acquire(self, flow: str = "loopback", **kwargs) -> CredentialResult:
        """Run the connect flow and persist the resulting key.

        The old credential is *not* deleted first: if the login fails the user
        still has whatever they had.
        """
        try:
            if flow == "oob":
                result = self.client.connect_oob(**kwargs)
            elif flow == "device":
                result = self.client.connect_device(**kwargs)
            else:
                result = self.client.connect_loopback(**kwargs)
        except Exception as exc:  # OrcaAuthError and transport failures alike
            message = getattr(exc, "full_message", None) or f"{type(exc).__name__}: {exc}"
            return CredentialResult(None, False, message)

        previous = self.store.load()
        credential = OrcaCredential(
            api_key=result.key,
            method=PKCE_METHOD,
            user_id=result.user_id,
            scope=result.scope,
            source=f"pkce:{result.flow}",
            generation=self.store.next_generation(previous),
        )
        self.store.save(credential)
        return CredentialResult(credential, True, f"authorized as {credential.user_id or 'account'}")


def resolve_credential(
    store: Optional[CredentialStore] = None,
    env: Optional[Dict[str, str]] = None,
) -> Optional[OrcaCredential]:
    """Pick the credential to use, without caring where it came from.

    Order: an explicit ``ORCAROUTER_API_KEY`` (transient - it is not copied to
    disk), then a stored PKCE/API credential. A credential marked
    ``needs_reauth`` is returned as-is so callers can report the required
    reauthentication rather than pretending nothing is configured.
    """
    store = store or CredentialStore()
    env = os.environ if env is None else env
    from_env = (env.get(_ENV_VAR) or "").strip()
    if from_env and looks_like_orca_key(from_env):
        return OrcaCredential(
            api_key=from_env, method=API_KEY_METHOD, source=_ENV_VAR,
        )
    return store.load()


def credential_for_request(store: Optional[CredentialStore] = None):
    """Snapshot the credential that a request should run with.

    The returned object carries ``generation`` and ``account_id``; those two
    values are what the caller hands back to
    :meth:`CredentialStore.mark_needs_reauth` if the relay answers 401.
    """
    return resolve_credential(store)


ENV_VAR = _ENV_VAR
