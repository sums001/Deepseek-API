"""Credentials: the two adapters, storage, redaction, and terminal 401 handling.

The property under test is the one the provider depends on: a pasted API key
and a PKCE login produce the *same* kind of credential, so nothing downstream
can tell them apart or copy authentication logic.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from fake_orca_auth import FAKE_KEY, FakeOrcaAuth

from server.orcarouter.connect import OrcaOAuthClient
from server.orcarouter.credentials import (
    ACTIVE,
    AUTH_METHODS,
    NEEDS_REAUTH,
    API_KEY_METHOD,
    PKCE_METHOD,
    ApiKeyAdapter,
    CredentialStore,
    OrcaCredential,
    PkceAdapter,
    looks_like_orca_key,
    redact_key,
    resolve_credential,
)
from server.orcarouter.endpoints import OrcaEndpoints

FAKE_KEY_2 = "sk-orca-second-key-000000000000000"


@pytest.fixture()
def store(tmp_path) -> CredentialStore:
    return CredentialStore(tmp_path / "orcarouter.json")


# --- redaction ------------------------------------------------------------


def test_redact_never_returns_the_key():
    key = "sk-orca-abcdefghijklmnopqrstuvwxyz012345"
    masked = redact_key(key)
    assert key not in masked
    assert masked.startswith("sk-orca-")
    assert masked.endswith(key[-4:])
    assert len(masked) < len(key)


def test_redact_handles_missing_and_short_values():
    assert redact_key(None) == "(no key)"
    assert redact_key("") == "(no key)"
    assert redact_key("short") == "sk-orca-…"


def test_credential_repr_hides_the_key():
    credential = OrcaCredential(api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(credential)
    assert FAKE_KEY not in str(credential)


def test_shape_check_is_a_hint_not_a_validation():
    assert looks_like_orca_key(FAKE_KEY)
    assert not looks_like_orca_key("sk-openai-abc")
    assert not looks_like_orca_key("")
    assert not looks_like_orca_key("sk-orca-")


# --- the API-key adapter (method 1) ---------------------------------------


def test_api_key_adapter_saves_reads_and_clears(store):
    adapter = ApiKeyAdapter(env_var="UNUSED_ENV", store=store)

    result = adapter.acquire(FAKE_KEY)
    assert result.ok
    assert result.credential.method == API_KEY_METHOD
    assert result.credential.usable

    stored = store.load()
    assert stored.api_key == FAKE_KEY
    assert stored.method == API_KEY_METHOD

    assert adapter.clear() is True
    assert store.load() is None


def test_api_key_adapter_rejects_an_obvious_paste_mistake(store):
    result = ApiKeyAdapter(env_var="UNUSED_ENV", store=store).acquire("not-a-key")
    assert not result.ok
    assert "does not look like" in result.message
    assert store.load() is None


def test_api_key_adapter_reads_the_environment(store, monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    adapter = ApiKeyAdapter(store=store)
    result = adapter.acquire()
    assert result.ok
    assert result.credential.source == "ORCAROUTER_API_KEY"


def test_api_key_adapter_reports_an_empty_environment(store, monkeypatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    result = ApiKeyAdapter(store=store).acquire()
    assert not result.ok
    assert "no API key" in result.message


def test_stored_file_is_owner_only_and_contains_the_key(store, tmp_path):
    ApiKeyAdapter(env_var="UNUSED_ENV", store=store).acquire(FAKE_KEY)
    path = store.path
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["orcarouter"]["api_key"] == FAKE_KEY
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


# --- the PKCE adapter (method 2) ------------------------------------------


def _pkce_adapter(auth: FakeOrcaAuth, store: CredentialStore) -> PkceAdapter:
    client = OrcaOAuthClient(
        OrcaEndpoints(auth_base=auth.auth_base, api_base=auth.api_base),
        timeout=10.0,
        open_browser=lambda url: False,
    )
    return PkceAdapter(client, store)


def _consent_code(url: str) -> str:
    """Visit the consent URL and read the code it displays (out-of-band mode).

    Using the *same* authorize URL the adapter built keeps the code bound to
    the adapter's real challenge - a separately minted code would (correctly)
    be rejected as a verifier mismatch.
    """
    import httpx

    body = httpx.get(url, timeout=10).text
    return body.split("code: ")[1].split("<")[0]


def test_pkce_adapter_persists_the_same_credential_type(store):
    with FakeOrcaAuth() as auth:
        adapter = _pkce_adapter(auth, store)
        holder = {}
        result = adapter.acquire(
            flow="oob",
            on_url=lambda url: holder.__setitem__("url", url),
            prompt_code=lambda: _consent_code(holder["url"]),
        )

    assert result.ok
    assert result.credential.method == PKCE_METHOD
    stored = store.load()
    assert stored.api_key == FAKE_KEY
    assert stored.scope == "api"
    assert stored.user_id == "12345"


def test_pkce_adapter_reports_denial_without_deleting_the_old_credential(store):
    import threading

    import httpx

    ApiKeyAdapter(env_var="UNUSED_ENV", store=store).acquire(FAKE_KEY)

    with FakeOrcaAuth(behavior="deny") as auth:
        adapter = _pkce_adapter(auth, store)

        def on_url(url):
            threading.Thread(
                target=lambda: httpx.get(url, timeout=10, follow_redirects=True),
                daemon=True,
            ).start()

        result = adapter.acquire(flow="loopback", on_url=on_url)

    assert not result.ok
    # The previous credential survived the failed login.
    assert store.load().api_key == FAKE_KEY


def test_both_adapters_produce_the_same_credential_shape(store):
    key_result = ApiKeyAdapter(env_var="UNUSED_ENV", store=store).acquire(FAKE_KEY)
    with FakeOrcaAuth() as auth:
        holder = {}
        pkce_result = _pkce_adapter(auth, store).acquire(
            flow="oob",
            on_url=lambda url: holder.__setitem__("url", url),
            prompt_code=lambda: _consent_code(holder["url"]),
        )

    assert key_result.ok and pkce_result.ok
    for result in (key_result, pkce_result):
        credential = result.credential
        assert isinstance(credential, OrcaCredential)
        assert credential.method in AUTH_METHODS
        assert credential.usable
        assert credential.scope == "api"
        # Same public surface, regardless of how the key was obtained.
        assert credential.api_key.startswith("sk-orca-")
        assert credential.masked == redact_key(credential.api_key)


def test_generations_increase_across_credential_replacements(store):
    first = ApiKeyAdapter(env_var="UNUSED_ENV", store=store).acquire(FAKE_KEY)
    second = ApiKeyAdapter(env_var="UNUSED_ENV", store=store).acquire(FAKE_KEY_2)
    assert first.credential.generation == 1
    assert second.credential.generation == 2


# --- resolution and precedence --------------------------------------------


def test_environment_key_takes_precedence_over_a_stored_pkce_login(store, monkeypatch):
    store.save(OrcaCredential(api_key=FAKE_KEY, method=PKCE_METHOD, user_id="9"))
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY_2)
    resolved = resolve_credential(store)
    assert resolved.api_key == FAKE_KEY_2
    assert resolved.method == API_KEY_METHOD


def test_stored_credential_used_when_the_environment_is_empty(store, monkeypatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    store.save(OrcaCredential(api_key=FAKE_KEY, method=PKCE_METHOD, user_id="9"))
    assert resolve_credential(store).api_key == FAKE_KEY


def test_a_corrupt_store_reads_as_no_credential_plus_a_flag(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{ this is not json")
    assert store.load() is None
    assert store.corrupt is True


def test_a_record_missing_its_key_is_rejected(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps({"orcarouter": {"method": "pkce"}}))
    assert store.load() is None


# --- terminal 401 handling ------------------------------------------------


def test_mark_needs_reauth_targets_the_exact_account(store):
    credential = store.save(OrcaCredential(api_key=FAKE_KEY, user_id="42"))
    updated = store.mark_needs_reauth(
        account_id=credential.account_id, generation=credential.generation
    )
    assert updated is not None
    assert updated.status == NEEDS_REAUTH
    assert store.load().status == NEEDS_REAUTH


def test_mark_needs_reauth_ignores_a_stale_generation(store):
    store.save(OrcaCredential(api_key=FAKE_KEY, user_id="42"))
    # A new login replaced the credential before the old request failed.
    fresh = store.save(
        OrcaCredential(api_key=FAKE_KEY_2, user_id="42", generation=2)
    )
    stale = store.mark_needs_reauth(account_id=fresh.account_id, generation=1)
    assert stale is None
    assert store.load().status == ACTIVE
    assert store.load().api_key == FAKE_KEY_2


def test_mark_needs_reauth_ignores_a_different_account(store):
    store.save(OrcaCredential(api_key=FAKE_KEY, user_id="42", account_id="acct-a"))
    assert store.mark_needs_reauth(account_id="acct-b", generation=1) is None
    assert store.load().status == ACTIVE


def test_needs_reauth_credential_is_not_usable_but_is_reported():
    credential = OrcaCredential(api_key=FAKE_KEY, status=NEEDS_REAUTH)
    assert not credential.usable
    assert credential.needs_reauth


def test_no_refresh_machinery_exists():
    """A revoked durable key must not be silently refreshed."""
    import server.orcarouter.credentials as credentials

    source = open(credentials.__file__).read()
    assert "refresh_token" not in source
    assert "grant_type" not in source
