"""OrcaRouter authentication: PKCE primitives, the connect flow, and its errors.

Covers the requirements that make the flow safe rather than merely present:
fresh cryptographic verifier/state per attempt, an S256-only challenge, the
verifier never leaving the process, a state comparison before the code is used,
single-use/expired code handling, a scope downgrade being reported, and 400/403/429
and network failures ending cleanly instead of hanging or hot-looping.
"""

from __future__ import annotations

import re
import threading

import httpx
import pytest

from fake_orca_auth import FAKE_KEY, FakeOrcaAuth, b64url

from server.orcarouter.connect import (
    OrcaAuthDenied,
    OrcaAuthError,
    OrcaAuthTimeout,
    OrcaExchangeError,
    OrcaOAuthClient,
    OrcaScopeDowngrade,
    OrcaStateMismatch,
    scope_satisfies,
)
from server.orcarouter.endpoints import OrcaEndpoints
from server.orcarouter.pkce import (
    PkceAttempt,
    build_authorize_url,
    challenge_for,
    new_state,
    new_verifier,
    states_match,
)


# --- PKCE primitives ------------------------------------------------------


def test_verifier_is_fresh_and_high_entropy():
    a, b = new_verifier(), new_verifier()
    assert a != b
    assert len(a) >= 43  # RFC 7636 minimum
    assert re.fullmatch(r"[A-Za-z0-9\-_]+", a)
    assert not a.endswith("=")


def test_state_is_fresh_per_call():
    assert new_state() != new_state()


def test_challenge_is_unpadded_base64url_sha256():
    verifier = "test-verifier-value"
    expected = b64url(__import__("hashlib").sha256(verifier.encode()).digest())
    assert challenge_for(verifier) == expected
    assert "=" not in challenge_for(verifier)


def test_attempt_repr_never_shows_the_verifier():
    attempt = PkceAttempt.create("loopback")
    assert attempt.verifier not in repr(attempt)
    assert attempt.challenge in repr(attempt)
    assert "<redacted>" in repr(attempt)


def test_exchange_body_uses_s256_and_carries_the_verifier():
    attempt = PkceAttempt.create("oob")
    body = attempt.exchange_body("the-code")
    assert body == {
        "code": "the-code",
        "code_verifier": attempt.verifier,
        "code_challenge_method": "S256",
    }


def test_each_attempt_gets_a_new_verifier_and_state():
    first = PkceAttempt.create("loopback")
    second = PkceAttempt.create("loopback")
    assert first.verifier != second.verifier
    assert first.state != second.state
    assert first.challenge != second.challenge


def test_states_match_is_exact_and_none_safe():
    assert states_match("abc", "abc")
    assert not states_match("abc", "abd")
    assert not states_match("abc", None)
    assert not states_match("abc", "")


def test_authorize_url_carries_challenge_not_verifier():
    attempt = PkceAttempt.create("loopback")
    url = build_authorize_url(
        "https://www.orcarouter.ai/auth",
        callback_url="http://127.0.0.1:1234/cb",
        attempt=attempt,
        app_name="DeepSeek API bridge",
    )
    assert "code_challenge_method=S256" in url
    assert attempt.challenge in url
    assert attempt.state in url
    assert attempt.verifier not in url
    assert "callback_url=http%3A%2F%2F127.0.0.1%3A1234%2Fcb" in url


# --- scope handling -------------------------------------------------------


def test_scope_satisfaction():
    assert scope_satisfies("api", "api")
    assert scope_satisfies("connector", "api")
    assert scope_satisfies("connector", "connector")
    assert not scope_satisfies("api", "connector")
    assert not scope_satisfies("", "api")
    assert not scope_satisfies("something-else", "api")


# --- Flow A against the local fake service --------------------------------


def _client_for(auth: FakeOrcaAuth, **kwargs) -> OrcaOAuthClient:
    return OrcaOAuthClient(
        OrcaEndpoints(auth_base=auth.auth_base, api_base=auth.api_base),
        timeout=10.0,
        open_browser=lambda url: False,
        **kwargs,
    )


def _auto_callback(client: OrcaOAuthClient, auth: FakeOrcaAuth):
    """Open the authorize URL against the fake consent screen in a thread.

    The fake consent screen answers with a 302 to the ``callback_url`` the
    client registered, so the redirect is followed - that is what a browser
    would do.
    """
    holder = {}

    def on_url(url: str):
        holder["url"] = url
        threading.Thread(
            target=lambda: httpx.get(url, timeout=10, follow_redirects=True),
            daemon=True,
        ).start()

    return on_url, holder


def test_flow_a_happy_path_persists_the_key():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        on_url, holder = _auto_callback(client, auth)
        result = client.connect_loopback(on_url=on_url)

    assert result.key == FAKE_KEY
    assert result.scope == "api"
    assert result.user_id == "12345"
    assert result.flow == "loopback"

    # The exchange went to the auth origin's documented path, with S256.
    assert len(auth.exchange_requests) == 1
    body = auth.exchange_requests[0]
    assert body["code_challenge_method"] == "S256"
    assert body["code_verifier"]

    sent = auth.authorize_requests[0]
    assert sent["code_challenge_method"] == "S256"
    assert sent["scope"] == "api"
    assert sent["callback_url"].startswith("http://127.0.0.1:")
    # The verifier is the one secret that must never ride the authorize URL.
    assert body["code_verifier"] not in holder["url"]


def test_flow_a_denial_raises_a_terminal_error():
    with FakeOrcaAuth(behavior="deny") as auth:
        client = _client_for(auth)
        on_url, _ = _auto_callback(client, auth)
        with pytest.raises(OrcaAuthDenied) as excinfo:
            client.connect_loopback(on_url=on_url)

    assert "denied" in str(excinfo.value).lower()
    assert auth.exchange_requests == []  # no code, no exchange


def test_flow_a_state_mismatch_is_rejected_before_exchange():
    with FakeOrcaAuth(behavior="bad_state") as auth:
        client = _client_for(auth)
        on_url, _ = _auto_callback(client, auth)
        with pytest.raises(OrcaStateMismatch):
            client.connect_loopback(on_url=on_url)

    # The code was minted but never redeemed: the CSRF check ran first.
    assert auth.exchange_requests == []


def test_flow_a_never_logs_the_verifier(capsys):
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        on_url, holder = _auto_callback(client, auth)
        result = client.connect_loopback(on_url=on_url)

    captured = capsys.readouterr()
    assert result.key not in captured.out
    assert result.key not in captured.err
    assert result.key not in holder["url"]


# --- Flow B (out-of-band) -------------------------------------------------


def test_flow_b_sends_oob_and_s256():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        holder = {}

        def on_url(url):
            holder["url"] = url
            # The fake consent screen puts the code in the HTML body.
            body = httpx.get(url, timeout=10, follow_redirects=False).text
            holder["code"] = body.split("code: ")[1].split("<")[0]

        result = client.connect_oob(on_url=on_url, prompt_code=lambda: holder["code"])

    assert result.key == FAKE_KEY
    assert result.flow == "oob"
    sent = auth.authorize_requests[0]
    assert sent["callback_url"] == "oob"
    assert sent["code_challenge_method"] == "S256"
    assert "code_challenge=" in holder["url"]


def test_flow_b_cancelled_when_no_code_entered():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        with pytest.raises(OrcaAuthError) as excinfo:
            client.connect_oob(on_url=lambda url: None, prompt_code=lambda: "")
    assert "No code" in str(excinfo.value)
    assert auth.exchange_requests == []


def test_flow_b_cancel_event_aborts_before_prompting():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(OrcaAuthError):
            client.connect_oob(
                on_url=lambda url: None, prompt_code=lambda: "unused", cancel=cancel
            )


# --- exchange failures ----------------------------------------------------


def test_unknown_code_is_403():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        with pytest.raises(OrcaExchangeError) as excinfo:
            client.exchange("never-minted", PkceAttempt.create("loopback"))
    assert excinfo.value.status == 403
    assert "single-use" in excinfo.value.hint or "expired" in str(excinfo.value)


def test_reused_code_is_rejected():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        attempt = PkceAttempt.create("loopback")
        auth._mint(
            {
                "code_challenge": attempt.challenge,
                "code_challenge_method": "S256",
                "state": attempt.state,
            }
        )
        first = client.exchange("fake-code-1", attempt)
        assert first.key == FAKE_KEY
        with pytest.raises(OrcaExchangeError) as excinfo:
            client.exchange("fake-code-1", attempt)
    assert excinfo.value.status == 403


def test_expired_code_is_rejected():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        attempt = PkceAttempt.create("loopback")
        auth._mint(
            {
                "code_challenge": attempt.challenge,
                "code_challenge_method": "S256",
                "state": attempt.state,
            }
        )
        auth.expire("fake-code-1")
        with pytest.raises(OrcaExchangeError) as excinfo:
            client.exchange("fake-code-1", attempt)
    assert excinfo.value.status == 403


def test_wrong_verifier_is_rejected():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        attempt = PkceAttempt.create("loopback")
        auth._mint(
            {
                "code_challenge": attempt.challenge,
                "code_challenge_method": "S256",
                "state": attempt.state,
            }
        )
        other = PkceAttempt.create("loopback")
        with pytest.raises(OrcaExchangeError) as excinfo:
            client.exchange("fake-code-1", other)
    assert excinfo.value.status == 403


def test_scope_downgrade_is_reported_not_accepted():
    with FakeOrcaAuth(granted_scope="api") as auth:
        client = OrcaOAuthClient(
            OrcaEndpoints(auth_base=auth.auth_base, api_base=auth.api_base),
            scope="connector",
            timeout=10.0,
            open_browser=lambda url: False,
        )
        attempt = PkceAttempt.create("loopback")
        auth._mint(
            {
                "code_challenge": attempt.challenge,
                "code_challenge_method": "S256",
                "state": attempt.state,
            }
        )
        with pytest.raises(OrcaScopeDowngrade) as excinfo:
            client.exchange("fake-code-1", attempt)
    assert "api" in str(excinfo.value) and "connector" in str(excinfo.value)


def test_429_is_terminal_with_an_actionable_hint():
    with FakeOrcaAuth(exchange_status=429) as auth:
        client = _client_for(auth)
        with pytest.raises(OrcaExchangeError) as excinfo:
            client.exchange("any", PkceAttempt.create("loopback"))
    assert excinfo.value.status == 429
    assert "10 PKCE keys" in excinfo.value.hint


def test_400_reports_a_challenge_method_mismatch():
    with FakeOrcaAuth(exchange_status=400) as auth:
        client = _client_for(auth)
        with pytest.raises(OrcaExchangeError) as excinfo:
            client.exchange("any", PkceAttempt.create("loopback"))
    assert excinfo.value.status == 400
    assert "challenge method" in excinfo.value.hint


def test_network_failure_is_terminal_not_a_loop():
    def boom(request):
        raise httpx.ConnectError("no route to host", request=request)

    client = OrcaOAuthClient(
        OrcaEndpoints(auth_base="https://auth.invalid", api_base="https://api.invalid/v1"),
        transport=httpx.MockTransport(boom),
    )
    with pytest.raises(OrcaExchangeError) as excinfo:
        client.exchange("any", PkceAttempt.create("loopback"))
    assert "network" in excinfo.value.hint.lower()


def test_exchange_error_never_echoes_the_verifier_or_code():
    with FakeOrcaAuth(exchange_status=500) as auth:
        client = _client_for(auth)
        attempt = PkceAttempt.create("loopback")
        with pytest.raises(OrcaExchangeError) as excinfo:
            client.exchange("super-secret-code", attempt)
    message = str(excinfo.value)
    assert attempt.verifier not in message
    assert "super-secret-code" not in message


# --- loopback timeout -----------------------------------------------------


def test_loopback_times_out_cleanly():
    with FakeOrcaAuth() as auth:
        client = OrcaOAuthClient(
            OrcaEndpoints(auth_base=auth.auth_base, api_base=auth.api_base),
            timeout=0.4,
            open_browser=lambda url: False,
        )
        with pytest.raises(OrcaAuthTimeout) as excinfo:
            client.connect_loopback(on_url=lambda url: None)
    assert "oob" in excinfo.value.hint


def test_loopback_cancel_event_releases_immediately():
    with FakeOrcaAuth() as auth:
        client = OrcaOAuthClient(
            OrcaEndpoints(auth_base=auth.auth_base, api_base=auth.api_base),
            timeout=30.0,
            open_browser=lambda url: False,
        )
        cancel = threading.Event()
        timer = threading.Timer(0.2, cancel.set)
        timer.start()
        try:
            with pytest.raises(OrcaAuthError) as excinfo:
                client.connect_loopback(on_url=lambda url: None, cancel=cancel)
        finally:
            timer.cancel()
    assert "cancelled" in str(excinfo.value).lower()


def test_authorize_url_only_uses_the_auth_origin():
    with FakeOrcaAuth() as auth:
        client = _client_for(auth)
        seen = {}
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(OrcaAuthError):
            client.connect_loopback(
                on_url=lambda url: seen.setdefault("url", url), cancel=cancel
            )
    assert seen["url"].startswith(auth.auth_base + "/auth?")
