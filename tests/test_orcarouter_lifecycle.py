"""Login-lock lifecycle: every terminal path releases it, and a stale response
can never overwrite a newer login.

A local server can have a login abandoned at any moment - the user denies, the
tab is closed, the page is hidden into the back-forward cache, the method is
switched, the modal unmounts. The lock has to clear on all of them, and the
generation counter is what keeps a late reply from clobbering fresh
credentials.
"""

from __future__ import annotations

import threading
import time

import httpx
from fake_orca_auth import FAKE_KEY, FakeOrcaAuth

from server.orcarouter.connect import (
    ConnectResult,
    OrcaAuthDenied,
    OrcaAuthTimeout,
    OrcaLoginManager,
    OrcaOAuthClient,
)
from server.orcarouter.endpoints import OrcaEndpoints


def _result(flow: str = "loopback") -> ConnectResult:
    return ConnectResult(key=FAKE_KEY, user_id="12345", scope="api", flow=flow)


# --- basic lifecycle ------------------------------------------------------


def test_begin_sets_busy_and_bumps_the_generation():
    manager = OrcaLoginManager()
    first, _ = manager.begin("pkce")
    assert manager.busy
    assert first == 1
    manager.cancel()
    second, _ = manager.begin("pkce")
    # cancel() invalidates the abandoned attempt, so the new one is at least
    # two generations on.
    assert second > first


def test_success_publishes_and_releases():
    manager = OrcaLoginManager()
    generation, _ = manager.begin("pkce")
    assert manager.publish(generation, _result())
    assert not manager.busy
    assert manager.hint == ""


def test_failure_releases_with_a_hint():
    manager = OrcaLoginManager()
    generation, _ = manager.begin("pkce")
    assert manager.fail(generation, "the user denied the request")
    assert not manager.busy
    assert "denied" in manager.hint


def test_explicit_cancel_clears_state_synchronously():
    manager = OrcaLoginManager()
    generation, cancel = manager.begin("pkce")
    assert manager.cancel("Authorization cancelled.")
    assert not manager.busy
    assert manager.hint == "Authorization cancelled."
    assert cancel.is_set()
    assert manager.abandoned == 1


def test_cancelling_when_idle_is_a_no_op():
    manager = OrcaLoginManager()
    assert manager.cancel() is False
    assert not manager.busy


# --- the page-hide case ---------------------------------------------------


def test_pagehide_clears_busy_and_allows_a_second_login_without_remount():
    """The back-forward-cache shape: no unmount happens, so the handler itself
    must clear the state, not a guarded ``finally`` that will refuse to run."""
    manager = OrcaLoginManager()
    first, first_cancel = manager.begin("pkce")
    assert manager.busy

    assert manager.abandon("Authorization window closed.") is True
    assert not manager.busy
    assert manager.hint
    assert first_cancel.is_set()

    # The page is restored from the bfcache; the same component starts again.
    second, _ = manager.begin("pkce")
    assert second > first
    assert manager.busy


def test_a_late_response_from_an_abandoned_attempt_is_ignored():
    manager = OrcaLoginManager()
    stale, _ = manager.begin("pkce")
    manager.abandon()
    fresh, _ = manager.begin("pkce")

    assert manager.publish(stale, _result()) is False
    assert manager.publish(fresh, _result()) is True
    assert manager.busy is False


def test_a_stale_failure_cannot_report_over_a_newer_login():
    manager = OrcaLoginManager()
    stale, _ = manager.begin("pkce")
    manager.abandon()
    fresh, _ = manager.begin("pkce")

    assert manager.fail(stale, "old attempt failed") is False
    assert manager.busy          # the new attempt is untouched
    assert manager.hint == ""


def test_switching_authentication_method_releases_the_first_attempt():
    manager = OrcaLoginManager()
    generation, cancel = manager.begin("pkce")
    manager.cancel("Switched to API key.")
    new_generation, _ = manager.begin("api_key")
    assert new_generation > generation
    assert cancel.is_set()
    assert manager.method == "api_key"


# --- run() integration ----------------------------------------------------


def test_run_happy_path_reports_the_credential():
    manager = OrcaLoginManager()
    seen = {}
    result = manager.run(
        "pkce",
        lambda generation, cancel: _result(),
        on_credential=lambda generation, r: seen.update({"generation": generation, "key": r.key}),
    )
    assert result.key == FAKE_KEY
    assert seen["generation"] == 1
    assert not manager.busy


def test_run_releases_the_lock_on_denial():
    manager = OrcaLoginManager()

    def work(generation, cancel):
        raise OrcaAuthDenied("Authorization was denied.", hint="Approve it.")

    errors = []
    assert manager.run("pkce", work, on_error=lambda g, m: errors.append(m)) is None
    assert not manager.busy
    assert errors and "denied" in errors[0].lower()


def test_run_releases_the_lock_on_timeout():
    manager = OrcaLoginManager()

    def work(generation, cancel):
        raise OrcaAuthTimeout("Timed out waiting for the browser.")

    assert manager.run("pkce", work) is None
    assert not manager.busy
    assert "Timed out" in manager.hint


def test_run_releases_the_lock_on_an_unexpected_error():
    manager = OrcaLoginManager()

    def work(generation, cancel):
        raise RuntimeError("something nobody predicted")

    assert manager.run("pkce", work) is None
    assert not manager.busy
    assert "RuntimeError" in manager.hint


def test_run_does_not_publish_a_superseded_result():
    manager = OrcaLoginManager()
    published = []

    def work(generation, cancel):
        # The user cancels and starts over while this attempt is "in flight".
        manager.cancel()
        return _result()

    result = manager.run("pkce", work, on_credential=lambda g, r: published.append(g))
    assert result is None
    assert published == []


# --- cancel/timeout against the real flow ---------------------------------


def _client(auth: FakeOrcaAuth, timeout: float = 10.0) -> OrcaOAuthClient:
    return OrcaOAuthClient(
        OrcaEndpoints(auth_base=auth.auth_base, api_base=auth.api_base),
        timeout=timeout,
        open_browser=lambda url: False,
    )


def test_cancel_releases_a_running_loopback_login():
    with FakeOrcaAuth() as auth:
        manager = OrcaLoginManager()

        def work(generation, cancel):
            return _client(auth).connect_loopback(on_url=lambda url: None, cancel=cancel)

        timer = threading.Timer(0.2, manager.cancel)
        timer.start()
        try:
            started = time.monotonic()
            assert manager.run("pkce", work) is None
            elapsed = time.monotonic() - started
        finally:
            timer.cancel()

    assert elapsed < 5.0            # released promptly, not after the timeout
    assert not manager.busy
    assert manager.hint


def test_denial_releases_a_running_loopback_login():
    with FakeOrcaAuth(behavior="deny") as auth:
        manager = OrcaLoginManager()

        def work(generation, cancel):
            def on_url(url):
                threading.Thread(
                    target=lambda: httpx.get(url, timeout=10, follow_redirects=True),
                    daemon=True,
                ).start()

            return _client(auth).connect_loopback(on_url=on_url, cancel=cancel)

        assert manager.run("pkce", work) is None
        assert not manager.busy
        assert "denied" in manager.hint.lower()


def test_reset_clears_everything_and_signals_waiters():
    manager = OrcaLoginManager()
    generation, cancel = manager.begin("pkce")
    manager.reset()
    assert not manager.busy
    assert manager.hint == ""
    assert cancel.is_set()
