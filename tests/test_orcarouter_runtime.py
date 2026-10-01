"""The runtime singletons must be constructible without deadlocking.

`get_catalog()` needs the resolved endpoints, so it calls `get_endpoints()`.
If the guard lock is held across construction, that call re-enters a
non-reentrant lock and hangs - which is exactly what a unit test that
pre-populates the singletons would never notice.
"""

from __future__ import annotations

import threading

import pytest

from server.orcarouter import runtime


@pytest.fixture(autouse=True)
def reset():
    runtime.reset_runtime()
    yield
    runtime.reset_runtime()


def _within(seconds: float, fn):
    """Run ``fn`` on a worker thread and fail fast instead of hanging forever."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported below
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "call did not return - it deadlocked"
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_get_catalog_does_not_deadlock_with_no_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("ORCA_BASE_URL", raising=False)
    monkeypatch.delenv("ORCA_AUTH_BASE_URL", raising=False)

    catalog = _within(5.0, runtime.get_catalog)
    assert catalog is not None
    assert catalog.endpoints.api_base == "https://api.example.com/v1"


def test_each_singleton_is_stable_across_calls():
    assert runtime.get_endpoints() is runtime.get_endpoints()
    assert runtime.get_store() is runtime.get_store()
    assert runtime.get_catalog() is runtime.get_catalog()


def test_concurrent_first_use_does_not_deadlock():
    results = []

    def worker():
        results.append(runtime.get_catalog())

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert not any(t.is_alive() for t in threads)
    assert len(results) == 8


def test_reset_clears_the_singletons():
    first = runtime.get_catalog()
    runtime.reset_runtime()
    assert runtime.get_catalog() is not first
