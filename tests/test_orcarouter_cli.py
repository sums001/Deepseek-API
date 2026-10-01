"""The two command-line authentication entry points.

A CLI user must be able to choose either way in without the other, and neither
may print a key. ``key`` is the paste-a-key path; ``connect`` is the OAuth 2.0 +
PKCE path. ``status`` shows both as available.
"""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from fake_orca_auth import FAKE_KEY, FakeOrcaAuth

from server.orcarouter import cli
from server.orcarouter.credentials import CredentialStore


@pytest.fixture()
def cred_file(tmp_path):
    return tmp_path / "orcarouter.json"


def _run(argv, cred_file, capsys):
    # The global flags are positional-before-subcommand, as argparse requires.
    code = cli.main(["--credential-file", str(cred_file)] + argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _consent_code(url: str) -> str:
    """Read the code the fake consent screen displays for an OOB request."""
    body = httpx.get(url, timeout=10).text
    return body.split("code: ")[1].split("<")[0]


def _fake_browser(monkeypatch, holder):
    """Replace the browser opener; record the URL, optionally drive the callback."""

    def opener(url):
        holder["url"] = url
        if holder.get("drive"):
            threading.Thread(
                target=lambda: httpx.get(url, timeout=10, follow_redirects=True),
                daemon=True,
            ).start()
        return True

    monkeypatch.setattr("server.orcarouter.connect.webbrowser.open", opener)
    return holder


# --- the API-key entry point ----------------------------------------------


def test_key_command_saves_a_pasted_key(cred_file, capsys):
    code, out = _run(["key", "--api-key", FAKE_KEY], cred_file, capsys)
    assert code == cli.EXIT_OK
    assert "Saved" in out
    assert FAKE_KEY not in out                       # only the masked form is shown
    assert CredentialStore(cred_file).load().api_key == FAKE_KEY


def test_key_command_reads_the_secret_from_a_prompt_not_argv(cred_file, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: FAKE_KEY)
    code, out = _run(["key"], cred_file, capsys)
    assert code == cli.EXIT_OK
    assert FAKE_KEY not in out
    assert CredentialStore(cred_file).load().api_key == FAKE_KEY


def test_key_command_rejects_a_mistake_and_says_where_to_get_one(cred_file, capsys):
    code, out = _run(["key", "--api-key", "wrong-shape"], cred_file, capsys)
    assert code == cli.EXIT_FAILURE
    assert "does not look like" in out
    assert "orcarouter.ai" in out
    assert CredentialStore(cred_file).load() is None


def test_key_command_clears(cred_file, capsys):
    _run(["key", "--api-key", FAKE_KEY], cred_file, capsys)
    code, out = _run(["key", "--clear"], cred_file, capsys)
    assert code == cli.EXIT_OK
    assert "Removed" in out
    assert CredentialStore(cred_file).load() is None


def test_logout_command_clears(cred_file, capsys):
    _run(["key", "--api-key", FAKE_KEY], cred_file, capsys)
    code, out = _run(["logout"], cred_file, capsys)
    assert code == cli.EXIT_OK
    assert CredentialStore(cred_file).load() is None


# --- the PKCE entry point -------------------------------------------------


@pytest.fixture()
def fake_origins(monkeypatch):
    with FakeOrcaAuth() as auth:
        monkeypatch.setenv("ORCA_AUTH_BASE_URL", auth.auth_base)
        monkeypatch.setenv("ORCA_API_BASE_URL", auth.api_base)
        yield auth


def test_connect_command_runs_the_oob_flow_and_saves_the_key(
    cred_file, capsys, fake_origins, monkeypatch
):
    holder = {}
    _fake_browser(monkeypatch, holder)
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: _consent_code(holder["url"]))

    code, out = _run(["connect", "--flow", "oob"], cred_file, capsys)
    assert code == cli.EXIT_OK
    assert "Connected" in out
    assert FAKE_KEY not in out
    assert CredentialStore(cred_file).load().api_key == FAKE_KEY


def test_connect_command_runs_the_loopback_flow(cred_file, capsys, fake_origins, monkeypatch):
    holder = {"drive": True}
    _fake_browser(monkeypatch, holder)
    code, out = _run(["connect", "--flow", "loopback", "--timeout", "10"], cred_file, capsys)
    assert code == cli.EXIT_OK
    assert CredentialStore(cred_file).load().api_key == FAKE_KEY
    # The consent URL is printed for browsers that do not open by themselves.
    assert "127.0.0.1" in out


def test_connect_command_reports_a_denial_without_a_traceback(cred_file, capsys, monkeypatch):
    with FakeOrcaAuth(behavior="deny") as auth:
        monkeypatch.setenv("ORCA_AUTH_BASE_URL", auth.auth_base)
        monkeypatch.setenv("ORCA_API_BASE_URL", auth.api_base)
        holder = {"drive": True}
        _fake_browser(monkeypatch, holder)
        code, out = _run(["connect", "--flow", "loopback", "--timeout", "10"], cred_file, capsys)

    assert code == cli.EXIT_FAILURE
    assert "denied" in out.lower()
    assert "Traceback" not in out
    assert CredentialStore(cred_file).load() is None


def test_connect_command_reports_an_expired_window_without_hanging(
    cred_file, capsys, fake_origins, monkeypatch
):
    _fake_browser(monkeypatch, {})
    code, out = _run(["connect", "--flow", "loopback", "--timeout", "0.5"], cred_file, capsys)
    assert code == cli.EXIT_FAILURE
    assert "Timed out" in out
    assert "oob" in out            # the hint points at the alternative flow
    assert "Traceback" not in out


def test_connect_command_reports_a_cancelled_prompt(cred_file, capsys, fake_origins, monkeypatch):
    _fake_browser(monkeypatch, {})
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: "")
    code, out = _run(["connect", "--flow", "oob"], cred_file, capsys)
    assert code in (cli.EXIT_FAILURE, cli.EXIT_CANCELLED)
    assert "Traceback" not in out


def test_connect_command_reports_an_unreachable_auth_origin(cred_file, capsys, monkeypatch):
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://auth.invalid.example")
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://api.invalid.example/v1")
    _fake_browser(monkeypatch, {})
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: "some-code")
    code, out = _run(["connect", "--flow", "oob"], cred_file, capsys)
    assert code == cli.EXIT_FAILURE
    assert "Traceback" not in out


def test_connect_command_refuses_to_derive_one_origin_from_the_other(cred_file, capsys, monkeypatch):
    """A typo'd override is loud rather than silently routed to the public service."""
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "http://auth.example.com")
    code, out = _run(["connect", "--flow", "oob"], cred_file, capsys)
    assert code == cli.EXIT_FAILURE
    assert "https" in out


# --- status: both methods are visible and independent ---------------------


def test_status_lists_both_methods_as_unconfigured_initially(cred_file, capsys):
    code, out = _run(["status"], cred_file, capsys)
    assert code == cli.EXIT_OK
    assert "API key" in out and "PKCE" in out
    assert "not configured" in out


def test_status_after_a_pasted_key_shows_the_key_choice_available(cred_file, capsys):
    _run(["key", "--api-key", FAKE_KEY], cred_file, capsys)
    code, out = _run(["status", "--json"], cred_file, capsys)
    payload = json.loads(out)
    assert payload["api_key_configured"] is True
    assert payload["pkce_configured"] is False
    assert FAKE_KEY not in out


def test_status_after_a_login_shows_the_pkce_choice_available(
    cred_file, capsys, fake_origins, monkeypatch
):
    holder = {}
    _fake_browser(monkeypatch, holder)
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: _consent_code(holder["url"]))
    _run(["connect", "--flow", "oob"], cred_file, capsys)

    code, out = _run(["status", "--json"], cred_file, capsys)
    payload = json.loads(out)
    assert payload["pkce_configured"] is True
    assert payload["active"]["method"] == "pkce"
    assert payload["active"]["masked"].startswith("sk-orca-")
    assert FAKE_KEY not in out


def test_status_reports_a_degraded_origin_configuration(cred_file, capsys, monkeypatch):
    monkeypatch.setenv("ORCA_BASE_URL", "https://selfhost.example.com")
    code, out = _run(["status", "--json"], cred_file, capsys)
    payload = json.loads(out)
    assert code == cli.EXIT_OK
    assert payload["auth_base"] == "https://selfhost.example.com"
    assert payload["api_base"] == "https://selfhost.example.com/v1"


def test_both_entry_points_are_independently_reachable(cred_file, capsys, fake_origins, monkeypatch):
    """Using one does not require or disable the other."""
    code, out = _run(["key", "--api-key", FAKE_KEY], cred_file, capsys)
    assert code == cli.EXIT_OK

    holder = {}
    _fake_browser(monkeypatch, holder)
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: _consent_code(holder["url"]))
    code, out = _run(["connect", "--flow", "oob"], cred_file, capsys)
    assert code == cli.EXIT_OK


# --- models ---------------------------------------------------------------


def _patch_catalog(monkeypatch, handler):
    from server.orcarouter import cli as cli_module

    original = cli_module.OrcaCatalogClient

    class Patched(original):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            kwargs["cache_file"] = None
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli_module, "OrcaCatalogClient", Patched)


def test_models_command_prints_the_live_catalog(cred_file, capsys, monkeypatch):
    payload = {"data": [
        {"id": "acme/chat", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text"]}},
        {"id": "acme/vision", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "acme/embed", "supported_endpoint_types": ["embeddings"]},
    ]}
    _patch_catalog(monkeypatch, lambda request: httpx.Response(200, json=payload))

    code, out = _run(["models", "--json"], cred_file, capsys)
    payload_out = json.loads(out)
    assert code == cli.EXIT_OK
    assert payload_out["source"] == "live"
    assert sorted(m["id"] for m in payload_out["models"]) == ["acme/chat", "acme/vision"]


def test_models_command_shows_a_multimodal_filtered_list(cred_file, capsys, monkeypatch):
    payload = {"data": [
        {"id": "acme/chat", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text"]}},
        {"id": "acme/vision", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text", "image"]}},
    ]}
    _patch_catalog(monkeypatch, lambda request: httpx.Response(200, json=payload))
    code, out = _run(
        ["models", "--json", "--capability", "multimodal", "--modality", "image"],
        cred_file, capsys,
    )
    payload_out = json.loads(out)
    assert [m["id"] for m in payload_out["models"]] == ["acme/vision"]


def test_models_command_reports_the_degraded_seed_honestly(cred_file, capsys, monkeypatch):
    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    _patch_catalog(monkeypatch, boom)
    code, out = _run(["models", "--json"], cred_file, capsys)
    payload_out = json.loads(out)
    assert code == cli.EXIT_OK
    assert payload_out["source"] == "seed"
    assert payload_out["degraded"] is True
    assert payload_out["error"]


def test_no_command_output_contains_a_credential(cred_file, capsys, fake_origins, monkeypatch):
    holder = {}
    _fake_browser(monkeypatch, holder)
    monkeypatch.setattr(cli, "_read_secret", lambda prompt: _consent_code(holder["url"]))
    outputs = [
        _run(["connect", "--flow", "oob"], cred_file, capsys)[1],
        _run(["status"], cred_file, capsys)[1],
        _run(["status", "--json"], cred_file, capsys)[1],
    ]
    for out in outputs:
        assert FAKE_KEY not in out
