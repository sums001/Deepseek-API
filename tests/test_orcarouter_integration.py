"""End-to-end checks that bind the pieces together.

Two properties are asserted here, request by request, because they are the ones
that fail silently when an integration is "done" but wrong:

* the model options handed to a selector come from the catalog API and from the
  capability filter for the entry point asking - not from a local list, and not
  from free text;
* authorization traffic goes to the auth origin and inference traffic to the
  inference base, for both credential kinds, including under overrides.

Everything runs over loopback against a fixture catalog and a fake consent
screen. No credential and no outbound network are involved.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from fake_orca_auth import FAKE_KEY, FakeOrcaAuth
from catalog_fixture import CATALOG_FIXTURE, EXPECTED

from server.orcarouter.catalog import (
    SEED_IDS,
    OrcaCatalogClient,
    filter_models,
    is_compatible,
)
from server.orcarouter.connect import OrcaOAuthClient
from server.orcarouter.credentials import (
    API_KEY_METHOD,
    PKCE_METHOD,
    CredentialStore,
    OrcaCredential,
    resolve_credential,
)
from server.orcarouter.endpoints import OrcaEndpoints, resolve_endpoints
from server.orcarouter.provider import OrcaRouterProvider


class _FixtureCatalogServer:
    """Serves the fixture catalog over real loopback HTTP."""

    def __init__(self, payload=None):
        self.payload = CATALOG_FIXTURE if payload is None else payload
        self.requests = []
        holder = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                return

            def do_GET(self):  # noqa: N802
                holder.requests.append(self.path)
                body = json.dumps(holder.payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


@pytest.fixture()
def catalog_server():
    with _FixtureCatalogServer() as server:
        yield server


@pytest.fixture()
def catalog_client(catalog_server):
    return OrcaCatalogClient(
        OrcaEndpoints(auth_base="https://www.orcarouter.ai", api_base=catalog_server.api_base),
        cache_file=None,
    )


# --- the options come from the API ----------------------------------------


def test_the_selector_options_come_from_the_catalog_api(catalog_client, catalog_server):
    result = catalog_client.fetch("chat", allow_cache=False)
    assert result.source == "live"
    assert not result.degraded
    assert any("capability=chat" in path for path in catalog_server.requests)
    assert sorted(result.ids) == sorted(EXPECTED["chat"])


def test_no_seed_is_mixed_into_a_live_result(catalog_client):
    result = catalog_client.fetch("chat", allow_cache=False)
    assert not (set(result.ids) & set(SEED_IDS))


def test_the_options_are_not_a_hand_written_list(catalog_client):
    """Changing the catalog changes the options."""
    other = CATALOG_FIXTURE["data"] + [
        {"id": "vendor/added-later", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text"]}}
    ]
    with _FixtureCatalogServer({"object": "list", "data": other}) as server:
        client = OrcaCatalogClient(
            OrcaEndpoints(auth_base="https://www.orcarouter.ai", api_base=server.api_base),
            cache_file=None,
        )
        assert "vendor/added-later" in client.fetch("chat", allow_cache=False).ids


# --- provider change / attachment change re-derives the options -----------


def test_adding_an_image_narrows_the_options_to_declared_models(catalog_client):
    chat = catalog_client.fetch("chat", allow_cache=False)
    multimodal = catalog_client.fetch(
        "multimodal", required_modalities=["image"], allow_cache=False
    )
    assert sorted(multimodal.ids) == sorted(EXPECTED["multimodal_image"])
    assert set(multimodal.ids) < set(chat.ids)


def test_a_selected_text_model_is_invalidated_by_the_attachment(catalog_client):
    chat = catalog_client.fetch("chat", allow_cache=False)
    multimodal = catalog_client.fetch(
        "multimodal", required_modalities=["image"], allow_cache=False
    )
    assert is_compatible("vendor/text-only", chat.models, "chat")
    assert not is_compatible(
        "vendor/text-only", multimodal.models, "multimodal", required_modalities=["image"]
    )


def test_switching_capability_changes_the_options_entirely(catalog_client):
    chat = catalog_client.fetch("chat", allow_cache=False)
    embedding = catalog_client.fetch("embedding", allow_cache=False)
    assert not (set(chat.ids) & set(embedding.ids))
    assert sorted(embedding.ids) == sorted(EXPECTED["embedding"])


@pytest.mark.parametrize(
    "capability,key",
    [("embedding", "embedding"), ("image", "image"),
     ("video", "video"), ("rerank", "rerank")],
)
def test_each_capability_keeps_to_its_own_kind(catalog_client, capability, key):
    result = catalog_client.fetch(capability, allow_cache=False)
    assert sorted(result.ids) == sorted(EXPECTED[key])


def test_a_multimodal_picker_never_offers_an_undeclared_model(catalog_client):
    result = catalog_client.fetch(
        "multimodal", required_modalities=["image"], allow_cache=False
    )
    for model in result.models:
        assert "image" in model.input_modalities
    assert filter_models(
        result.models, "multimodal", required_modalities=["audio"]
    ) == []


# --- the fallback is labelled and stays offline ---------------------------


def test_a_catalog_outage_degrades_to_the_labelled_seed():
    dead = OrcaEndpoints(auth_base="https://www.orcarouter.ai",
                         api_base="http://127.0.0.1:1/v1")
    result = OrcaCatalogClient(dead, cache_file=None, timeout=2.0).fetch("chat")
    assert result.source == "seed"
    assert result.degraded
    assert result.error
    assert sorted(result.ids) == sorted(SEED_IDS)


def test_the_degraded_fallback_keeps_verified_metadata():
    dead = OrcaEndpoints(auth_base="https://www.orcarouter.ai",
                         api_base="http://127.0.0.1:1/v1")
    result = OrcaCatalogClient(dead, cache_file=None, timeout=2.0).fetch("chat")
    gpt = next(m for m in result.models if m.id == "openai/gpt-5.5")
    assert gpt.reasoning_efforts == ("low", "medium", "high", "xhigh")


def test_the_degraded_fallback_offers_no_multimodal_model():
    dead = OrcaEndpoints(auth_base="https://www.orcarouter.ai",
                         api_base="http://127.0.0.1:1/v1")
    result = OrcaCatalogClient(dead, cache_file=None, timeout=2.0).fetch("chat")
    assert filter_models(result.models, "multimodal",
                         required_modalities=["image"]) == []


# --- origins: auth versus inference ---------------------------------------


def test_defaults_keep_the_two_origins_apart():
    endpoints = resolve_endpoints({})
    assert endpoints.auth_base == "https://www.orcarouter.ai"
    assert endpoints.api_base == "https://api.orcarouter.ai/v1"
    assert endpoints.exchange_url() == "https://www.orcarouter.ai/api/v1/auth/keys"
    assert endpoints.exchange_url() != "https://api.orcarouter.ai/v1/auth/keys"


def test_inference_paths_never_carry_the_auth_path():
    endpoints = resolve_endpoints({})
    assert "/auth" not in endpoints.models_url("chat")
    assert "/auth" not in endpoints.chat_completions_url()


def test_overrides_take_precedence_and_the_shared_base_fills_gaps():
    overridden = resolve_endpoints({
        "ORCA_BASE_URL": "https://shared.example.com",
        "ORCA_AUTH_BASE_URL": "https://auth.example.com",
    })
    assert overridden.auth_base == "https://auth.example.com"
    assert overridden.api_base == "https://shared.example.com/v1"


def test_a_single_self_hosted_origin_serves_both_purposes():
    endpoints = resolve_endpoints({"ORCA_BASE_URL": "https://selfhost.example.com"})
    assert endpoints.auth_base == "https://selfhost.example.com"
    assert endpoints.api_base == "https://selfhost.example.com/v1"


# --- both credential kinds, one inference path ----------------------------


def _recording_provider(credential, seen, api_base="https://api.orcarouter.ai/v1"):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url),
                     "auth": request.headers.get("authorization")})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    return OrcaRouterProvider(
        credential,
        endpoints=OrcaEndpoints(auth_base="https://www.orcarouter.ai", api_base=api_base),
        transport=transport,
        catalog=OrcaCatalogClient(cache_file=None, transport=transport),
    )


def test_both_credential_kinds_infer_on_the_inference_base():
    seen = []
    for method in (API_KEY_METHOD, PKCE_METHOD):
        credential = OrcaCredential(api_key=FAKE_KEY, method=method, user_id="1")
        _recording_provider(credential, seen).chat_completion(
            [{"role": "user", "content": "hi"}], "vendor/text-only"
        )
    assert [s["url"] for s in seen] == [
        "https://api.orcarouter.ai/v1/chat/completions",
        "https://api.orcarouter.ai/v1/chat/completions",
    ]
    assert all("www.orcarouter.ai" not in s["url"] for s in seen)


def test_inference_follows_an_explicit_api_override():
    seen = []
    credential = OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD)
    _recording_provider(credential, seen, api_base="https://api.example.com/v1").chat_completion(
        [{"role": "user", "content": "hi"}], "vendor/text-only"
    )
    assert seen[0]["url"] == "https://api.example.com/v1/chat/completions"


def test_the_connect_flow_only_ever_calls_the_auth_origin():
    with FakeOrcaAuth() as auth:
        client = OrcaOAuthClient(
            OrcaEndpoints(auth_base=auth.auth_base, api_base=auth.api_base),
            timeout=10.0,
            open_browser=lambda url: False,
        )
        holder = {}

        def on_url(url):
            holder["url"] = url
            holder["code"] = httpx.get(url, timeout=10).text.split("code: ")[1].split("<")[0]

        result = client.connect_oob(on_url=on_url, prompt_code=lambda: holder["code"])

    assert result.key == FAKE_KEY
    assert holder["url"].startswith(auth.auth_base + "/auth?")
    assert "code_challenge_method=S256" in holder["url"]
    assert "code_verifier" not in holder["url"]
    assert len(auth.exchange_requests) == 1


# --- resolution is independent of how the key arrived ---------------------


def test_a_stored_key_resolves_as_an_ordinary_credential(tmp_path):
    store = CredentialStore(tmp_path / "orcarouter.json")
    store.save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    resolved = resolve_credential(store, env={})
    assert resolved is not None and resolved.usable
    assert resolved.api_key.startswith("sk-orca-")


def test_switching_between_methods_needs_no_other_change(tmp_path):
    """The provider does not care which adapter produced the credential."""
    store = CredentialStore(tmp_path / "orcarouter.json")
    seen = []
    for method in (API_KEY_METHOD, PKCE_METHOD):
        store.save(OrcaCredential(api_key=FAKE_KEY, method=method))
        credential = resolve_credential(store, env={})
        result = _recording_provider(credential, seen).chat_completion(
            [{"role": "user", "content": "hi"}], "vendor/text-only"
        )
        assert result["choices"][0]["message"]["content"] == "ok"
    assert len(seen) == 2
