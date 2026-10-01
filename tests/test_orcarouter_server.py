"""Server integration: routing, the merged catalog, and the control surface.

These exercise the real FastAPI app - the same `ChatCompletionRequest`
validation, the same router, the same provider code path - with the network
replaced by a mock transport. What is being checked is that adding OrcaRouter
did not disturb the existing DeepSeek route, that an OrcaRouter request reaches
the gateway through the provider, and that neither choice leaks a credential.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import server.api as api
from server.orcarouter import runtime
from server.orcarouter.catalog import OrcaCatalogClient
from server.orcarouter.credentials import API_KEY_METHOD, CredentialStore, OrcaCredential
from server.orcarouter.endpoints import OrcaEndpoints
from server.orcarouter.provider import OrcaRouterProvider

FAKE_KEY = "sk-orca-fake-0000000000000000000000"

CATALOG = {
    "data": [
        {"id": "acme/chat", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text"]}, "context_length": 128000},
        {"id": "acme/vision", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "acme/embed", "supported_endpoint_types": ["embeddings"]},
    ]
}


DEEPSEEK_EXPERT = "routed-deepseek-expert"


class FakeDeepSeekClient:
    """Stands in for the Playwright-backed client on the DeepSeek route."""

    def chat(self, prompt, conversation_id=None, model=None, thinking=False, search=False):
        from deepseek.client import Reply

        return Reply(text=f"deepseek:{prompt}", conversation_id="cid-1")

    def stream(self, prompt, conversation_id=None, model=None, thinking=False, search=False):
        return iter(["deep", "seek"])


@pytest.fixture(autouse=True)
def clean_runtime():
    runtime.reset_runtime()
    # A fresh limiter per test: the counter is process-wide, so without this a
    # suite with many /v1 requests would start tripping its own rate limit.
    api._rate_limiter._hits.clear()
    yield
    runtime.reset_runtime()
    api._rate_limiter._hits.clear()


@pytest.fixture()
def orca_env(tmp_path, monkeypatch):
    """Point the runtime at a mock gateway, a temp store, and no real network."""
    calls = {"chat": [], "models": []}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/models?capability=chat"):
            calls["models"].append(url)
            return httpx.Response(200, json=CATALOG)
        if "/chat/completions" in url:
            calls["chat"].append(
                {"url": url, "auth": request.headers.get("authorization"),
                 "body": json.loads(request.content)}
            )
            body = json.loads(request.content)
            model = body.get("model")
            if model == "acme/embed":
                return httpx.Response(400, json={"error": {"message": "not a chat model"}})
            if model == "acme/chat" and any(
                isinstance(p, dict) and "image" in str(p.get("type"))
                for m in body.get("messages", [])
                for p in (m.get("content") if isinstance(m.get("content"), list) else [])
            ):
                return httpx.Response(400, json={"error": {"message": "no image support"}})
            return httpx.Response(200, json={
                "id": "chatcmpl-1", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "routed"}}],
            })
        return httpx.Response(404, json={"error": "unexpected"})

    transport = httpx.MockTransport(handler)
    endpoints = OrcaEndpoints(
        auth_base="https://www.orcarouter.ai", api_base="https://api.orcarouter.ai/v1"
    )
    store = CredentialStore(tmp_path / "orcarouter.json")
    catalog = OrcaCatalogClient(endpoints, transport=transport, cache_file=None)

    runtime._endpoints = endpoints
    runtime._store = store
    runtime._catalog = catalog

    def get_provider(credential=None):
        return OrcaRouterProvider(
            credential if credential is not None else runtime.current_credential(),
            endpoints=endpoints,
            store=store,
            transport=transport,
            catalog=catalog,
        )

    monkeypatch.setattr(runtime, "get_provider", get_provider)
    monkeypatch.setattr(api, "get_client", lambda: FakeDeepSeekClient())
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ORCA_BASE_URL", raising=False)
    monkeypatch.delenv("ORCA_AUTH_BASE_URL", raising=False)
    monkeypatch.delenv("ORCA_API_BASE_URL", raising=False)
    return {"calls": calls, "store": store, "endpoints": endpoints, "catalog": catalog}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(api.app)


def _post_chat(client, **body):
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    payload.update(body)
    return client.post("/v1/chat/completions", json=payload)


# --- the model catalog ----------------------------------------------------


def test_models_endpoint_lists_the_builtin_models_without_orcarouter(client, orca_env):
    body = client.get("/v1/models").json()
    ids = [m["id"] for m in body["data"]]
    assert "deepseek-chat" in ids
    assert "deepseek-expert" in ids
    # No credential is configured, so no gateway entries are merged in.
    assert not any("/" in i for i in ids)


def test_models_endpoint_merges_the_live_orcarouter_catalog(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    body = client.get("/v1/models").json()
    by_id = {m["id"]: m for m in body["data"]}

    assert "deepseek-chat" in by_id                 # unchanged
    assert "acme/chat" in by_id                     # merged
    assert "acme/embed" not in by_id                # not a chat model
    assert by_id["acme/chat"]["context_length"] == 128000
    assert by_id["acme/chat"]["input_modalities"] == ["text"]  # metadata, not just ids


def test_models_endpoint_never_exposes_a_credential(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    raw = client.get("/v1/models").text
    assert FAKE_KEY not in raw
    assert "sk-orca" not in raw


def test_models_endpoint_survives_a_catalog_outage(client, orca_env, monkeypatch):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))

    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    broken = OrcaCatalogClient(
        orca_env["endpoints"], transport=httpx.MockTransport(boom), cache_file=None
    )

    def provider_factory(credential=None):
        return OrcaRouterProvider(
            runtime.current_credential(),
            endpoints=orca_env["endpoints"],
            store=orca_env["store"],
            transport=httpx.MockTransport(boom),
            catalog=broken,
        )

    # Both discovery paths - the provider's and the server's shared catalog -
    # are offline for this test.
    monkeypatch.setattr(runtime, "get_provider", provider_factory)
    monkeypatch.setattr(runtime, "get_catalog", lambda: broken)
    body = client.get("/v1/models").json()
    ids = [m["id"] for m in body["data"]]
    assert "deepseek-chat" in ids          # the built-in list still answers
    assert "acme/chat" not in ids          # nothing stale is served
    # The only namespaced entries are the verified, labelled seed.
    from server.orcarouter.catalog import SEED_IDS

    assert set(i for i in ids if "/" in i) <= set(SEED_IDS)


def test_models_endpoint_degrades_to_the_verified_seed_not_invented_names(
    client, orca_env, monkeypatch
):
    """With discovery down and nothing cached, only the verified seed appears."""
    from server.orcarouter.catalog import SEED_IDS

    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))

    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    broken = OrcaCatalogClient(
        orca_env["endpoints"], transport=httpx.MockTransport(boom), cache_file=None
    )
    monkeypatch.setattr(runtime, "get_catalog", lambda: broken)
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "deepseek-chat" in ids
    # The live fixture models are gone; only the verified seed's namespaced
    # ids remain, and the DeepSeek ids are untouched.
    assert "acme/chat" not in ids
    assert set(i for i in ids if "/" in i) <= set(SEED_IDS)


# --- routing --------------------------------------------------------------


def test_an_explicit_prefix_routes_to_orcarouter(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(client, model="orcarouter:acme/chat")
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "routed"

    call = orca_env["calls"]["chat"][0]
    assert call["url"] == "https://api.orcarouter.ai/v1/chat/completions"
    assert call["auth"] == f"Bearer {FAKE_KEY}"
    assert call["body"]["model"] == "acme/chat"     # namespace preserved, prefix stripped


def test_a_catalog_model_id_routes_to_orcarouter_without_a_prefix(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(client, model="acme/chat")
    assert response.status_code == 200
    assert orca_env["calls"]["chat"][0]["body"]["model"] == "acme/chat"


def test_a_deepseek_model_still_uses_the_local_bridge(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(client, model="deepseek-chat")
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "deepseek:hi"
    assert orca_env["calls"]["chat"] == []          # nothing left for the gateway


def test_an_unknown_model_is_still_a_404(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(client, model="gpt-4-turbo")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_orcarouter_without_a_credential_says_how_to_configure_it(client, orca_env):
    response = _post_chat(client, model="orcarouter:acme/chat")
    assert response.status_code == 401
    message = response.json()["error"]["message"]
    assert "ORCAROUTER_API_KEY" in message
    assert "server.orcarouter connect" in message
    assert response.headers.get("X-OrcaRouter-Reauth") == "required"


def test_both_credential_kinds_route_identically(client, orca_env):
    """Same result whether the key was pasted or obtained by PKCE login."""
    for method in (API_KEY_METHOD, "pkce"):
        orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=method, user_id="9"))
        response = _post_chat(client, model="orcarouter:acme/chat")
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "routed"
    assert len(orca_env["calls"]["chat"]) == 2
    assert {c["url"] for c in orca_env["calls"]["chat"]} == {
        "https://api.orcarouter.ai/v1/chat/completions"
    }


def test_both_credential_kinds_are_refused_identically_when_revoked(client, orca_env):
    """A terminal 401 is a property of the upstream, not of how the key arrived."""
    for method in (API_KEY_METHOD, "pkce"):
        orca_env["store"].save(
            OrcaCredential(api_key=FAKE_KEY, method=method, status="needs_reauth")
        )
        response = _post_chat(client, model="orcarouter:acme/chat")
        assert response.status_code == 401
        assert response.headers.get("X-OrcaRouter-Reauth") == "required"
        assert orca_env["calls"]["chat"] == []


# --- the second-layer capability guard ------------------------------------


def test_a_non_chat_model_cannot_be_routed_through_the_chat_endpoint(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(client, model="acme/embed")
    # 404 here means the picker path let it through but the guard caught it.
    assert response.status_code in (404, 502)
    assert orca_env["calls"]["chat"] == []


def test_an_undeclared_attachment_is_refused(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(
        client,
        model="orcarouter:acme/chat",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    )
    assert response.status_code == 400
    assert "image" in response.json()["error"]["message"]
    assert orca_env["calls"]["chat"] == []


def test_a_declared_attachment_is_accepted(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(
        client,
        model="orcarouter:acme/vision",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    )
    assert response.status_code == 200
    sent = orca_env["calls"]["chat"][0]["body"]
    assert sent["messages"][0]["content"][1]["type"] == "image_url"


def test_deepseek_only_extras_are_not_forwarded(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    _post_chat(client, model="orcarouter:acme/chat", thinking=True, search=True,
               conversation_id="cid-9", temperature=0.3)
    sent = orca_env["calls"]["chat"][0]["body"]
    assert sent["temperature"] == 0.3
    for deepseek_only in ("thinking", "search", "conversation_id"):
        assert deepseek_only not in sent


def test_standard_openai_parameters_survive_validation(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = _post_chat(
        client,
        model="orcarouter:acme/chat",
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="auto",
        response_format={"type": "json_object"},
        max_tokens=32,
        stream_options={"include_usage": True},
    )
    assert response.status_code == 200
    sent = orca_env["calls"]["chat"][0]["body"]
    assert sent["tools"][0]["function"]["name"] == "f"
    assert sent["response_format"] == {"type": "json_object"}


# --- streaming ------------------------------------------------------------


def test_streaming_returns_the_gateway_sse(client, orca_env, monkeypatch):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))

    def handler(request: httpx.Request) -> httpx.Response:
        frames = (
            'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=frames.encode(),
                              headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        runtime, "get_provider",
        lambda credential=None: OrcaRouterProvider(
            runtime.current_credential(), endpoints=orca_env["endpoints"],
            store=orca_env["store"], transport=httpx.MockTransport(handler),
            catalog=orca_env["catalog"],
        ),
    )
    response = _post_chat(client, model="orcarouter:acme/chat", stream=True)
    assert response.status_code == 200
    assert '"he"' in response.text and '"llo"' in response.text
    assert "[DONE]" in response.text


# --- the control surface --------------------------------------------------


def test_status_reports_both_authentication_methods(client, orca_env):
    body = client.get("/v1/orcarouter/status").json()
    assert body["auth_base"] == "https://www.orcarouter.ai"
    assert body["api_base"] == "https://api.orcarouter.ai/v1"
    assert body["configured"] is False
    assert body["needs_reauth"] is False
    assert body["busy"] is False


def test_status_shows_a_redacted_credential_and_hides_the_key(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = client.get("/v1/orcarouter/status")
    body = response.json()
    assert body["configured"] is True
    assert body["masked"].startswith("sk-orca-")
    assert FAKE_KEY not in response.text


def test_api_key_method_can_be_configured_over_http(client, orca_env):
    response = client.post("/v1/orcarouter/connect/start",
                           json={"method": "api_key", "api_key": FAKE_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert FAKE_KEY not in response.text
    assert orca_env["store"].load().api_key == FAKE_KEY


def test_api_key_method_rejects_an_obvious_mistake(client, orca_env):
    response = client.post("/v1/orcarouter/connect/start",
                           json={"method": "api_key", "api_key": "hunter2"})
    assert response.status_code == 400
    assert orca_env["store"].load() is None


def test_logout_clears_the_stored_credential(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    assert client.post("/v1/orcarouter/logout").json()["cleared"] is True
    assert orca_env["store"].load() is None


def test_catalog_endpoint_returns_filtered_metadata_only(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    response = client.get("/v1/orcarouter/models", params={"capability": "chat"})
    body = response.json()
    assert body["source"] == "live"
    assert body["degraded"] is False
    assert sorted(m["id"] for m in body["data"]) == ["acme/chat", "acme/vision"]
    assert FAKE_KEY not in response.text


def test_multimodal_catalog_endpoint_filters_fail_closed(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    body = client.get("/v1/orcarouter/models",
                      params={"capability": "multimodal", "modality": "image"}).json()
    assert [m["id"] for m in body["data"]] == ["acme/vision"]


def test_an_unknown_capability_is_rejected(client, orca_env):
    response = client.get("/v1/orcarouter/models", params={"capability": "telepathy"})
    assert response.status_code == 400
    assert "capability" in response.json()["error"]["message"]


def test_login_lock_is_released_by_cancel_and_pagehide(client, orca_env):
    manager = runtime.get_login_manager()
    manager.begin("pkce")
    assert client.get("/v1/orcarouter/status").json()["busy"] is True

    assert client.post("/v1/orcarouter/connect/cancel").json()["cancelled"] is True
    assert manager.busy is False

    manager.begin("pkce")
    pagehide = client.post("/v1/orcarouter/connect/pagehide").json()
    assert pagehide["busy"] is False
    assert pagehide["can_start_again"] is True
    # A second login can start without remounting anything.
    assert manager.begin("pkce")[0] > pagehide["generation"]


def test_a_second_connect_while_busy_is_a_409(client, orca_env):
    runtime.get_login_manager().begin("pkce")
    response = client.post("/v1/orcarouter/connect/start", json={"method": "pkce"})
    assert response.status_code == 409
    assert response.json()["error"]["type"] == "login_in_progress"


def test_no_response_body_ever_contains_the_key(client, orca_env):
    orca_env["store"].save(OrcaCredential(api_key=FAKE_KEY, method=API_KEY_METHOD))
    for path in ("/v1/orcarouter/status", "/v1/orcarouter/models", "/v1/models"):
        assert FAKE_KEY not in client.get(path).text
    assert FAKE_KEY not in _post_chat(client, model="orcarouter:acme/chat").text
