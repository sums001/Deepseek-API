"""The inference provider: routing, Bearer transport, and terminal 401 handling.

The behaviours that matter are that both credential kinds reach the *same*
inference origin through the same code path, that a revoked durable key becomes
a terminal reauthentication requirement rather than a retry loop, and that a
late failure from an old request cannot mark a newer credential as broken.
"""

from __future__ import annotations

import json

import httpx
import pytest

from server.orcarouter.catalog import OrcaCatalogClient
from server.orcarouter.credentials import (
    ACTIVE,
    NEEDS_REAUTH,
    CredentialStore,
    OrcaCredential,
)
from server.orcarouter.endpoints import OrcaEndpoints
from server.orcarouter.provider import (
    OrcaRouterProvider,
    OrcaUpstreamError,
    content_modalities,
)

API_BASE = "https://api.orcarouter.ai/v1"
_UNSET = object()


@pytest.fixture()
def store(tmp_path) -> CredentialStore:
    return CredentialStore(tmp_path / "orcarouter.json")


def _provider(handler, store=None, credential=_UNSET, **kwargs) -> OrcaRouterProvider:
    transport = httpx.MockTransport(handler)
    if credential is _UNSET:
        credential = OrcaCredential(api_key="sk-orca-fake-000000000000")
    return OrcaRouterProvider(
        credential,
        endpoints=OrcaEndpoints(auth_base="https://www.orcarouter.ai", api_base=API_BASE),
        store=store,
        transport=transport,
        catalog=OrcaCatalogClient(cache_file=None, transport=transport),
        **kwargs,
    )


def _ok_completion(seen: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "deepseek/deepseek-v4-pro",
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": "hi"},
                             "finish_reason": "stop"}],
            },
        )
    return handler


# --- transport ------------------------------------------------------------


def test_completion_goes_to_the_inference_origin_with_bearer_auth():
    seen = {}
    provider = _provider(_ok_completion(seen))
    result = provider.chat_completion([{"role": "user", "content": "hello"}],
                                      "deepseek/deepseek-v4-pro")
    assert seen["url"] == f"{API_BASE}/chat/completions"
    assert seen["auth"].startswith("Bearer sk-orca-")
    assert result["choices"][0]["message"]["content"] == "hi"


def test_both_credential_kinds_run_through_one_path(store):
    """A pasted key and a PKCE key are indistinguishable at inference time."""
    seen = {}
    provider = _provider(_ok_completion(seen))
    for credential in (
        OrcaCredential(api_key="sk-orca-from-paste-00000000", method="api_key"),
        OrcaCredential(api_key="sk-orca-from-login-00000000", method="pkce", user_id="9"),
    ):
        provider.credential = credential
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")
        assert seen["url"] == f"{API_BASE}/chat/completions"
    assert seen["auth"] == "Bearer sk-orca-from-login-00000000"


def test_provider_sends_no_orcarouter_specific_extras():
    seen = {}
    provider = _provider(_ok_completion(seen))
    body = provider.build_body(
        [{"role": "user", "content": "hi"}],
        "deepseek/deepseek-v4-pro",
        options={"temperature": 0.2, "conversation_id": "abc", "thinking": True,
                 "search": True, "max_tokens": 10},
    )
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 10
    for deepseek_only in ("conversation_id", "thinking", "search"):
        assert deepseek_only not in body


def test_vision_content_parts_are_forwarded_unchanged():
    seen = {}
    provider = _provider(_ok_completion(seen))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    provider.chat_completion(messages, "deepseek/deepseek-v4-pro")
    assert seen["body"]["messages"] == messages


# --- unconfigured / revoked credentials -----------------------------------


def test_no_credential_gives_an_actionable_401():
    provider = _provider(lambda r: httpx.Response(500), credential=None)
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")
    assert excinfo.value.status == 401
    assert "ORCAROUTER_API_KEY" in str(excinfo.value)
    assert "server.orcarouter connect" in str(excinfo.value)


def test_a_revoked_key_is_terminal_and_never_refreshed():
    credential = OrcaCredential(api_key="sk-orca-revoked-000000000", status=NEEDS_REAUTH)
    provider = _provider(lambda r: httpx.Response(500), credential=credential)
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")
    assert excinfo.value.needs_reauth
    assert "revoked" in str(excinfo.value)


def test_a_401_marks_only_the_rejected_generation(store):
    credential = store.save(OrcaCredential(api_key="sk-orca-old-key-00000000", user_id="7"))
    store.save(
        OrcaCredential(api_key="sk-orca-new-key-00000000", user_id="7",
                       generation=credential.generation + 1)
    )
    provider = _provider(lambda r: httpx.Response(401), store=store, credential=credential)
    with pytest.raises(OrcaUpstreamError):
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")

    # The stale request's 401 must not mark the freshly authorized credential.
    assert store.load().status == ACTIVE
    assert store.load().api_key == "sk-orca-new-key-00000000"


def test_a_401_from_the_current_generation_marks_the_account(store):
    credential = store.save(OrcaCredential(api_key="sk-orca-live-key-0000000", user_id="7"))
    provider = _provider(lambda r: httpx.Response(401), store=store, credential=credential)
    with pytest.raises(OrcaUpstreamError):
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")
    assert store.load().status == NEEDS_REAUTH


def test_no_refresh_request_is_made_after_a_401(store):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    credential = store.save(OrcaCredential(api_key="sk-orca-live-key-0000000", user_id="7"))
    provider = _provider(handler, store=store, credential=credential)
    with pytest.raises(OrcaUpstreamError):
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")

    assert calls == [f"{API_BASE}/chat/completions"]
    assert not any("refresh" in url for url in calls)


# --- error surfaces -------------------------------------------------------


def test_429_surfaces_retry_after():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}},
                              headers={"retry-after": "12"})

    provider = _provider(handler)
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")
    assert excinfo.value.status == 429
    assert excinfo.value.err_type == "rate_limit_error"
    assert excinfo.value.retry_after == 12.0


def test_network_failure_is_reported_not_raised_raw():
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    provider = _provider(boom)
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")
    assert excinfo.value.err_type == "upstream_unreachable"


def test_upstream_error_detail_never_contains_a_key():
    def handler(request):
        return httpx.Response(
            502,
            json={"error": {"message": "bad gateway for sk-orca-abcdefghijklmnop"}},
        )

    provider = _provider(handler)
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.chat_completion([{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro")
    assert "sk-orca-abcdefghijklmnop" not in str(excinfo.value)
    assert "<redacted>" in str(excinfo.value)


# --- the compatibility guard ----------------------------------------------


def _catalog_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [
        {"id": "acme/text", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text"]}},
        {"id": "acme/vision", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "acme/embed", "supported_endpoint_types": ["embeddings"]},
    ]})


def test_guard_refuses_a_non_chat_model():
    provider = _provider(_catalog_handler)
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.ensure_model_compatible("acme/embed", "chat")
    assert excinfo.value.status == 404
    assert "acme/embed" in str(excinfo.value)


def test_guard_refuses_an_attachment_an_undeclared_model_cannot_take():
    provider = _provider(_catalog_handler)
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.ensure_modalities_supported("acme/text", ["image"])
    assert excinfo.value.status == 400
    provider.ensure_modalities_supported("acme/vision", ["image"])  # no raise


def test_modality_detection_reads_the_message_parts():
    messages = [
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {}}]},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]},
    ]
    assert content_modalities(messages) == {"image", "audio"}
    assert content_modalities([{"role": "user", "content": "text only"}]) == set()


# --- streaming ------------------------------------------------------------


def test_stream_forwards_sse_frames_verbatim():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body.encode(),
                              headers={"content-type": "text/event-stream"})

    provider = _provider(handler)
    lines = list(provider.chat_completion(
        [{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro", stream=True
    ))
    joined = "\n".join(lines)
    assert '"hel"' in joined and '"lo"' in joined
    assert "[DONE]" in joined


def test_stream_error_status_raises_before_any_frames():
    provider = _provider(lambda r: httpx.Response(401, json={"error": {"message": "nope"}}))
    with pytest.raises(OrcaUpstreamError) as excinfo:
        list(provider.chat_completion(
            [{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro", stream=True
        ))
    assert excinfo.value.status == 401
