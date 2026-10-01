"""Model discovery, capability filtering, and the catalog fallback chain.

The rule this file defends: the model control is populated from the catalog the
service actually advertises, filtered by the capability of the entry point that
is asking. A model that does not *declare* a capability is not offered for it -
absence of a declaration is never treated as permission - and live success is
authoritative, with the verified seed used only as a bounded, clearly-labelled
degraded fallback.
"""

from __future__ import annotations

import json

import httpx
import pytest

from server.orcarouter.catalog import (
    SEED_IDS,
    SEED_MODELS,
    OrcaCatalogClient,
    filter_models,
    is_compatible,
    parse_catalog,
    seed_for,
)
from server.orcarouter.credentials import OrcaCredential
from server.orcarouter.endpoints import OrcaEndpoints

# A fixture catalog with one model of each shape the product needs to filter,
# including deliberate traps: a video model that also lists `openai`, and a
# chat model with no declared modalities at all.
CATALOG_FIXTURE = {
    "object": "list",
    "data": [
        {
            "id": "acme/text-model",
            "object": "model",
            "supported_endpoint_types": ["openai"],
            "architecture": {"input_modalities": ["text"]},
            "context_length": 128000,
        },
        {
            "id": "acme/vision-chat",
            "object": "model",
            "supported_endpoint_types": ["openai", "anthropic"],
            "architecture": {"input_modalities": ["text", "image"]},
            "context_length": 200000,
        },
        {
            "id": "acme/audio-chat",
            "object": "model",
            "supported_endpoint_types": ["openai-response"],
            "architecture": {"input_modalities": ["text", "audio"]},
        },
        {
            "id": "acme/video-chat",
            "object": "model",
            "supported_endpoint_types": ["openai-video"],
            "architecture": {"input_modalities": ["text"]},
        },
        {
            "id": "acme/undeclared-chat",
            "object": "model",
            "supported_endpoint_types": ["openai"],
        },
        {
            "id": "acme/embed",
            "object": "model",
            "supported_endpoint_types": ["embeddings"],
        },
        {
            "id": "acme/painter",
            "object": "model",
            "supported_endpoint_types": ["image-generation"],
        },
        {
            "id": "acme/movie-maker",
            "object": "model",
            "supported_endpoint_types": ["openai-video"],
        },
        {
            "id": "acme/reranker",
            "object": "model",
            "supported_endpoint_types": ["jina-rerank"],
        },
        {
            "id": "acme/unknown-route",
            "object": "model",
            "supported_endpoint_types": ["something-else"],
        },
        {
            "id": "acme/gpt-5.5",
            "object": "model",
            "supported_endpoint_types": ["openai"],
            "architecture": {"input_modalities": ["text"]},
            "reasoning_efforts": ["low", "medium", "high", "xhigh"],
            "context_length": 400000,
        },
    ],
}

LIVE_IDS = [m["id"] for m in CATALOG_FIXTURE["data"]]


@pytest.fixture()
def parsed():
    return parse_catalog(CATALOG_FIXTURE)


# --- parsing --------------------------------------------------------------


def test_parse_preserves_vendor_namespaced_ids(parsed):
    assert "acme/text-model" in [m.id for m in parsed]
    assert all("/" in m.id for m in parsed)


def test_parse_keeps_context_and_modality_metadata(parsed):
    model = next(m for m in parsed if m.id == "acme/vision-chat")
    assert model.context_length == 200000
    assert "image" in model.input_modalities
    reasoning = next(m for m in parsed if m.id == "acme/gpt-5.5")
    assert reasoning.reasoning_efforts == ("low", "medium", "high", "xhigh")


def test_parse_rejects_a_malformed_payload():
    with pytest.raises(ValueError):
        parse_catalog({"not": "a list"})
    with pytest.raises(ValueError):
        parse_catalog([{"id": "x"}])


def test_parse_skips_records_it_cannot_understand():
    parsed = parse_catalog(
        {"data": [{"no_id": True}, "a string", {"id": "  "}, {"id": "ok/model"}]}
    )
    assert [m.id for m in parsed] == ["ok/model"]


def test_parse_deduplicates_and_drops_non_string_modalities():
    parsed = parse_catalog(
        {
            "data": [
                {"id": "a/b", "architecture": {"input_modalities": ["text", 7, None]}},
                {"id": "a/b"},
            ]
        }
    )
    assert len(parsed) == 1
    assert parsed[0].input_modalities == frozenset({"text"})


# --- capability filtering (chat) ------------------------------------------


def test_chat_excludes_models_specialised_for_other_modalities(parsed):
    ids = [m.id for m in filter_models(parsed, "chat")]
    assert "acme/embed" not in ids
    assert "acme/painter" not in ids
    assert "acme/movie-maker" not in ids
    assert "acme/reranker" not in ids
    assert "acme/unknown-route" not in ids
    # A chat model that also advertises a video route is not treated as text.
    assert "acme/video-chat" not in ids


def test_chat_accepts_every_supported_text_wire_format(parsed):
    ids = [m.id for m in filter_models(parsed, "chat")]
    assert "acme/text-model" in ids          # openai
    assert "acme/audio-chat" in ids          # openai-response
    assert "acme/vision-chat" in ids         # openai + anthropic


def test_a_text_model_without_declared_modalities_is_still_a_chat_model(parsed):
    """The chat capability needs a text endpoint, not a declared modality."""
    assert "acme/undeclared-chat" in [m.id for m in filter_models(parsed, "chat")]


# --- capability filtering (multimodal) ------------------------------------


def test_multimodal_keeps_only_chat_models_declaring_the_attachment(parsed):
    ids = [m.id for m in filter_models(parsed, "multimodal", required_modalities=["image"])]
    assert ids == ["acme/vision-chat"]


def test_multimodal_fails_closed_for_undeclared_models(parsed):
    """The trap: a text-only chat model must not appear for an image request."""
    ids = [m.id for m in filter_models(parsed, "multimodal", required_modalities=["image"])]
    assert "acme/undeclared-chat" not in ids
    assert "acme/text-model" not in ids
    assert "acme/gpt-5.5" not in ids


def test_multimodal_audio_filtering_is_separate_from_image(parsed):
    image_ids = {m.id for m in filter_models(parsed, "multimodal", required_modalities=["image"])}
    audio_ids = {m.id for m in filter_models(parsed, "multimodal", required_modalities=["audio"])}
    assert image_ids == {"acme/vision-chat"}
    assert audio_ids == {"acme/audio-chat"}
    assert not (image_ids & audio_ids)


def test_multimodal_without_a_required_modality_offers_nothing(parsed):
    assert filter_models(parsed, "multimodal") == []


def test_multimodal_requires_all_named_modalities(parsed):
    assert filter_models(
        parsed, "multimodal", required_modalities=["image", "audio"]
    ) == []


# --- capability filtering (the non-chat routes) ---------------------------


def test_embedding_filter(parsed):
    assert [m.id for m in filter_models(parsed, "embedding")] == ["acme/embed"]


def test_image_generation_filter(parsed):
    assert [m.id for m in filter_models(parsed, "image")] == ["acme/painter"]


def test_video_filter(parsed):
    ids = [m.id for m in filter_models(parsed, "video")]
    assert "acme/movie-maker" in ids
    assert "acme/embed" not in ids


def test_rerank_filter(parsed):
    assert [m.id for m in filter_models(parsed, "rerank")] == ["acme/reranker"]


def test_chat_is_a_superset_of_the_chat_capable_multimodal_models(parsed):
    chat = {m.id for m in filter_models(parsed, "chat")}
    multimodal = {
        m.id for m in filter_models(parsed, "multimodal", required_modalities=["image"])
    }
    assert multimodal <= chat


def test_unknown_capability_is_a_programming_error(parsed):
    with pytest.raises(ValueError):
        filter_models(parsed, "telepathy")


# --- stale selections -----------------------------------------------------


def test_a_remembered_model_that_was_deselected_is_not_compatible(parsed):
    # The user attached an image while a text-only model was selected.
    assert is_compatible("acme/text-model", parsed, "chat")
    assert not is_compatible(
        "acme/text-model", parsed, "multimodal", required_modalities=["image"]
    )
    # ...and the model that should replace it is.
    assert is_compatible(
        "acme/vision-chat", parsed, "multimodal", required_modalities=["image"]
    )


def test_unknown_and_empty_ids_are_not_compatible(parsed):
    assert not is_compatible("acme/deleted", parsed, "chat")
    assert not is_compatible("", parsed, "chat")


# --- the catalog client ---------------------------------------------------


def _client(payload, *, transport=None, **kwargs) -> OrcaCatalogClient:
    return OrcaCatalogClient(
        OrcaEndpoints(auth_base="https://www.orcarouter.ai",
                      api_base="https://api.orcarouter.ai/v1"),
        transport=transport,
        **kwargs,
    )


def _transport(payload, status: int = 200, headers=None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers=headers or {})
    return httpx.MockTransport(handler)


def test_live_success_is_authoritative(tmp_path):
    client = _client(CATALOG_FIXTURE, transport=_transport(CATALOG_FIXTURE),
                     cache_file=tmp_path / "c.json")
    result = client.fetch("chat")
    assert result.source == "live"
    assert not result.degraded
    assert result.count == len([i for i in LIVE_IDS if i in result.ids])
    # No seed entries leaked into a successful live result.
    assert not (set(result.ids) & set(SEED_IDS))


def test_live_request_hits_the_api_origin_models_path(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=CATALOG_FIXTURE)

    client = _client(CATALOG_FIXTURE, transport=httpx.MockTransport(handler),
                     cache_file=tmp_path / "c.json")
    client.fetch("chat", credential=OrcaCredential(api_key="sk-orca-fake-000000000000"))
    assert seen["url"] == "https://api.orcarouter.ai/v1/models?capability=chat"
    assert seen["auth"] == "Bearer sk-orca-fake-000000000000"


def test_embedding_query_uses_the_embedding_capability(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": []})

    client = _client(CATALOG_FIXTURE, transport=httpx.MockTransport(handler),
                     cache_file=tmp_path / "c.json")
    client.fetch("embedding")
    assert seen["url"].endswith("?capability=embedding")


def test_catalog_failure_falls_back_to_the_verified_seed(tmp_path):
    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    client = _client(None, transport=httpx.MockTransport(boom), cache_file=tmp_path / "c.json")
    result = client.fetch("chat")
    assert result.source == "seed"
    assert result.degraded
    assert result.ids == list(SEED_IDS)
    assert result.error


def test_seed_metadata_is_preserved_including_reasoning_efforts(tmp_path):
    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    client = _client(None, transport=httpx.MockTransport(boom), cache_file=tmp_path / "c.json")
    result = client.fetch("chat")
    gpt = next(m for m in result.models if m.id == "openai/gpt-5.5")
    assert gpt.reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert "text" in gpt.input_modalities

    deepseek = next(m for m in result.models if m.id == "deepseek/deepseek-v4-pro")
    assert deepseek.context_length == 1048576


def test_an_http_error_status_falls_back_rather_than_returning_junk(tmp_path):
    client = _client(None, transport=_transport({"error": "boom"}, status=500),
                     cache_file=tmp_path / "c.json")
    result = client.fetch("chat")
    assert result.source == "seed"
    assert "500" in result.error


def test_a_401_is_reported_as_a_possible_revocation(tmp_path):
    client = _client(None, transport=_transport({}, status=401), cache_file=tmp_path / "c.json")
    result = client.fetch("chat")
    assert result.degraded
    assert "revoked" in result.error


def test_a_malformed_body_falls_back(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"<html>not json</html>")

    client = _client(None, transport=httpx.MockTransport(handler), cache_file=tmp_path / "c.json")
    result = client.fetch("chat")
    assert result.source == "seed"


def test_an_oversized_body_is_refused(tmp_path):
    big = {"data": [{"id": f"m/{i}"} for i in range(5000)]}
    client = _client(
        big, transport=_transport(big), cache_file=tmp_path / "c.json", max_bytes=2048
    )
    result = client.fetch("chat")
    assert result.source == "seed"
    assert "byte limit" in result.error


def test_item_count_is_capped(tmp_path):
    payload = {"data": [{"id": f"m/{i}", "supported_endpoint_types": ["openai"]}
                        for i in range(5000)]}
    client = _client(payload, transport=_transport(payload),
                     cache_file=tmp_path / "c.json", max_items=50)
    result = client.fetch("chat")
    assert result.count == 50


def test_a_successful_fetch_is_reused_from_cache(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=CATALOG_FIXTURE)

    client = _client(CATALOG_FIXTURE, transport=httpx.MockTransport(handler),
                     cache_file=tmp_path / "c.json", cache_ttl=300)
    first = client.fetch("chat")
    second = client.fetch("chat")
    assert first.ids == second.ids
    assert calls["n"] == 1


def test_refresh_bypasses_the_cache(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=CATALOG_FIXTURE)

    client = _client(CATALOG_FIXTURE, transport=httpx.MockTransport(handler),
                     cache_file=tmp_path / "c.json", cache_ttl=300)
    client.fetch("chat")
    client.fetch("chat", allow_cache=False)
    assert calls["n"] == 2


def test_last_known_good_catalog_survives_a_restart(tmp_path):
    cache = tmp_path / "c.json"
    live = _client(CATALOG_FIXTURE, transport=_transport(CATALOG_FIXTURE), cache_file=cache)
    live.fetch("chat")
    assert cache.exists()

    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    offline = _client(None, transport=httpx.MockTransport(boom), cache_file=cache)
    result = offline.fetch("chat", allow_seed=False)
    assert result.source == "last_known_good"
    assert result.degraded
    assert "acme/text-model" in result.ids


def test_seed_is_not_mixed_into_a_live_result(tmp_path):
    client = _client(CATALOG_FIXTURE, transport=_transport(CATALOG_FIXTURE),
                     cache_file=tmp_path / "c.json")
    result = client.fetch("chat")
    assert result.source == "live"
    assert not (set(result.ids) & set(SEED_IDS))


def test_multimodal_fetch_filters_after_discovery(tmp_path):
    client = _client(CATALOG_FIXTURE, transport=_transport(CATALOG_FIXTURE),
                     cache_file=tmp_path / "c.json")
    result = client.fetch("multimodal", required_modalities=["image"])
    assert result.ids == ["acme/vision-chat"]
    assert result.source == "live"


def test_public_payload_carries_no_credential(tmp_path):
    client = _client(CATALOG_FIXTURE, transport=_transport(CATALOG_FIXTURE),
                     cache_file=tmp_path / "c.json")
    payload = client.fetch("chat").public()
    blob = json.dumps(payload)
    assert "sk-orca" not in blob
    assert "api_key" not in blob
    assert payload["source"] == "live"
    assert payload["count"] == len(payload["data"])


def test_seed_models_are_all_chat_capable():
    assert all(m.supports_capability("chat") for m in SEED_MODELS)
    assert len(SEED_IDS) == 5


def test_seed_declares_no_image_input_so_multimodal_fails_closed():
    assert seed_for("multimodal", required_modalities=["image"]) == []
