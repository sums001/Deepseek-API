"""Live checks against the real OrcaRouter service.

This is the only file that may touch the network or a real credential. It runs
the **implemented provider code path** - not a bare curl - so a green result
means the integration itself works:

* ``GET https://api.orcarouter.ai/v1/models`` through
  :class:`~server.orcarouter.catalog.OrcaCatalogClient`;
* one real ``POST /v1/chat/completions`` through
  :class:`~server.orcarouter.provider.OrcaRouterProvider`;
* the capability filters applied to the *real* catalog.

The credential comes from ``ORCAROUTER_API_KEY`` and is never printed. Without
it every test here skips, so the rest of the suite stays offline.
"""

from __future__ import annotations

import os

import pytest

from conftest import REAL_API_KEY as _CAPTURED_KEY

# The real key. conftest captures it from the environment and then removes it so
# that no offline test can read it; this file is the one place that puts it back.
LIVE_API_KEY = _CAPTURED_KEY or os.environ.get("ORCAROUTER_API_KEY", "")

from server.orcarouter.catalog import OrcaCatalogClient, filter_models
from server.orcarouter.credentials import OrcaCredential, redact_key
from server.orcarouter.endpoints import resolve_endpoints
from server.orcarouter.provider import OrcaRouterProvider

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_provider():
    if not LIVE_API_KEY:
        pytest.skip("ORCAROUTER_API_KEY is not set")
    endpoints = resolve_endpoints({})
    credential = OrcaCredential(
        api_key=LIVE_API_KEY, method="api_key", source="ORCAROUTER_API_KEY"
    )
    return OrcaRouterProvider(
        credential,
        endpoints=endpoints,
        catalog=OrcaCatalogClient(endpoints, cache_file=None),
    )


# How many catalog entries the live probe may try before giving up. The catalog
# lists everything the gateway offers, while a key is often scoped to a handful;
# there is no endpoint that reports the scope, so the probe walks a bounded
# prefix rather than the whole list.
MAX_SCOPE_PROBE = 12


def _callable(catalog, provider) -> list:
    """Which advertised models this workspace's key may actually call, bounded.

    Discovering the subset is itself a real check - it proves the catalog is
    being read live - and it is what the provider code path is then exercised
    against.
    """
    usable = []
    for model_id in catalog.ids[:MAX_SCOPE_PROBE]:
        try:
            provider.chat_completion(
                [{"role": "user", "content": "ok"}], model_id, options={"max_tokens": 4}
            )
            usable.append(model_id)
        except Exception as exc:
            if getattr(exc, "status", None) in (403, 404):
                continue          # out of scope, or not callable by this key
            raise
    return usable


@pytest.fixture(scope="module")
def live_usable_model(live_provider, live_catalog):
    usable = _callable(live_catalog, live_provider)
    if not usable:
        pytest.skip(
            f"none of the first {MAX_SCOPE_PROBE} advertised chat models is "
            "callable by this API key"
        )
    return _pick_live_model(usable)


# Preference order for a live completion: a stable general-purpose chat model,
# so the check is about the integration rather than about one model's quirks.
PREFERRED_LIVE_MODELS = (
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.8",
)


def _pick_live_model(usable):
    for preferred in PREFERRED_LIVE_MODELS:
        if preferred in usable:
            return preferred
    return usable[0]


@pytest.fixture(scope="module")
def live_catalog(live_provider):
    return live_provider.list_models("chat", allow_cache=False)


# --- the origins actually used --------------------------------------------


def test_live_origins_are_the_documented_ones():
    endpoints = resolve_endpoints({})
    assert endpoints.auth_base == "https://www.orcarouter.ai"
    assert endpoints.api_base == "https://api.orcarouter.ai/v1"
    assert endpoints.exchange_url() == "https://www.orcarouter.ai/api/v1/auth/keys"
    # The inference path must not be the auth path with a hostname swapped.
    assert "/auth" not in endpoints.models_url("chat")


# --- model discovery ------------------------------------------------------


def test_live_model_list_is_reachable_and_authoritative(live_catalog):
    assert live_catalog.source == "live", live_catalog.error
    assert not live_catalog.degraded
    assert live_catalog.count > 0
    # Vendor-namespaced ids are preserved exactly as advertised.
    assert all("/" in model_id for model_id in live_catalog.ids)


def test_live_model_records_carry_capability_metadata(live_catalog):
    with_metadata = [m for m in live_catalog.models if m.supported_endpoint_types]
    assert with_metadata, "the catalog advertised no endpoint types at all"
    for model in with_metadata:
        assert model.id
        assert isinstance(model.supported_endpoint_types, frozenset)


def test_live_chat_filter_excludes_non_text_models(live_catalog):
    """Whatever the workspace advertises, the chat list is the compatible subset."""
    for model in live_catalog.models:
        types = set(model.supported_endpoint_types)
        assert types & {"openai", "anthropic", "gemini", "openai-response"}
        assert not (types & {"embeddings", "image-generation", "openai-video", "jina-rerank"})


def test_live_multimodal_filter_fails_closed(live_catalog):
    """Only models that *declare* image input may appear for an image request."""
    multimodal = filter_models(live_catalog.models, "multimodal", required_modalities=["image"])
    for model in multimodal:
        assert "image" in model.input_modalities
    # The seeded, un-declared models are never offered here.
    assert "orcarouter/auto" not in [m.id for m in multimodal]


# --- a real inference call through the provider ---------------------------


def test_live_chat_completion_through_the_provider(live_provider, live_usable_model):
    """A real completion, through the implemented provider, on the real base URL.

    The budget is deliberately not tiny: the flagship models here are reasoning
    models, and a very small ``max_tokens`` can be spent entirely on the
    reasoning channel, leaving ``content`` empty. That is a property of the
    model, not of the integration, so the assertion allows for it while still
    requiring a real answer.
    """
    result = live_provider.chat_completion(
        [{"role": "user", "content": "Reply with the single word: ready"}],
        live_usable_model,
        options={"max_tokens": 256},
    )
    assert result.get("choices"), result
    assert result.get("model")
    message = result["choices"][0].get("message") or {}
    produced = (message.get("content") or "") + (message.get("reasoning_content") or "")
    assert produced.strip(), result


def test_live_streaming_through_the_provider(live_provider, live_usable_model):
    frames = list(
        live_provider.chat_completion(
            [{"role": "user", "content": "Count to three."}],
            live_usable_model,
            stream=True,
            options={"max_tokens": 32},
        )
    )
    assert frames, "the stream produced no frames"
    assert any(frame.startswith("data:") for frame in frames)
    assert any("[DONE]" in frame for frame in frames)


def test_live_out_of_scope_model_is_a_terminal_403(live_provider, live_catalog):
    """A model this key may not call fails once, terminally, naming the reason.

    The gateway still lists it - the catalog is workspace-wide - so this also
    documents that the picker can legitimately offer a model the key is not
    scoped for, and that the failure is reported rather than retried.
    """
    from server.orcarouter.provider import OrcaUpstreamError

    denied = None
    for model_id in live_catalog.ids[:MAX_SCOPE_PROBE]:
        try:
            live_provider.chat_completion(
                [{"role": "user", "content": "ok"}], model_id, options={"max_tokens": 4}
            )
        except OrcaUpstreamError as exc:
            if exc.status == 403:
                denied = exc
                break
            raise
    if denied is None:
        pytest.skip(
            f"none of the first {MAX_SCOPE_PROBE} advertised chat models is out of "
            "scope for this API key"
        )
    assert "does not have access" in str(denied)
    assert "refresh" not in str(denied).lower()


def test_live_chat_completion_against_a_deliberately_bad_model(live_provider):
    """A model the workspace does not advertise is refused before it is sent."""
    from server.orcarouter.provider import OrcaUpstreamError

    with pytest.raises(OrcaUpstreamError) as excinfo:
        live_provider.ensure_model_compatible("nonexistent/vendor-model", "chat")
    assert excinfo.value.status == 404


def test_live_unknown_model_is_rejected_by_the_gateway():
    if not LIVE_API_KEY:
        pytest.skip("ORCAROUTER_API_KEY is not set")
    from server.orcarouter.provider import OrcaUpstreamError

    endpoints = resolve_endpoints({})
    provider = OrcaRouterProvider(
        OrcaCredential(api_key=LIVE_API_KEY, method="api_key"),
        endpoints=endpoints,
        catalog=OrcaCatalogClient(endpoints, cache_file=None),
    )
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.chat_completion(
            [{"role": "user", "content": "hi"}], "orcarouter/does-not-exist",
            options={"max_tokens": 4},
        )
    assert excinfo.value.status in (400, 403, 404)


def test_live_revoked_style_401_is_terminal_not_a_retry():
    """A deliberately invalid key must fail once and stop."""
    if not LIVE_API_KEY:
        pytest.skip("ORCAROUTER_API_KEY is not set")
    from server.orcarouter.provider import OrcaUpstreamError

    endpoints = resolve_endpoints({})
    bogus = OrcaCredential(api_key="sk-orca-invalid-0000000000000000", method="api_key")
    provider = OrcaRouterProvider(
        bogus, endpoints=endpoints, catalog=OrcaCatalogClient(endpoints, cache_file=None)
    )
    with pytest.raises(OrcaUpstreamError) as excinfo:
        provider.chat_completion(
            [{"role": "user", "content": "hi"}], "deepseek/deepseek-v4-pro",
            options={"max_tokens": 4},
        )
    assert excinfo.value.status == 401
    assert excinfo.value.needs_reauth is True
    # A rejected durable key is never refreshed.
    assert "refresh" not in str(excinfo.value).lower()
    assert "sk-orca-invalid" not in str(excinfo.value)


# --- the credential is never exposed --------------------------------------


def test_live_key_is_only_ever_shown_redacted(live_provider, live_catalog, capsys):
    assert LIVE_API_KEY not in repr(live_provider.credential)
    assert LIVE_API_KEY not in redact_key(LIVE_API_KEY)
    captured = capsys.readouterr()
    assert LIVE_API_KEY not in captured.out
    assert LIVE_API_KEY not in captured.err
