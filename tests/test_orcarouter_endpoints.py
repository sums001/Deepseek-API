"""The two OrcaRouter origins and their override policy.

The failure this guards against is subtle and common: deriving the inference
origin from the auth origin (or the reverse) by swapping a hostname or
appending ``/v1``. The auth endpoints live on ``www.orcarouter.ai`` under
``/api/v1/auth``; the relay lives on ``api.orcarouter.ai`` under ``/v1``;
``https://api.orcarouter.ai/v1/auth/keys`` is a 404.
"""

from __future__ import annotations

import pytest

from server.orcarouter.endpoints import (
    DEFAULT_API_BASE,
    DEFAULT_AUTH_BASE,
    OrcaConfigError,
    resolve_endpoints,
    validate_base_url,
)


def test_public_defaults_split_auth_from_inference():
    endpoints = resolve_endpoints({})
    assert endpoints.auth_base == "https://www.orcarouter.ai"
    assert endpoints.api_base == "https://api.orcarouter.ai/v1"
    assert endpoints.auth_base != endpoints.api_base.split("/v1")[0]


def test_exchange_path_is_the_auth_origin_api_v1_auth_keys():
    endpoints = resolve_endpoints({})
    assert endpoints.exchange_url() == "https://www.orcarouter.ai/api/v1/auth/keys"
    # The single most common mistake, asserted explicitly.
    assert endpoints.exchange_url() != "https://api.orcarouter.ai/v1/auth/keys"


def test_authorize_and_inference_paths_are_distinct():
    endpoints = resolve_endpoints({})
    assert endpoints.authorize_url() == "https://www.orcarouter.ai/auth"
    assert endpoints.models_url("chat") == "https://api.orcarouter.ai/v1/models?capability=chat"
    assert endpoints.chat_completions_url() == "https://api.orcarouter.ai/v1/chat/completions"


def test_explicit_overrides_take_precedence():
    endpoints = resolve_endpoints(
        {
            "ORCA_BASE_URL": "https://shared.example.com",
            "ORCA_AUTH_BASE_URL": "https://auth.example.com",
            "ORCA_API_BASE_URL": "https://api.example.com/v1",
        }
    )
    assert endpoints.auth_base == "https://auth.example.com"
    assert endpoints.api_base == "https://api.example.com/v1"


def test_shared_base_is_the_fallback_for_both_origins():
    endpoints = resolve_endpoints({"ORCA_BASE_URL": "https://selfhost.example.com"})
    assert endpoints.auth_base == "https://selfhost.example.com"
    # The inference side gets the /v1 suffix exactly once.
    assert endpoints.api_base == "https://selfhost.example.com/v1"


def test_shared_base_does_not_double_the_v1_suffix():
    endpoints = resolve_endpoints({"ORCA_BASE_URL": "https://selfhost.example.com/v1"})
    assert endpoints.api_base == "https://selfhost.example.com/v1"


def test_one_explicit_override_leaves_the_other_derivation_intact():
    endpoints = resolve_endpoints(
        {"ORCA_AUTH_BASE_URL": "https://auth.example.com",
         "ORCA_BASE_URL": "https://shared.example.com"}
    )
    assert endpoints.auth_base == "https://auth.example.com"
    assert endpoints.api_base == "https://shared.example.com/v1"


def test_remote_origin_must_be_https():
    with pytest.raises(OrcaConfigError):
        validate_base_url("http://evil.example.com", what="ORCA_AUTH_BASE_URL")
    with pytest.raises(OrcaConfigError):
        resolve_endpoints({"ORCA_API_BASE_URL": "http://api.example.com/v1"})


@pytest.mark.parametrize("loopback", ["http://localhost", "http://127.0.0.1:8080", "http://[::1]:9000"])
def test_loopback_may_use_plain_http(loopback):
    assert validate_base_url(loopback, what="ORCA_BASE_URL").startswith("http://")


def test_relative_or_odd_values_are_refused():
    for bad in ("not-a-url", "ftp://example.com", "https://", "https://x.example.com/?q=1"):
        with pytest.raises(OrcaConfigError):
            validate_base_url(bad, what="ORCA_BASE_URL")


def test_an_inference_base_carrying_the_auth_path_is_refused():
    endpoints = resolve_endpoints({"ORCA_API_BASE_URL": "https://api.example.com/v1/auth"})
    with pytest.raises(OrcaConfigError):
        endpoints.assert_distinct_purposes()


def test_defaults_are_stable_constants():
    assert DEFAULT_AUTH_BASE == "https://www.orcarouter.ai"
    assert DEFAULT_API_BASE == "https://api.orcarouter.ai/v1"
