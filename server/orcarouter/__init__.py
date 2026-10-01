"""OrcaRouter provider: API-key and OAuth 2.0 + PKCE access to an
OpenAI-compatible gateway.

OrcaRouter is an OpenAI-compatible AI gateway that routes many providers
behind one endpoint. This package adds it as a first-class provider next to
the built-in DeepSeek bridge:

* :mod:`server.orcarouter.endpoints` - the two public origins (auth vs
  inference) and their override policy.
* :mod:`server.orcarouter.pkce` - verifier/challenge/state primitives.
* :mod:`server.orcarouter.connect` - the OAuth 2.0 + PKCE connect flow.
* :mod:`server.orcarouter.credentials` - the credential interface and its
  two adapters (pasted API key, PKCE login).
* :mod:`server.orcarouter.catalog` - live model discovery with a verified
  cold-start seed and per-capability filtering.
* :mod:`server.orcarouter.provider` - the inference adapter used by the
  server.
"""

from .catalog import (
    CAPABILITIES,
    SEED_MODELS,
    CatalogResult,
    ModelInfo,
    OrcaCatalogClient,
    filter_models,
    parse_catalog,
)
from .connect import (
    OrcaAuthError,
    OrcaOAuthClient,
)
from .credentials import (
    AUTH_METHODS,
    ApiKeyAdapter,
    CredentialStore,
    OrcaCredential,
    PkceAdapter,
    redact_key,
    resolve_credential,
)
from .endpoints import (
    AUTHORIZE_PATH,
    DEFAULT_API_BASE,
    DEFAULT_AUTH_BASE,
    EXCHANGE_PATH,
    OrcaEndpoints,
    resolve_endpoints,
)
from .provider import OrcaRouterProvider, OrcaUpstreamError

__all__ = [
    "AUTHORIZE_PATH",
    "AUTH_METHODS",
    "ApiKeyAdapter",
    "CAPABILITIES",
    "CatalogResult",
    "CredentialStore",
    "DEFAULT_API_BASE",
    "DEFAULT_AUTH_BASE",
    "EXCHANGE_PATH",
    "ModelInfo",
    "OrcaAuthError",
    "OrcaCatalogClient",
    "OrcaCredential",
    "OrcaEndpoints",
    "OrcaOAuthClient",
    "OrcaRouterProvider",
    "OrcaUpstreamError",
    "PkceAdapter",
    "SEED_MODELS",
    "filter_models",
    "parse_catalog",
    "redact_key",
    "resolve_credential",
    "resolve_endpoints",
]
