"""Model discovery and capability filtering for OrcaRouter.

The single source of truth for the model list is ``GET <api_base>/models`` on
the configured inference origin (public default
``https://api.orcarouter.ai/v1/models``). Nothing here is allowed to invent a
model list from names: a model's capability comes from the catalog record it
was advertised with - ``supported_endpoint_types`` and
``architecture.input_modalities`` - and never from the shape of its id.

Three properties matter for correctness:

* **Bounded.** A catalog response cannot consume unbounded memory: the body is
  read with a byte ceiling, the item count is capped, and each record must
  match a known shape before it is accepted.
* **Fail closed.** A model that does not *declare* image input is not offered
  to a multimodal picker. Absence of a declaration is not permission.
* **Degrade, never invent.** When live discovery fails the verified cold-start
  seed (or the last known-good catalog) is used, and the result is reported as
  degraded so the UI can say so. A failed catalog never turns into a free-text
  model field.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from .credentials import OrcaCredential
from .endpoints import OrcaEndpoints, resolve_endpoints

# Endpoint types the inference path here can actually speak.
TEXT_ENDPOINT_TYPES = frozenset({"openai", "anthropic", "gemini", "openai-response"})
EMBEDDING_ENDPOINT_TYPES = frozenset({"embeddings", "embedding"})
IMAGE_ENDPOINT_TYPES = frozenset({"image-generation"})
VIDEO_ENDPOINT_TYPES = frozenset({"openai-video"})
RERANK_ENDPOINT_TYPES = frozenset({"jina-rerank"})
NON_TEXT_ENDPOINT_TYPES = (
    EMBEDDING_ENDPOINT_TYPES
    | IMAGE_ENDPOINT_TYPES
    | VIDEO_ENDPOINT_TYPES
    | RERANK_ENDPOINT_TYPES
)

# capability -> catalog query + accepted endpoint types
#
# ``forbidden_endpoint_types`` is the fail-closed half of the text rule: a
# record that advertises an image/video/rerank/embedding route is specialised
# for something other than text chat, so it is kept out of the chat picker even
# if it also lists ``openai``.
CAPABILITIES: Dict[str, dict] = {
    "chat": {
        "query": "chat",
        "endpoint_types": TEXT_ENDPOINT_TYPES,
        "forbidden_endpoint_types": NON_TEXT_ENDPOINT_TYPES,
        "requires_declared_input": False,
    },
    "multimodal": {
        "query": "chat",
        "endpoint_types": TEXT_ENDPOINT_TYPES,
        "forbidden_endpoint_types": NON_TEXT_ENDPOINT_TYPES,
        "requires_declared_input": True,
    },
    "embedding": {
        "query": "embedding",
        "endpoint_types": EMBEDDING_ENDPOINT_TYPES,
        "forbidden_endpoint_types": frozenset(),
        "requires_declared_input": False,
    },
    "image": {
        "query": "image",
        "endpoint_types": IMAGE_ENDPOINT_TYPES,
        "forbidden_endpoint_types": frozenset(),
        "requires_declared_input": False,
    },
    "video": {
        "query": None,
        "endpoint_types": VIDEO_ENDPOINT_TYPES,
        "forbidden_endpoint_types": frozenset(),
        "requires_declared_input": False,
    },
    "rerank": {
        "query": None,
        "endpoint_types": RERANK_ENDPOINT_TYPES,
        "forbidden_endpoint_types": frozenset(),
        "requires_declared_input": False,
    },
}

MAX_BYTES = 512 * 1024
MAX_ITEMS = 2000
DEFAULT_TIMEOUT = 10.0

DEFAULT_CATALOG_FILE = (
    Path(__file__).resolve().parent.parent.parent / "session" / "orcarouter_models.json"
)

# Sentinel: omitting the argument uses the project's default cache file, while
# passing ``None`` explicitly disables persistence (used by tests).
USE_DEFAULT_CACHE = object()


@dataclass(frozen=True)
class ModelInfo:
    """One advertised model, as the catalog described it."""

    id: str
    name: str = ""
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    input_modalities: frozenset = field(default_factory=frozenset)
    output_modalities: frozenset = field(default_factory=frozenset)
    supported_endpoint_types: frozenset = field(default_factory=frozenset)
    reasoning_efforts: Tuple[str, ...] = ()
    owned_by: str = ""
    description: str = ""
    source: str = "live"
    observed_live: bool = True

    @property
    def namespaced_id(self) -> str:
        """The id exactly as advertised - ``vendor/model`` is preserved."""
        return self.id

    def supports_capability(
        self,
        capability: str,
        *,
        required_modalities: Optional[Iterable[str]] = None,
    ) -> bool:
        """Whether this model may be offered for ``capability``.

        Endpoint types must intersect the capability's accepted set, and - for
        a multimodal picker - every modality the entry point actually uploads
        must be explicitly declared. An undeclared modality fails closed.
        """
        spec = CAPABILITIES.get(capability)
        if spec is None:
            raise ValueError(f"unknown capability: {capability!r}")

        types = set(self.supported_endpoint_types)
        if not types:
            return False
        if not (types & set(spec["endpoint_types"])):
            return False
        forbidden = set(spec.get("forbidden_endpoint_types") or ())
        if forbidden and (types & forbidden):
            return False
        if spec["requires_declared_input"]:
            wanted = {str(m).lower() for m in (required_modalities or ()) if m}
            if not wanted:
                return False
            declared = {str(m).lower() for m in self.input_modalities}
            if not wanted <= declared:
                return False
        return True

    def to_public_dict(self) -> Dict:
        """The minimal, credential-free shape handed to a browser or API client."""
        return {
            "id": self.id,
            "name": self.name or self.id,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "input_modalities": sorted(self.input_modalities),
            "output_modalities": sorted(self.output_modalities),
            "supported_endpoint_types": sorted(self.supported_endpoint_types),
            "reasoning_efforts": list(self.reasoning_efforts),
            "owned_by": self.owned_by,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelInfo":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or ""),
            context_length=_int_or_none(data.get("context_length")),
            max_completion_tokens=_int_or_none(data.get("max_completion_tokens")),
            input_modalities=frozenset(data.get("input_modalities") or ()),
            output_modalities=frozenset(data.get("output_modalities") or ()),
            supported_endpoint_types=frozenset(data.get("supported_endpoint_types") or ()),
            reasoning_efforts=tuple(data.get("reasoning_efforts") or ()),
            owned_by=str(data.get("owned_by") or ""),
            description=str(data.get("description") or ""),
            source=str(data.get("source") or "live"),
            observed_live=bool(data.get("observed_live", True)),
        )


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class CatalogResult:
    """A filtered model list plus where it came from."""

    models: List[ModelInfo]
    source: str  # "live" | "seed" | "last_known_good"
    capability: str
    degraded: bool = False
    error: str = ""
    fetched_at: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.models)

    @property
    def ids(self) -> List[str]:
        return [m.id for m in self.models]

    def public(self) -> Dict:
        return {
            "object": "list",
            "source": self.source,
            "capability": self.capability,
            "degraded": self.degraded,
            "error": self.error,
            "fetched_at": self.fetched_at,
            "count": self.count,
            "data": [m.to_public_dict() for m in self.models],
        }


# --- the verified cold-start seed -----------------------------------------
#
# Small, bounded, and carrying real metadata. It exists so a fresh install - or
# an install whose catalog call is failing - still has something to select.
# It is never mixed into a successful live result.
#
# Sources for the metadata below:
#   * ``deepseek/deepseek-v4-pro`` and ``orcarouter/auto``: observed live in
#     this project's own ``GET /v1/models`` on 2026-10-01, and their
#     ``context_length`` / ``supported_endpoint_types`` are taken from that
#     response verbatim.
#   * The remaining ids are the documented cross-workspace cold-start seed.
#     Only metadata that can be attributed is attached; nothing is inferred
#     from a model name, so modalities are declared ``text`` only, which fails
#     closed for a multimodal picker.
#   * ``openai/gpt-5.5`` carries the verified reasoning-effort ladder.

SEED_MODELS: Tuple[ModelInfo, ...] = (
    ModelInfo(
        id="openai/gpt-5.5",
        name="OpenAI: GPT-5.5",
        input_modalities=frozenset({"text"}),
        supported_endpoint_types=frozenset({"openai", "openai-response"}),
        reasoning_efforts=("low", "medium", "high", "xhigh"),
        owned_by="OpenAI",
        source="seed",
        observed_live=False,
    ),
    ModelInfo(
        id="anthropic/claude-opus-4.8",
        name="Anthropic: Claude Opus 4.8",
        input_modalities=frozenset({"text"}),
        supported_endpoint_types=frozenset({"openai", "anthropic"}),
        owned_by="Anthropic",
        source="seed",
        observed_live=False,
    ),
    ModelInfo(
        id="google/gemini-3.5-flash",
        name="Google: Gemini 3.5 Flash",
        input_modalities=frozenset({"text"}),
        supported_endpoint_types=frozenset({"openai", "gemini"}),
        owned_by="Google",
        source="seed",
        observed_live=False,
    ),
    ModelInfo(
        id="deepseek/deepseek-v4-pro",
        name="DeepSeek: DeepSeek V4 Pro",
        context_length=1048576,
        max_completion_tokens=384000,
        input_modalities=frozenset({"text"}),
        supported_endpoint_types=frozenset({"openai", "openai-response"}),
        owned_by="DeepSeek",
        source="seed",
        observed_live=True,
    ),
    ModelInfo(
        id="orcarouter/auto",
        name="OrcaRouter: Auto",
        input_modalities=frozenset({"text"}),
        supported_endpoint_types=frozenset(
            {"openai", "openai-response", "anthropic", "gemini"}
        ),
        owned_by="orcarouter",
        source="seed",
        observed_live=True,
    ),
)

SEED_IDS: Tuple[str, ...] = tuple(m.id for m in SEED_MODELS)


# --- parsing --------------------------------------------------------------


def parse_catalog(payload, capability: str = "chat") -> List[ModelInfo]:
    """Validate and convert a ``/v1/models`` body.

    Accepts only an object with a list of records, each with a non-empty
    string ``id``. Anything else in a record is optional; a record that cannot
    be understood is skipped rather than guessed at.
    """
    if not isinstance(payload, dict):
        raise ValueError("catalog payload is not an object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("catalog payload has no `data` list")

    models: List[ModelInfo] = []
    seen = set()
    for raw in data[:MAX_ITEMS]:
        if not isinstance(raw, dict):
            continue
        model_id = raw.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        if model_id in seen:
            continue
        seen.add(model_id)

        architecture = raw.get("architecture")
        architecture = architecture if isinstance(architecture, dict) else {}
        modalities = architecture.get("input_modalities")
        out_modalities = architecture.get("output_modalities")
        endpoint_types = raw.get("supported_endpoint_types")

        models.append(
            ModelInfo(
                id=model_id,
                name=str(raw.get("name") or ""),
                context_length=_int_or_none(raw.get("context_length")),
                max_completion_tokens=_int_or_none(raw.get("max_completion_tokens")),
                input_modalities=frozenset(
                    str(m).lower() for m in (modalities or ()) if isinstance(m, str)
                ),
                output_modalities=frozenset(
                    str(m).lower() for m in (out_modalities or ()) if isinstance(m, str)
                ),
                supported_endpoint_types=frozenset(
                    str(t).lower() for t in (endpoint_types or ()) if isinstance(t, str)
                ),
                reasoning_efforts=tuple(
                    str(e) for e in (raw.get("reasoning_efforts") or ()) if isinstance(e, str)
                ),
                owned_by=str(raw.get("owned_by") or ""),
                description=str(raw.get("description") or ""),
                source="live",
                observed_live=True,
            )
        )
    return models


def filter_models(
    models: Sequence[ModelInfo],
    capability: str = "chat",
    *,
    required_modalities: Optional[Iterable[str]] = None,
) -> List[ModelInfo]:
    """Keep only the models compatible with ``capability``.

    This is the function the model picker is bound to; the guard that refuses
    an incompatible attachment at send time is a second line of defence, not a
    substitute for filtering the options the user can pick from.
    """
    return [
        m
        for m in models
        if m.supports_capability(capability, required_modalities=required_modalities)
    ]


def seed_for(
    capability: str = "chat",
    *,
    required_modalities: Optional[Iterable[str]] = None,
) -> List[ModelInfo]:
    """The verified seed, filtered for the requested capability."""
    return filter_models(SEED_MODELS, capability, required_modalities=required_modalities)


def is_compatible(
    model_id: str,
    models: Sequence[ModelInfo],
    capability: str = "chat",
    *,
    required_modalities: Optional[Iterable[str]] = None,
) -> bool:
    """Whether a previously selected model id is still offerable.

    Used before restoring a persisted selection: a remembered id that is no
    longer in the compatible list must be cleared, not silently kept.
    """
    if not model_id:
        return False
    return any(
        m.id == model_id
        for m in filter_models(
            models, capability, required_modalities=required_modalities
        )
    )


# --- discovery client -----------------------------------------------------


class OrcaCatalogClient:
    """Fetches ``/v1/models`` with bounds, caching, and a seed fallback."""

    def __init__(
        self,
        endpoints: Optional[OrcaEndpoints] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = MAX_BYTES,
        max_items: int = MAX_ITEMS,
        transport: Optional[httpx.BaseTransport] = None,
        cache_file: object = USE_DEFAULT_CACHE,
        cache_ttl: float = 300.0,
    ):
        self.endpoints = endpoints or resolve_endpoints()
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_items = max_items
        self._transport = transport
        self.cache_file = (
            DEFAULT_CATALOG_FILE if cache_file is USE_DEFAULT_CACHE else cache_file
        )
        self.cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._lkg: Dict[str, CatalogResult] = {}

    # --- fetching ---------------------------------------------------------

    def _read_bounded(self, resp: httpx.Response) -> bytes:
        """Read at most ``max_bytes`` of the body, then stop."""
        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > self.max_bytes:
                raise ValueError("catalog response exceeded the byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def fetch(
        self,
        capability: str = "chat",
        *,
        credential: Optional[OrcaCredential] = None,
        required_modalities: Optional[Iterable[str]] = None,
        allow_cache: bool = True,
        allow_seed: bool = True,
    ) -> CatalogResult:
        """Return the model list for ``capability``.

        Live success is authoritative. On failure the last known-good list for
        this capability is used, then the verified seed; both are marked
        degraded so the caller can say so.
        """
        spec = CAPABILITIES.get(capability)
        if spec is None:
            raise ValueError(f"unknown capability: {capability!r}")

        # A multimodal picker starts from the chat-capable set and narrows per
        # attachment; filtering by capability alone would drop everything,
        # because "multimodal" deliberately requires a declared modality.
        base_capability = "chat" if capability == "multimodal" else capability

        if allow_cache:
            cached = self._lkg.get(capability)
            if cached and (time.time() - cached.fetched_at) < self.cache_ttl:
                return self._with_modalities(cached, capability, required_modalities)

        url = self.endpoints.models_url(spec["query"])
        headers = {"Accept": "application/json"}
        # The user's own key, when present, narrows the catalog to what their
        # workspace can actually call. It stays in this process.
        if credential and credential.api_key:
            headers["Authorization"] = f"Bearer {credential.api_key}"

        error = ""
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 200:
                        payload = json.loads(self._read_bounded(resp).decode("utf-8"))
                        parsed = parse_catalog(payload, capability)[: self.max_items]
                        result = CatalogResult(
                            models=filter_models(parsed, base_capability),
                            source="live",
                            capability=capability,
                        )
                        with self._lock:
                            self._lkg[capability] = result
                            self._lkg.setdefault(base_capability, result)
                        self._persist(capability, result)
                        return self._with_modalities(result, capability, required_modalities)
                    if resp.status_code in (401, 403):
                        error = (
                            "OrcaRouter rejected the model-list request "
                            f"(HTTP {resp.status_code}); the credential may have "
                            "been revoked."
                        )
                    else:
                        error = f"model list request failed with HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            error = f"could not reach the OrcaRouter model list ({type(exc).__name__})"
        except ValueError as exc:
            # RangeError (the byte ceiling) carries a useful message; keep it.
            detail = str(exc) or type(exc).__name__
            error = f"model list response was unusable ({detail})"
        except json.JSONDecodeError as exc:
            error = f"model list response was unusable ({type(exc).__name__})"

        fallback = self._lkg.get(capability)
        if fallback is None:
            fallback = self._load_persisted(capability)
        if fallback is not None:
            return self._with_modalities(
                CatalogResult(
                    models=fallback.models,
                    source="last_known_good",
                    capability=capability,
                    degraded=True,
                    error=error,
                ),
                capability,
                required_modalities,
            )

        if allow_seed:
            return CatalogResult(
                models=seed_for(capability, required_modalities=required_modalities),
                source="seed",
                capability=capability,
                degraded=True,
                error=error,
            )

        return CatalogResult(
            models=[], source="live", capability=capability, degraded=True, error=error
        )

    @staticmethod
    def _with_modalities(
        result: CatalogResult,
        capability: str,
        required_modalities: Optional[Iterable[str]],
    ) -> CatalogResult:
        if capability != "multimodal" or not required_modalities:
            return result
        return CatalogResult(
            models=filter_models(
                result.models, capability, required_modalities=required_modalities
            ),
            source=result.source,
            capability=capability,
            degraded=result.degraded,
            error=result.error,
            fetched_at=result.fetched_at,
        )

    # --- persistence ------------------------------------------------------

    def _persist(self, capability: str, result: CatalogResult) -> None:
        """Remember the last live catalog so a later outage still has models."""
        if self.cache_file is None:
            return
        try:
            existing = {}
            if self.cache_file.exists():
                existing = json.loads(self.cache_file.read_text(encoding="utf-8"))
            existing[capability] = {
                "fetched_at": result.fetched_at,
                "models": [m.to_public_dict() for m in result.models],
            }
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            # A cache write must never fail a request.
            pass

    def _load_persisted(self, capability: str) -> Optional[CatalogResult]:
        if self.cache_file is None or not self.cache_file.exists():
            return None
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            entry = data.get(capability)
            if not isinstance(entry, dict):
                return None
            models = []
            for raw in (entry.get("models") or [])[: self.max_items]:
                if not isinstance(raw, dict) or not raw.get("id"):
                    continue
                record = dict(raw)
                record.setdefault("source", "last_known_good")
                record["observed_live"] = True
                models.append(ModelInfo.from_dict(record))
            if not models:
                return None
            return CatalogResult(
                models=models,
                source="last_known_good",
                capability=capability,
                degraded=True,
                fetched_at=float(entry.get("fetched_at") or 0.0),
            )
        except Exception:
            return None

    def invalidate(self, capability: Optional[str] = None) -> None:
        with self._lock:
            if capability is None:
                self._lkg.clear()
            else:
                self._lkg.pop(capability, None)


def env_catalog_file() -> Optional[Path]:
    """Optional override of the last-known-good catalog location."""
    value = os.environ.get("ORCAROUTER_CATALOG_FILE")
    return Path(value) if value else None
