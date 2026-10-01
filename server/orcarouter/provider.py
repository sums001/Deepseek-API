"""The OrcaRouter inference adapter.

A thin, OpenAI-compatible pass-through to ``<api_base>/chat/completions`` and
``<api_base>/models``, using whichever credential the caller hands it. It never
obtains credentials itself - that is the adapters' job in
:mod:`server.orcarouter.credentials` - so nothing in the inference path can
tell an API-key user from a PKCE user.

Two failure modes are treated as first-class:

* **401** - terminal reauthentication. There is no refresh grant for an
  OrcaRouter key, so a rejected credential is marked ``needs_reauth`` for the
  *exact* account and credential generation that made the request, and no
  retry or refresh is attempted. A late failure from an older generation is
  ignored, so it cannot mark a freshly reauthorized credential as broken.
* **429** - surfaced with the retry hint instead of being retried in a loop.

Nothing here logs a credential, and no error message carries one.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List, Optional

import httpx

from .catalog import OrcaCatalogClient, is_compatible
from .credentials import OrcaCredential, redact_key
from .endpoints import OrcaEndpoints, resolve_endpoints

DEFAULT_TIMEOUT = 120.0
STREAM_READ_TIMEOUT = 300.0

# Fields forwarded to the gateway. Everything else the OpenAI SDK sends is
# accepted by the server and dropped here, matching how this project already
# treats unsupported OpenAI parameters.
_FORWARDED_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "user",
)


class OrcaUpstreamError(RuntimeError):
    """An inference or discovery failure, in a shape the HTTP layer can render."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 502,
        err_type: str = "upstream_error",
        needs_reauth: bool = False,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.status = status
        self.err_type = err_type
        self.needs_reauth = needs_reauth
        self.retry_after = retry_after


class OrcaRouterProvider:
    """OpenAI-compatible inference through OrcaRouter."""

    def __init__(
        self,
        credential: Optional[OrcaCredential],
        *,
        endpoints: Optional[OrcaEndpoints] = None,
        store=None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
        catalog: Optional[OrcaCatalogClient] = None,
    ):
        self.credential = credential
        self.endpoints = endpoints or resolve_endpoints()
        self.store = store
        self.timeout = timeout
        self._transport = transport
        self.catalog = catalog or OrcaCatalogClient(self.endpoints, transport=transport)

    # --- identity ---------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.credential and self.credential.api_key)

    @property
    def usable(self) -> bool:
        return bool(self.credential and self.credential.usable)

    def _require_credential(self) -> OrcaCredential:
        if not self.configured:
            raise OrcaUpstreamError(
                "No OrcaRouter credential is configured. Set "
                "ORCAROUTER_API_KEY, or run `python -m server.orcarouter connect` "
                "to authorize a key.",
                status=401,
                err_type="orca_not_configured",
            )
        if self.credential.needs_reauth:
            raise OrcaUpstreamError(
                f"The stored OrcaRouter credential ({redact_key(self.credential.api_key)}) "
                "was revoked. Run `python -m server.orcarouter connect` to authorize "
                "a new one.",
                status=401,
                err_type="orca_needs_reauth",
                needs_reauth=True,
            )
        return self.credential

    # --- model catalog ----------------------------------------------------

    def list_models(
        self,
        capability: str = "chat",
        *,
        required_modalities: Optional[Iterable[str]] = None,
        allow_cache: bool = True,
    ):
        """Discover models for a capability via the catalog client."""
        return self.catalog.fetch(
            capability,
            credential=self.credential,
            required_modalities=required_modalities,
            allow_cache=allow_cache,
        )

    def ensure_model_compatible(self, model_id: str, capability: str = "chat") -> None:
        """Second-layer guard: refuse a model the current entry point cannot use.

        The picker is already bound to a filtered list; this exists so a
        hand-written request cannot route a non-chat model through the chat
        endpoint either.
        """
        result = self.list_models(capability)
        if not is_compatible(model_id, result.models, capability):
            available = ", ".join(result.ids[:12]) or "(none advertised)"
            raise OrcaUpstreamError(
                f"`{model_id}` is not an OrcaRouter {capability} model. "
                f"Currently available: {available}.",
                status=404,
                err_type="model_not_found",
            )

    def ensure_modalities_supported(self, model_id: str, modalities: Iterable[str]) -> None:
        """Refuse an attachment the selected model did not declare."""
        wanted = {str(m).lower() for m in modalities if m}
        if not wanted:
            return
        result = self.list_models("multimodal", required_modalities=wanted)
        if not is_compatible(
            model_id, result.models, "multimodal", required_modalities=wanted
        ):
            raise OrcaUpstreamError(
                f"`{model_id}` does not declare "
                f"{', '.join(sorted(wanted))} input, so this request cannot be "
                "sent to it. Pick a model that supports the attachment.",
                status=400,
                err_type="invalid_request_error",
            )

    # --- inference --------------------------------------------------------

    def build_body(
        self,
        messages: List[Dict],
        model: str,
        *,
        stream: bool = False,
        options: Optional[Dict] = None,
    ) -> Dict:
        """Assemble the request body sent to the gateway.

        DeepSeek-only extras (``conversation_id``, ``thinking``, ``search``)
        are not part of the OpenAI wire format and are dropped here rather than
        forwarded as unknown fields.
        """
        body: Dict = {"model": model, "messages": messages, "stream": bool(stream)}
        for field in _FORWARDED_FIELDS:
            if options and options.get(field) is not None:
                body[field] = options[field]
        return body

    def _headers(self, credential: OrcaCredential) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {credential.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _note_rejection(self, status: int, credential: OrcaCredential) -> None:
        """Record a terminal credential rejection for this exact generation."""
        if status != 401 or self.store is None:
            return
        try:
            self.store.mark_needs_reauth(
                account_id=credential.account_id,
                generation=credential.generation,
            )
        except Exception:
            # Recording state must never mask the original upstream error.
            pass

    def _raise_for_status(self, resp: httpx.Response, credential: OrcaCredential) -> None:
        status = resp.status_code
        if status < 400:
            return
        self._note_rejection(status, credential)
        detail = _safe_detail(resp)

        if status == 401:
            raise OrcaUpstreamError(
                "OrcaRouter rejected the credential (HTTP 401). The key may have "
                "been revoked from the console; authorize a new one with "
                "`python -m server.orcarouter connect`.",
                status=401,
                err_type="orca_needs_reauth",
                needs_reauth=True,
            )
        if status == 403:
            raise OrcaUpstreamError(
                f"OrcaRouter denied this request (HTTP 403){': ' + detail if detail else '.'}",
                status=403,
                err_type="orca_forbidden",
            )
        if status == 429:
            retry_after = _retry_after(resp)
            raise OrcaUpstreamError(
                f"OrcaRouter rate limited this request (HTTP 429)"
                f"{f'; retry in {retry_after:.1f}s' if retry_after else ''}.",
                status=429,
                err_type="rate_limit_error",
                retry_after=retry_after,
            )
        raise OrcaUpstreamError(
            f"OrcaRouter returned HTTP {status}{': ' + detail if detail else '.'}",
            status=status,
            err_type="upstream_error",
        )

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=10.0, read=STREAM_READ_TIMEOUT),
            transport=self._transport,
            follow_redirects=False,
        )

    def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        *,
        stream: bool = False,
        options: Optional[Dict] = None,
    ):
        """Run a chat completion. Returns a dict, or an iterator of SSE lines."""
        credential = self._require_credential()
        body = self.build_body(messages, model, stream=stream, options=options)
        url = self.endpoints.chat_completions_url()
        headers = self._headers(credential)

        if stream:
            return self._stream(url, headers, body, credential)

        try:
            with self._client() as client:
                resp = client.post(url, headers=headers, json=body)
                self._raise_for_status(resp, credential)
                return resp.json()
        except httpx.HTTPError as exc:
            raise OrcaUpstreamError(
                f"Could not reach OrcaRouter ({type(exc).__name__}).",
                status=502,
                err_type="upstream_unreachable",
            ) from None
        except ValueError:
            raise OrcaUpstreamError(
                "OrcaRouter returned a response that was not valid JSON.",
                status=502,
                err_type="upstream_error",
            ) from None

    def _stream(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict,
        credential: OrcaCredential,
    ) -> Iterator[str]:
        """Yield Server-Sent Events from the gateway unchanged.

        The gateway already speaks the OpenAI stream format, so the frames are
        forwarded verbatim - rewriting them would drop finish reasons and usage
        that this project's own bridge does not produce.
        """
        client = self._client()
        try:
            with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    self._raise_for_status(resp, credential)
                for line in resp.iter_lines():
                    yield line
        except httpx.HTTPError as exc:
            raise OrcaUpstreamError(
                f"The OrcaRouter stream failed ({type(exc).__name__}).",
                status=502,
                err_type="upstream_unreachable",
            ) from None
        finally:
            client.close()


def _safe_detail(resp: httpx.Response, limit: int = 240) -> str:
    """A bounded, credential-free extract of an upstream error body."""
    try:
        resp.read()
        payload = resp.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return _scrub(str(message), limit)
            if isinstance(error, str):
                return _scrub(error, limit)
        return _scrub(resp.text, limit)
    except Exception:
        try:
            return _scrub(resp.text, limit)
        except Exception:
            return ""


def _scrub(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    for marker in ("sk-orca-", "sk-"):
        idx = cleaned.find(marker)
        if idx != -1:
            end = idx
            while end < len(cleaned) and not cleaned[end].isspace():
                end += 1
            cleaned = cleaned[:idx] + "<redacted>" + cleaned[end:]
    return cleaned[:limit]


def _retry_after(resp: httpx.Response) -> Optional[float]:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def content_modalities(messages: List[Dict]) -> set:
    """Which non-text modalities a message list actually carries."""
    found = set()
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            # OpenAI names these parts `image_url`, `input_audio`, `input_video`;
            # match on the modality token rather than one exact spelling.
            for modality in ("image", "audio", "video"):
                if modality in part_type:
                    found.add(modality)
                    break
    return found


__all__ = [
    "OrcaRouterProvider",
    "OrcaUpstreamError",
    "content_modalities",
    "json",
]
