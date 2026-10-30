"""Server configuration: the OpenAI-facing model names and what they map to."""

import os

from dotenv import load_dotenv

# Load .env before reading settings below, so they apply however the server is
# started (`python app.py` or `uvicorn server.api:app`).
load_dotenv()


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() not in ("0", "false", "no", "off")


# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# Key the rate limiter on the X-Forwarded-For header instead of the peer IP.
# Only turn this on behind a reverse proxy that sets the header; otherwise any
# caller can send a fake one and dodge the limit.
TRUST_PROXY_HEADERS = _env_flag("TRUST_PROXY_HEADERS", "0")

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish logging
# in)? On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = _env_flag("SERVER_INTERACTIVE_LOGIN", "1")

# Public model ids the server advertises (via /v1/models) and accepts, mapped to
# DeepSeek's `model_type` wire value. This is the MODEL axis ONLY — it picks
# which model answers. DeepThink and web Search are orthogonal tools requested
# per call via the `thinking` / `search` request fields, never encoded in the
# model name.
#
# "vision" is deferred: it only does anything with an image attached, which needs
# ref_file_ids / file-upload plumbing we don't have yet.
MODEL_MAP = {
    "deepseek-chat":   "default",   # Instant — the fast default model
    "deepseek-expert": "expert",    # Expert  — the stronger, slower model
}

DEFAULT_MODEL = "deepseek-chat"


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept (used to 404 unknown models)."""
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value.

    Caller must check `is_known_model` first; this raises KeyError otherwise.
    """
    return MODEL_MAP[name]
