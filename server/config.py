"""Server configuration: the OpenAI-facing model names and what they map to."""

import os

# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish logging
# in)? On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

# Public model ids the server advertises (via /v1/models) and accepts, mapped to
# DeepSeek's `model_type` wire value. This is the MODEL axis ONLY — it picks
# which model answers. DeepThink and web Search are orthogonal tools requested
# per call via `tool_names` (see deepseek.client.KNOWN_TOOLS), never encoded in
# the model name.
#
# "vision" is deferred: it only does anything with an image attached, which needs
# ref_file_ids / file-upload plumbing we don't have yet.
MODEL_MAP = {
    "deepseek-chat":   "default",   # Instant — the fast default model
    "deepseek-expert": "expert",    # Expert  — the stronger, slower model
}

DEFAULT_MODEL = "deepseek-chat"


# --- OrcaRouter provider ----------------------------------------------------
#
# A second, independent provider alongside the DeepSeek bridge. OrcaRouter is
# an OpenAI-compatible AI gateway that routes many providers behind one
# endpoint; its origins, credential handling, connect flow, and model catalog
# live in `server.orcarouter`. Only the routing names belong here, next to the
# existing model registry.
#
# Clients select it either by the explicit `orcarouter:<model>` prefix or by an
# id advertised by the live OrcaRouter catalog (a `vendor/model` namespace,
# which can never collide with the DeepSeek ids above because those carry no
# slash).
ORCAROUTER_PROVIDER = "orcarouter"
ORCAROUTER_MODEL_PREFIX = "orcarouter:"
ORCAROUTER_DEFAULT_MODEL = "orcarouter/auto"


def is_orcarouter_model(name: str) -> bool:
    """Whether a request named the OrcaRouter provider explicitly."""
    return bool(name) and name.startswith(ORCAROUTER_MODEL_PREFIX)


def strip_orcarouter_prefix(name: str) -> str:
    """`orcarouter:vendor/model` -> `vendor/model` (namespace preserved)."""
    return name[len(ORCAROUTER_MODEL_PREFIX):] if is_orcarouter_model(name) else name


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept (used to 404 unknown models)."""
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value.

    Caller must check `is_known_model` first; this raises KeyError otherwise.
    """
    return MODEL_MAP[name]
