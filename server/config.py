"""Server configuration: the OpenAI-facing model names and what they map to."""

import os

# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# Optional API key protection. If set, clients must send Authorization: Bearer <key>
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish logging
# in)? On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

# Public model ids the server advertises (via /v1/models) and accepts, mapped to
# DeepSeek's `model_type` wire value.
#
# "default" is Instant (fast model); "expert" is the stronger, slower model.
MODEL_MAP = {
    "deepseek-chat": "default",
    "deepseek-reasoner": "expert",
    "deepseek-expert": "expert",
    "deepseek-coder": "default",
    # OpenAI model aliases to enable seamless drop-in replacement with standard tools
    "gpt-4o": "expert",
    "gpt-4o-mini": "default",
    "gpt-4-turbo": "expert",
    "gpt-4": "expert",
    "gpt-3.5-turbo": "default",
}

DEFAULT_MODEL = "deepseek-chat"


def is_known_model(name: str) -> bool:
    """Check if model name is in MODEL_MAP or starts with standard prefixes."""
    if not name:
        return False
    return name.lower() in MODEL_MAP or name.lower().startswith(("deepseek", "gpt-"))


def should_enable_thinking(model_name: str, explicit_thinking: bool | None = None) -> bool:
    """Determine whether DeepThink reasoning mode should be enabled for this request."""
    if explicit_thinking is not None:
        return explicit_thinking
    name = (model_name or "").lower()
    return "reasoner" in name or "r1" in name


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value.
    Gracefully falls back to 'default' if not specifically mapped.
    """
    if not name:
        return "default"
    return MODEL_MAP.get(name.lower(), "default")
