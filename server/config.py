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

# Admin token to secure the /v1/session/import endpoint (leave empty to disable endpoint)
PROXY_ADMIN_TOKEN = os.getenv("PROXY_ADMIN_TOKEN", "").strip() or None

# API key to secure user endpoints /v1/chat/completions and /v1/models (leave empty to keep them public)
DEEPSEEK_PROXY_API_KEY = os.getenv("DEEPSEEK_PROXY_API_KEY", "").strip() or None

# Discord, Slack, or generic webhook URL to send notifications when session expires critically
SESSION_ALERT_WEBHOOK_URL = os.getenv("SESSION_ALERT_WEBHOOK_URL", "").strip() or None


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


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept.
    We return True for everything to allow automatic fallback to DEFAULT_MODEL,
    preventing 404 errors in strict clients like LangChain/n8n.
    """
    return True


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value."""
    base_model = name if name in MODEL_MAP else DEFAULT_MODEL
    return MODEL_MAP[base_model]


def parse_virtual_model(name: str) -> tuple[str, bool, bool, str | None]:
    """Parse virtual suffixes like -thinking and -search from the model name.
    Also parses optional user ID suffix like ':user_id'.
    
    If the base model is unknown, it falls back to DEFAULT_MODEL.
    
    Returns:
        tuple: (base_model_name, thinking_enabled, search_enabled, user_id)
    """
    user_id = None
    if ":" in name:
        parts = name.split(":", 1)
        name = parts[0]
        user_id = parts[1].strip() or None

    name_lower = name.lower().strip()
    thinking = False
    search = False
    
    if "-thinking" in name_lower:
        thinking = True
        name_lower = name_lower.replace("-thinking", "")
    if "-search" in name_lower:
        search = True
        name_lower = name_lower.replace("-search", "")
        
    base_model = name_lower if name_lower in MODEL_MAP else DEFAULT_MODEL
    return base_model, thinking, search, user_id



def get_all_model_names() -> list[str]:
    """Return all available models including virtual combinations for client dropdowns."""
    names = []
    for base in MODEL_MAP.keys():
        names.append(base)
        names.append(f"{base}-thinking")
        names.append(f"{base}-search")
        names.append(f"{base}-thinking-search")
    return names
