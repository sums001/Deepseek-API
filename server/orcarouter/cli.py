"""Command-line access to the OrcaRouter credential and model catalog.

    python -m server.orcarouter status                 # what is configured
    python -m server.orcarouter connect                # OAuth 2.0 + PKCE login
    python -m server.orcarouter connect --flow oob     # paste-the-code fallback
    python -m server.orcarouter key                    # paste an existing key
    python -m server.orcarouter logout                 # forget the credential
    python -m server.orcarouter models                 # live model catalog

Both ways in are first-class and independently usable: ``key`` for someone who
already has an ``sk-orca-...`` value, ``connect`` for someone who would rather
sign in. They produce the same kind of credential and never interfere with the
DeepSeek session.

No command prints a key. ``status`` shows a redacted form, and the key prompt
reads from the terminal rather than taking the secret as an argument, so it
does not land in shell history.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .catalog import CAPABILITIES, OrcaCatalogClient
from .connect import (
    OrcaAuthCancelled,
    OrcaAuthDenied,
    OrcaAuthError,
    OrcaOAuthClient,
)
from .credentials import (
    API_KEY_METHOD,
    PKCE_METHOD,
    ApiKeyAdapter,
    CredentialStore,
    PkceAdapter,
    redact_key,
    resolve_credential,
)
from .endpoints import OrcaConfigError, resolve_endpoints

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CANCELLED = 2


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    for line in payload.get("lines", []):
        print(line)


def cmd_status(args) -> int:
    """Report which authentication methods are available, without printing keys."""
    store = CredentialStore(args.credential_file) if args.credential_file else CredentialStore()
    try:
        endpoints = resolve_endpoints()
    except OrcaConfigError as exc:
        _emit(
            {"ok": False, "error": str(exc), "lines": [f"configuration error: {exc}"]},
            args.json,
        )
        return EXIT_FAILURE

    stored = store.load()
    from_env = resolve_credential(store)
    env_key = from_env is not None and from_env.source == "ORCAROUTER_API_KEY"
    stored_pkce = bool(stored and stored.method == PKCE_METHOD)
    stored_key = bool(stored and stored.method == API_KEY_METHOD)

    lines = [
        "OrcaRouter provider",
        f"  auth origin      {endpoints.auth_base}",
        f"  inference base   {endpoints.api_base}",
        "",
        "Authentication methods",
        f"  API key   {'available' if (env_key or stored_key) else 'not configured'}"
        f"{' (ORCAROUTER_API_KEY)' if env_key else ''}"
        f"{' (saved key)' if stored_key else ''}",
        f"  PKCE      {'available' if stored_pkce else 'not configured'}",
        "  CLI       `python -m server.orcarouter key` or `... connect`",
    ]
    if from_env is not None:
        lines.append("")
        lines.append(f"Active credential  {redact_key(from_env.api_key)}")
        lines.append(f"  source           {from_env.source or from_env.method}")
        lines.append(f"  scope            {from_env.scope}")
        if from_env.user_id:
            lines.append(f"  user             {from_env.user_id}")
        lines.append(
            f"  status           {from_env.status}"
            + ("  (a new login is required)" if from_env.needs_reauth else "")
        )
    else:
        lines.append("")
        lines.append("Active credential  none")
        lines.append("  Run `python -m server.orcarouter key` to paste a key, or")
        lines.append("  `python -m server.orcarouter connect` to sign in.")
    if store.corrupt:
        lines.append("")
        lines.append(
            f"warning: {store.path} could not be read; run `logout` to reset it."
        )

    _emit(
        {
            "ok": True,
            "auth_base": endpoints.auth_base,
            "api_base": endpoints.api_base,
            "api_key_configured": bool(env_key or stored_key),
            "pkce_configured": stored_pkce,
            "active": (
                {
                    "masked": redact_key(from_env.api_key),
                    "method": from_env.method,
                    "source": from_env.source,
                    "scope": from_env.scope,
                    "user_id": from_env.user_id,
                    "status": from_env.status,
                }
                if from_env
                else None
            ),
            "lines": lines,
        },
        args.json,
    )
    return EXIT_OK


def _read_secret(prompt: str) -> str:
    """Read a secret without echoing it and without putting it in argv."""
    try:
        import getpass

        return getpass.getpass(prompt).strip()
    except Exception:
        return sys.stdin.readline().strip()


def cmd_key(args) -> int:
    """Save an existing ``sk-orca-...`` key (adapter 1)."""
    store = CredentialStore(args.credential_file) if args.credential_file else CredentialStore()

    if args.clear:
        removed = store.clear()
        _emit(
            {
                "ok": True,
                "cleared": removed,
                "lines": [
                    "Removed the stored OrcaRouter credential."
                    if removed
                    else "No stored OrcaRouter credential to remove."
                ],
            },
            args.json,
        )
        return EXIT_OK

    value = args.api_key or _read_secret("OrcaRouter API key (input hidden): ")
    result = ApiKeyAdapter(store=store).acquire(value)
    if not result.ok:
        _emit(
            {
                "ok": False,
                "error": result.message,
                "lines": [
                    f"Not saved: {result.message}",
                    "Create a key at https://www.orcarouter.ai and try again.",
                ],
            },
            args.json,
        )
        return EXIT_FAILURE

    credential = result.credential
    _emit(
        {
            "ok": True,
            "masked": credential.masked,
            "method": credential.method,
            "lines": [
                f"Saved {credential.masked} to {store.path}",
                "The value itself is never printed or logged.",
            ],
        },
        args.json,
    )
    return EXIT_OK


def cmd_logout(args) -> int:
    store = CredentialStore(args.credential_file) if args.credential_file else CredentialStore()
    removed = store.clear()
    _emit(
        {
            "ok": True,
            "cleared": removed,
            "lines": [
                "Removed the stored OrcaRouter credential."
                if removed
                else "No stored OrcaRouter credential to remove.",
                "Note: ORCAROUTER_API_KEY, if exported, still takes precedence.",
            ],
        },
        args.json,
    )
    return EXIT_OK


def cmd_connect(args) -> int:
    """Sign in with an OrcaRouter account (adapter 2)."""
    store = CredentialStore(args.credential_file) if args.credential_file else CredentialStore()
    endpoints = resolve_endpoints()
    client = OrcaOAuthClient(endpoints=endpoints, timeout=args.timeout)

    def show_url(url: str) -> None:
        # The authorize URL carries the challenge and state, never the verifier.
        print("Opening your browser to authorize this app.")
        print(f"If it does not open, visit:\n  {url}\n")

    def show_device(payload: dict) -> None:
        complete = payload.get("verification_uri_complete")
        if complete:
            print(f"Open this on any device:\n  {complete}\n")
        else:
            print(
                f"Open {payload.get('verification_uri')} and enter the code: "
                f"{payload.get('user_code')}\n"
            )

    kwargs = {"on_url": show_url}
    if args.flow == "oob":
        kwargs["prompt_code"] = lambda: _read_secret("Code from the consent screen: ")
    elif args.flow == "device":
        kwargs = {"on_prompt": show_device}

    adapter = PkceAdapter(client, store)
    result = adapter.acquire(flow=args.flow, **kwargs)

    if not result.ok:
        payload = {"ok": False, "error": result.message}
        if args.json:
            payload["lines"] = [result.message]
        else:
            payload["lines"] = [result.message]
        _emit(payload, args.json)
        return EXIT_CANCELLED if isinstance(result, object) and "cancelled" in result.message.lower() else EXIT_FAILURE

    credential = result.credential
    _emit(
        {
            "ok": True,
            "masked": credential.masked,
            "scope": credential.scope,
            "user_id": credential.user_id,
            "flow": args.flow,
            "lines": [
                f"Connected. Saved {credential.masked} to {store.path}",
                f"Granted scope: {credential.scope}",
                "This key is reused until it is revoked in the OrcaRouter console;",
                "it is not refreshed, and re-authorizing is rate limited.",
            ],
        },
        args.json,
    )
    return EXIT_OK


def cmd_models(args) -> int:
    """Show the live model catalog, filtered for one capability."""
    store = CredentialStore(args.credential_file) if args.credential_file else CredentialStore()
    credential = resolve_credential(store)
    client = OrcaCatalogClient(resolve_endpoints())
    modalities = args.modality or None

    try:
        result = client.fetch(
            args.capability,
            credential=credential,
            required_modalities=modalities,
            allow_cache=not args.refresh,
        )
    except ValueError as exc:
        _emit({"ok": False, "error": str(exc), "lines": [str(exc)]}, args.json)
        return EXIT_FAILURE

    lines = [
        f"{result.capability} models ({result.source}"
        + (", degraded" if result.degraded else "")
        + f") — {result.count} available",
        f"  from {resolve_endpoints().models_url(CAPABILITIES[result.capability]['query'])}",
    ]
    if result.degraded and result.error:
        lines.append(f"  live discovery unavailable: {result.error}")
    lines.append("")
    for model in result.models:
        parts = [f"  {model.id}"]
        if model.input_modalities:
            parts.append(f"[{','.join(sorted(model.input_modalities))}]")
        if model.context_length:
            parts.append(f"{model.context_length} ctx")
        if model.source != "live":
            parts.append(f"({model.source})")
        lines.append(" ".join(parts))

    _emit(
        {
            "ok": True,
            "source": result.source,
            "degraded": result.degraded,
            "error": result.error,
            "count": result.count,
            "models": [m.to_public_dict() for m in result.models],
            "lines": lines,
        },
        args.json,
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    # The two shared flags are accepted on either side of the subcommand. The
    # copies on the subparsers use SUPPRESS so that a value given before the
    # subcommand is not overwritten by the subparser's default.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="machine-readable output",
    )
    common.add_argument(
        "--credential-file", default=argparse.SUPPRESS,
        help="override the credential path",
    )

    parser = argparse.ArgumentParser(
        prog="python -m server.orcarouter",
        description="OrcaRouter credential and model catalog CLI.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--credential-file", default=None, help="override the credential path")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", parents=[common], help="show configured authentication methods")

    key = sub.add_parser("key", parents=[common],
                         help="save or clear an existing API key")
    key.add_argument("--api-key", default=None,
                     help="the sk-orca-... value (prefer the prompt)")
    key.add_argument("--clear", action="store_true", help="remove the stored credential")

    connect = sub.add_parser("connect", parents=[common],
                             help="sign in with OAuth 2.0 + PKCE")
    connect.add_argument(
        "--flow",
        choices=("loopback", "oob", "device"),
        default="loopback",
        help="loopback: browser + local callback (default); oob: paste the code; "
             "device: RFC 8628 device grant",
    )
    connect.add_argument("--timeout", type=float, default=300.0, help="login timeout in seconds")

    sub.add_parser("logout", parents=[common], help="forget the stored credential")

    models = sub.add_parser("models", parents=[common],
                            help="list models from the live catalog")
    models.add_argument("--capability", choices=sorted(CAPABILITIES), default="chat")
    models.add_argument(
        "--modality",
        action="append",
        choices=("image", "audio", "video"),
        help="require a declared input modality (repeatable)",
    )
    models.add_argument("--refresh", action="store_true", help="bypass the cache")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_OK

    handlers = {
        "status": cmd_status,
        "key": cmd_key,
        "connect": cmd_connect,
        "logout": cmd_logout,
        "models": cmd_models,
    }
    try:
        return handlers[args.command](args)
    except OrcaAuthDenied as exc:
        _emit({"ok": False, "error": exc.full_message, "lines": [exc.full_message]}, args.json)
        return EXIT_FAILURE
    except OrcaAuthCancelled as exc:
        _emit({"ok": False, "error": exc.full_message, "lines": [exc.full_message]}, args.json)
        return EXIT_CANCELLED
    except OrcaAuthError as exc:
        _emit({"ok": False, "error": exc.full_message, "lines": [exc.full_message]}, args.json)
        return EXIT_FAILURE
    except OrcaConfigError as exc:
        _emit({"ok": False, "error": str(exc), "lines": [f"configuration error: {exc}"]}, args.json)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        _emit({"ok": False, "error": "cancelled", "lines": ["Cancelled."]}, args.json)
        return EXIT_CANCELLED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
