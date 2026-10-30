"""
Authentication — Playwright login + session capture.

Mirrors the Windows-Copilot-API design: the browser is used ONLY to establish a
signed-in session (handling the AWS WAF / "verify you're human" check and the
email/password form). It does not chat. We then capture the bearer token from
`localStorage.userToken` plus the session cookies, and hand them to the
pure-HTTP client in `deepseek.client`.

A persistent Chromium profile means the human-check is a one-time thing: once
you've signed in, later runs reuse the profile and capture the token headlessly.

    from deepseek.auth import get_session
    session = get_session()          # logs in (visible) the first time, else headless
    print(session.token[:8], "...")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
# Load .env before reading settings below, whichever entry point imported us.
load_dotenv(ROOT / ".env")
# Override with DEEPSEEK_PROFILE_DIR to reuse an existing signed-in Chrome profile.
DEFAULT_PROFILE_DIR = Path(os.getenv("DEEPSEEK_PROFILE_DIR", ROOT / "session" / "profile"))
DEFAULT_SESSION_FILE = ROOT / "session" / "session.json"

CHAT_URL = "https://chat.deepseek.com/"
SIGNIN_URL = "https://chat.deepseek.com/sign_in"

LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]
# Token is trusted for this long before we refresh it from the browser again.
SESSION_MAX_AGE = 6 * 60 * 60  # 6 hours


class LoginRequired(RuntimeError):
    """Raised when no usable session exists and interactive login is disallowed
    (e.g. inside the server, where we can't pop open a browser mid-request).
    The message tells the user how to log in."""

    DEFAULT = (
        "No DeepSeek session found. Log in first by running:\n"
        "    python -m deepseek.auth\n"
        "This opens a browser once so you can sign in and clear the human-check; "
        "afterwards the server reuses the saved session automatically."
    )

    def __init__(self, message: str = DEFAULT):
        super().__init__(message)


@dataclass
class Session:
    """A captured signed-in DeepSeek session."""

    token: str
    cookies: Dict[str, str]
    user_agent: str
    captured_at: float

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    def save(self, path: Path = DEFAULT_SESSION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_SESSION_FILE) -> Optional["Session"]:
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


# --- in-page helpers --------------------------------------------------------

# Reads the bearer token the web app stores after login. Shape:
#   localStorage.userToken = {"value":"<TOKEN>","__version":"0"}
_READ_TOKEN_JS = """
() => {
  try {
    const raw = window.localStorage.getItem('userToken');
    if (!raw) return null;
    const o = JSON.parse(raw);
    return (o && o.value) ? o.value : null;
  } catch (e) { return null; }
}
"""


def _safe_evaluate(page, js: str):
    """Run page.evaluate, swallowing the benign 'Execution context was destroyed'
    error that fires when a navigation (e.g. the post-login redirect) happens to
    land mid-evaluate. Returns None on any such transient failure instead of
    raising, so callers can just retry on the next poll."""
    try:
        return page.evaluate(js)
    except Exception as e:
        msg = str(e)
        if "Execution context was destroyed" in msg or "navigation" in msg.lower():
            return None
        raise


def _capture_from_context(context, page) -> Optional[Session]:
    """Read token + cookies + UA off a logged-in page, or None if not signed in."""
    token = _safe_evaluate(page, _READ_TOKEN_JS)
    if not token:
        return None
    cookies = {c["name"]: c["value"] for c in context.cookies()}
    ua = _safe_evaluate(page, "() => navigator.userAgent") or ""
    return Session(token=token, cookies=cookies, user_agent=ua, captured_at=time.time())


def _wait_for_token(page, timeout: float) -> Optional[str]:
    """Poll localStorage.userToken until it appears or we time out. Tolerates
    transient navigations (e.g. the Google OAuth redirect chain) that briefly
    destroy the page's JS execution context — those just count as "no token
    yet" rather than aborting the whole login."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        token = _safe_evaluate(page, _READ_TOKEN_JS)
        if token:
            return token
        page.wait_for_timeout(1000)
    return None


def _safe_goto(page, url: str) -> None:
    """Navigate, tolerating the benign `net::ERR_ABORTED` that DeepSeek's SPA
    redirects and the AWS WAF check often raise mid-navigation. We wait only for
    the initial commit, not full `load`; the token-poll afterwards is what
    actually gates sign-in, so an aborted/partial load here is fine."""
    try:
        page.goto(url, wait_until="commit", timeout=60000)
    except Exception as e:
        print(f"[auth] navigation to {url} was interrupted ({type(e).__name__}); "
              "continuing — finish signing in in the window if needed.")
    # Give the SPA a moment to render its login UI before we touch the form.
    page.wait_for_timeout(2000)


def login(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    headless: bool = False,
    assume_logged_out: bool = False,
) -> Session:
    """Interactive login. Opens a visible window and waits for you to sign in by
    hand (and clear the AWS WAF human-check); once a token appears it captures
    and saves the session. The persistent profile means later `get_session()`
    calls capture the token headlessly without a window.

    `assume_logged_out=True` skips the initial "are we already signed in?" hop to
    CHAT_URL and goes straight to the sign-in page. Callers that have just
    confirmed there's no token (e.g. get_session after a failed headless refresh)
    pass this so the window doesn't visibly bounce CHAT_URL -> SIGNIN_URL."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=headless, channel="chrome", args=LAUNCH_ARGS,
            )
        except Exception:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=headless, args=LAUNCH_ARGS,
            )
        page = context.pages[0] if context.pages else context.new_page()

        # Normally we first land on CHAT_URL to reuse an already-signed-in
        # profile. When the caller already knows we're logged out, skip straight
        # to the sign-in page so the window doesn't appear to "refresh".
        existing = None
        if not assume_logged_out:
            _safe_goto(page, CHAT_URL)
            existing = _safe_evaluate(page, _READ_TOKEN_JS)

        if not existing:
            _safe_goto(page, SIGNIN_URL)
            print("[auth] Please sign in in the window (solve the human-check if "
                  "shown). Waiting for the session...")
            if not _wait_for_token(page, timeout=300):
                context.close()
                raise RuntimeError("Login timed out — no token captured.")

        session = _capture_from_context(context, page)
        context.close()
        if session is None:
            raise RuntimeError("Logged in but could not read the token.")
        session.save()
        return session


def _headless_refresh(profile_dir: Path) -> Optional[Session]:
    """Try to capture a token headlessly from the persistent profile. Returns a
    saved Session if the profile is still signed in, else None. Never opens a
    visible window."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=True, channel="chrome", args=LAUNCH_ARGS,
            )
        except Exception:
            context = p.chromium.launch_persistent_context(
                str(profile_dir), headless=True, args=LAUNCH_ARGS,
            )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            _safe_goto(page, CHAT_URL)
            session = _capture_from_context(context, page)
        finally:
            context.close()

    if session is not None:
        session.save()
    return session


def get_session(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    session_file: Path = DEFAULT_SESSION_FILE,
    max_age: int = SESSION_MAX_AGE,
    allow_interactive: bool = True,
) -> Session:
    """Return a usable session: cached file if fresh, else a headless refresh
    from the browser profile.

    If neither works and `allow_interactive` is True, open a visible window for
    manual sign-in. If it's False (the server's case — we can't pop a browser
    mid-request), raise `LoginRequired` telling the user to run the login step.

    Note: this uses Playwright's *sync* API, so it must not be called from inside
    an asyncio event loop — call it from a worker thread (e.g. run_in_threadpool)."""
    cached = Session.load(session_file)
    if cached and cached.age < max_age:
        return cached

    # Try a headless refresh from the (presumably logged-in) persistent profile.
    session = _headless_refresh(profile_dir)
    if session is not None:
        return session

    if not allow_interactive:
        raise LoginRequired()

    # Not logged in yet — open a visible window so the user can sign in (and
    # clear the human-check) by hand. The persistent profile means this only
    # happens once — later calls capture the token headlessly. We just confirmed
    # (above) there's no token, so go straight to the sign-in page.
    print("[auth] No valid session found — opening a browser window to log in...")
    return login(profile_dir=profile_dir, assume_logged_out=True)


if __name__ == "__main__":
    s = login()
    print(f"[auth] captured token {s.token[:10]}... ({len(s.cookies)} cookies)")
