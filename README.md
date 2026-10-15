# DeepSeek API: a free LLM API powered by DeepSeek

**Using your own DeepSeek account.** No API key, no credits, no paid plan: it turns the free chat at [chat.deepseek.com](https://chat.deepseek.com) into an API you can call from code.

You can use it in two ways:

- 🐍 **As a Python library:** just call `client.chat("Hi")`. Supports streaming and multi-turn conversations.
- 🔌 **As a local OpenAI-compatible API:** runs a server at `http://localhost:8000/v1` that speaks the OpenAI format, so the official `openai` SDK (and any OpenAI-compatible app) works as a drop-in, with `localhost` in place of OpenAI.
- 🧰 **With tool / function calling:** pass OpenAI-style `tools` and get standard `tool_calls` back — enough for agent loops, function calling, and structured tool use.

You sign in once in a browser with your DeepSeek account; your session is saved and refreshed automatically after that.

> **Unofficial project.** Not affiliated with or endorsed by DeepSeek. It automates the consumer DeepSeek web experience for personal use, so use it responsibly and within DeepSeek's terms.

---

## Status

| Area | State |
| --- | --- |
| Python library (`deepseek/`) | Stable — chat, streaming, conversations |
| OpenAI-compatible server (`server/`) | Stable — `/v1/chat/completions`, `/v1/models`, `/healthz` |
| Tool / function calling | **New** — prompt-based shim (see below) |
| DeepThink & web search toggles | Stable |
| Vision / image input | Deferred |
| Automated test suite | None yet |

---

## Table of contents

- [Why use this?](#why-use-this)
- [Requirements](#requirements)
- [Setup (2 minutes)](#setup-2-minutes)
- [Usage 1: In Python (no server)](#usage-1-in-python-no-server)
- [Usage 2: As an OpenAI-compatible server](#usage-2-as-an-openai-compatible-server)
- [Tool / function calling](#tool--function-calling)
- [Command line](#command-line)
- [Human-check & proof-of-work (automatic)](#human-check--proof-of-work-automatic)
- [Models, DeepThink & web search](#models-deepthink--web-search)
- [Concurrency](#concurrency)
- [Rate limiting](#rate-limiting)
- [Project layout](#project-layout)
- [Notes & limitations](#notes--limitations)
- [Status](#status)
- [License](#license)

---

## Why use this?

- **Free:** uses your normal signed-in DeepSeek account, no API billing.
- **Drop-in OpenAI replacement:** point any OpenAI client at `localhost` and it just works.
- **Full DeepSeek toolset:** pick the fast or expert model, and toggle DeepThink reasoning and web search per request.
- **Tool calling:** pass OpenAI `tools` / `tool_choice` and receive standard `tool_calls` responses.
- **Streaming + conversations:** token-by-token output and multi-turn threads addressed by `conversation_id`.

---

## Requirements

- **Python 3.9+**
- A **DeepSeek account** (the free one you use for [chat.deepseek.com](https://chat.deepseek.com) is fine)
- Works on Windows, macOS, and Linux

---

## Setup (2 minutes)

```bash
# 1. Clone the project
git clone https://github.com/sums001/Deepseek-API.git
cd "Deepseek-API"
```

**2. Create and activate a virtual environment**

On **macOS / Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

On **Windows** (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> On Windows you may need to allow script execution once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In `cmd.exe` activate with `venv\Scripts\activate.bat` instead.

**3. Install dependencies and sign in**

```bash
# Install dependencies
pip install -r requirements.txt

# Install the browser Playwright needs (one-time)
playwright install chromium

# Sign in once: a browser opens, log into your DeepSeek account
python -m deepseek.auth
```

The login window opens so you can sign in by hand and solve the human-check once. After that your session (bearer token + cookies) is saved under `session/` (git-ignored, never shared) and reused on every run — the cached session is refreshed automatically, so your first request works right away.

> The server can also open this window for you on demand the first time it needs a session, so this step is optional for local single-user use.

---

## Usage 1: In Python (no server)

The simplest way if your code is already Python.

```python
from deepseek import DeepSeekClient

client = DeepSeekClient()                # loads your signed-in session

# Get a full reply
reply = client.chat("Say hello in one short sentence.")
print(reply.text)

# Continue the SAME conversation — pass the id back
reply2 = client.chat("And now in French?", conversation_id=reply.conversation_id)
print(reply2.text)

# Stream the answer as it's typed
for chunk in client.stream("Tell me a short joke"):
    print(chunk, end="", flush=True)
```

`chat()` returns the full text plus a `conversation_id`; pass that id back to keep the thread going, or omit it to start fresh. `stream()` yields the reply piece by piece.

👉 More: [examples/01_direct_chat.py](examples/01_direct_chat.py), [02_direct_conversation.py](examples/02_direct_conversation.py), [03_direct_stream.py](examples/03_direct_stream.py)

---

## Usage 2: As an OpenAI-compatible server

Start a local server that speaks the OpenAI API, so existing OpenAI tools and SDKs work unchanged.

```bash
python app.py
# -> DeepSeek OpenAI-compatible API on http://127.0.0.1:8000
```

Then point any OpenAI client at it (the API key is required by the SDK but ignored):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Or call it with plain HTTP / `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Endpoints**

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat (supports `"stream": true`, optional `"conversation_id"`, `"thinking"`, `"search"`, and `"tools"` / `"tool_choice"`) |
| `GET`  | `/v1/models` | Lists the available models |
| `GET`  | `/healthz` | Health check (rate-limit exempt) |

> Change the address with env vars: `HOST=0.0.0.0 PORT=8080 python app.py`, or run `uvicorn server.api:app --host 0.0.0.0 --port 8080`.

👉 More: [examples/04_server_http.py](examples/04_server_http.py), [examples/05_server_stream.py](examples/05_server_stream.py), [examples/06_server_openai_sdk.py](examples/06_server_openai_sdk.py)

---

## Tool / function calling

The server accepts OpenAI-style tool definitions and returns OpenAI-style
`tool_calls`, so the standard function-calling loop works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What's the weather in Auckland?"}],
    tools=tools,
    tool_choice="auto",
)

msg = resp.choices[0].message
if msg.tool_calls:
    call = msg.tool_calls[0]
    print(call.function.name, call.function.arguments)   # -> get_weather {"city": "Auckland"}

    # Send the tool result back to continue the loop
    follow = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "What's the weather in Auckland?"},
            msg,
            {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 18, "sky": "cloudy"}'},
        ],
        tools=tools,
    )
    print(follow.choices[0].message.content)
```

### How it works

DeepSeek's web protocol has **no native tool-call channel** — only a single
`prompt` string. So the server implements a *shim* (see
[server/openai_format.py](server/openai_format.py)):

1. Your `tools` are rendered into a prompt instruction asking the model to
   reply with a JSON block `{"tool_calls": [{"name": ..., "arguments": {...}}]}`.
2. The model's text is scanned for that JSON block and translated into OpenAI's
   `tool_calls` shape — `content` becomes `null` and `finish_reason` is
   `"tool_calls"`.
3. `tool`-role result messages and prior assistant `tool_calls` are serialised
   back into the flattened prompt so multi-turn tool conversations work.

`tool_choice` is honoured as a prompt-level hint: `"none"` (never call),
`"required"` (must call one), or `{"type": "function", "function": {"name": ...}}`
(force a specific function). `"auto"` is the default behaviour.

> **Reliability caveat.** Because this is prompt-driven rather than a native
> tool API, the model may occasionally emit a tool call as prose or malformed
> JSON. Parse defensively and keep prompts explicit.

---

## Command line

```bash
python -m deepseek.auth          # sign in and save the session
```

---

## Human-check & proof-of-work (automatic)

DeepSeek's chat sits behind two gates, both handled for you:

- **AWS WAF human-check:** access needs a signed-in browser session that has
  cleared the "verify you're human" check. `python -m deepseek.auth` opens a real
  browser so you can sign in and solve it once; the resulting token + cookies are
  cached under `session/` and reused on every request.
- **Proof-of-work:** every completion is gated by a PoW challenge. The bridge
  solves it by running DeepSeek's own `sha3_wasm_bg.wasm` module — the same one
  the browser loads — inside a `wasmtime` sandbox, so there's nothing to do on
  your end.

A cached session is reused for ~6 hours and refreshed headlessly from your saved
Chrome profile when possible; only a full expiry sends you back to the browser.

---

## Models, DeepThink & web search

The `model` name selects **which model** answers. DeepThink and web search are
**not** models — they're orthogonal toggles you pass per request.

| Model | DeepSeek mode | Notes |
| --- | --- | --- |
| `deepseek-chat` | Instant | Fast default model |
| `deepseek-expert` | Expert | Stronger, slower |

Pass `thinking: true` (DeepThink reasoning) and/or `search: true` (web search) in
the request body — or via the OpenAI SDK's `extra_body`:

```python
resp = client.chat.completions.create(
    model="deepseek-expert",
    messages=[{"role": "user", "content": "What changed in the news today?"}],
    extra_body={"thinking": True, "search": True},
)
```

`conversation_id`, `thinking`, `search`, `tools`, and `tool_choice` are
non-OpenAI extras. A thread's model is fixed at creation, so `model` can't be
combined with `conversation_id` on resume. Unknown model names return a `404`
(no silent fallback). See [server/config.py](server/config.py).

---

## Concurrency

The server bridges a **single** signed-in DeepSeek account behind one shared
client. The PoW solver's `wasmtime` store isn't reentrant, so upstream calls are
**serialized**: parallel HTTP requests queue behind a lock and run one at a time
(see [server/api.py](server/api.py)). This is intentional — throughput is
sequential, not parallel. Keep concurrent in-flight requests low, and please
don't hammer your account.

---

## Rate limiting

On top of serialization, the bridge enforces a self-imposed rate limit with a
dependency-free sliding-window limiter ([server/ratelimit.py](server/ratelimit.py)):
it caps accepted requests **per client IP** and returns a standard `429` +
`Retry-After` when you exceed it. `/healthz` is exempt.

| Env var | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests/minute accepted per client IP |

```bash
RATE_LIMIT_PER_MINUTE=60 python app.py   # raise it
```

**On the client side, use exponential backoff.** Transient `429`s clear if you
retry with growing delays (e.g. 1s, 2s, 4s). The official `openai` SDK does this
automatically and honours `Retry-After`; with plain HTTP, add a few retries
yourself.

---

## Project layout

| Path | What it does |
| --- | --- |
| [deepseek/](deepseek/) | The core library: `DeepSeekClient`, auth/browser sign-in ([auth.py](deepseek/auth.py)), the HTTP driver ([client.py](deepseek/client.py)), and the PoW solver ([pow.py](deepseek/pow.py)) |
| [server/](server/) | The FastAPI OpenAI-compatible server — request/response schemas ([schemas.py](server/schemas.py)), OpenAI↔DeepSeek translation incl. tool calling ([openai_format.py](server/openai_format.py)), endpoints ([api.py](server/api.py)), config ([config.py](server/config.py)), rate limiting ([ratelimit.py](server/ratelimit.py)) |
| [examples/](examples/) | Runnable examples for every feature ([examples/README.md](examples/README.md)) |
| [app.py](app.py) | Starts the server |

---

## Notes & limitations

- **Sign in once, then reuse.** The cached session refreshes automatically; you only re-sign-in if it fully expires.
- **Be reasonable.** Please use it in moderation, and don't spam or hammer it with automated bulk requests.
- **No real token counts.** `usage` in responses is a rough ~4-chars/token estimate.
- **Tool calling is a shim.** It's prompt-driven, not a native tool API — parse responses defensively. See [Tool / function calling](#tool--function-calling).
- **Most other OpenAI params are accepted but ignored** (`temperature`, `top_p`, `max_tokens`); only `model`, `messages`, `stream`, `conversation_id`, `thinking`, `search`, `tools`, and `tool_choice` do anything.
- **Vision is deferred.** It needs image-upload plumbing that isn't built yet.
- **Your session is private.** Everything in `session/` (cookies + token) stays on your machine and is git-ignored.

---

## License

Released under the [MIT License](LICENSE). As this is an unofficial project, you remain responsible for complying with DeepSeek's terms of service.
