# DeepSeek API: a free LLM API powered by DeepSeek

**Using your own DeepSeek account.** No API key, no credits, no paid plan: it turns the free chat at [chat.deepseek.com](https://chat.deepseek.com) into an API you can call from code.

You can use it in two ways:

- 🐍 **As a Python library:** just call `client.chat("Hi")`. Supports streaming and multi-turn conversations.
- 🔌 **As a local OpenAI-compatible API:** runs a server at `http://localhost:8000/v1` that speaks the OpenAI format, so the official `openai` SDK (and any OpenAI-compatible app) works as a drop-in, with `localhost` in place of OpenAI.

You sign in once in a browser with your DeepSeek account; your session is saved and refreshed automatically after that.

> **Unofficial project.** Not affiliated with or endorsed by DeepSeek. It automates the consumer DeepSeek web experience for personal use, so use it responsibly and within DeepSeek's terms.

---

## Table of contents

- [Why use this?](#why-use-this)
- [Requirements](#requirements)
- [Setup (2 minutes)](#setup-2-minutes)
- [Usage 1: In Python (no server)](#usage-1-in-python-no-server)
- [Usage 2: As an OpenAI-compatible server](#usage-2-as-an-openai-compatible-server)
- [Command line](#command-line)
- [Human-check & proof-of-work (automatic)](#human-check--proof-of-work-automatic)
- [Models, DeepThink & web search](#models-deepthink--web-search)
- [Tool / Function Calling (Emulated)](#tool--function-calling-emulated)
- [Concurrency](#concurrency)
- [Rate limiting](#rate-limiting)
- [Project layout](#project-layout)
- [Notes & limitations](#notes--limitations)
- [License](#license)

---

## Why use this?

- **Free:** uses your normal signed-in DeepSeek account, no API billing.
- **Drop-in OpenAI replacement:** point any OpenAI client at `localhost` and it just works.
- **Full DeepSeek toolset:** pick the fast or expert model, and toggle DeepThink reasoning and web search per request.
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

| Method | Path | Description | Authentication |
| --- | --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat completions (supports `"stream": true`, plus optional `"conversation_id"`, `"thinking"`, `"search"`, `"user"`) | Bearer Proxy API Key |
| `GET`  | `/v1/models` | Lists the available models | Bearer Proxy API Key |
| `GET`  | `/v1/chat/sessions` | Lists chat sessions from the DeepSeek dashboard (supports date filtering via `?days=N`) | Bearer Proxy API Key |
| `DELETE` | `/v1/chat/session/{user_id}` | Deletes a user's session from DeepSeek and the local mapping | Bearer Proxy API Key |
| `POST` | `/v1/session/import` | Administratively imports a raw session JSON and hot-reloads client | Bearer Admin Token |
| `GET`  | `/healthz` | Health check | Exempt |

> Change the address with env vars: `HOST=0.0.0.0 PORT=8080 python app.py`, or run `uvicorn server.api:app --host 0.0.0.0 --port 8080`.

👉 More: [examples/04_server_http.py](examples/04_server_http.py), [examples/05_server_stream.py](examples/05_server_stream.py), [examples/06_server_openai_sdk.py](examples/06_server_openai_sdk.py)

#### 🔑 Authentication & API Keys

To protect your proxy from unauthorized access, configure these key variables in your `.env` file:
*   `DEEPSEEK_PROXY_API_KEY`: A standard API Key (e.g. `sk-my-proxy-key`) that clients must send in the standard `Authorization: Bearer <KEY>` header when calling `/v1/chat/completions`, `/v1/models`, or `/v1/chat/sessions`.
*   `PROXY_ADMIN_TOKEN`: A token used to secure administrative endpoints like `/v1/session/import`.
*   **Swagger Interactive Authorization:** When visiting `http://localhost:8000/docs`, you will see lock icons on the endpoints and an "Authorize" button. Click "Authorize", paste your configured key, and you can test endpoints interactively.

#### 🧠 Automatic Session Mapping & Memory (the `user` parameter)

For stateless clients (like **n8n**, **WhatsApp bots**, or **Telegram bots**) that do not maintain conversation IDs natively, the proxy includes an **automatic session mapping layer**:
1.  **Passing the `user` ID:** In your chat completions payload, pass a unique identifier (like the WhatsApp phone number or a session ID) in the standard OpenAI `user` parameter:
    ```json
    {
      "model": "deepseek-chat",
      "messages": [{"role": "user", "content": "Hello!"}],
      "user": "whatsapp_+34600000000"
    }
    ```
2.  **Thread Resuming & Isolation:** The proxy maps this `user` ID to a unique DeepSeek thread ID in `session/user_threads.json`. On subsequent requests, it resumes that thread automatically. Each user's context is completely isolated.
3.  **Native DeepSeek Memory:** When resuming a thread, the proxy sends **only the latest user message** to DeepSeek. This lets DeepSeek's backend manage conversation memory natively, preventing duplicate histories and reducing bandwidth.
4.  **Memory Reset:** To clear a user's thread (erasing their memory and deleting the chat from your DeepSeek dashboard), call `DELETE /v1/chat/session/{user_id}`.

#### 🔌 n8n Integration & AI Agents

This server is fully compatible with **n8n** (using the OpenAI Chat Model node inside the AI Agent). Since n8n does not easily support custom request body parameters, the server exposes **virtual model names** (e.g. `deepseek-chat-thinking-search`) to toggle DeepThink and web search.

See the [n8n Integration Guide](N8N_INTEGRATION.md) for step-by-step setup details.

---

## Command line & Headless Import/Export

For local environments with a GUI:
```bash
python -m deepseek.auth          # sign in via browser and save the session
```

### Remote / Headless Environments (VPS, Docker, Proxmox)

If you run the server on a remote headless machine, you cannot open a browser to log in. Instead, you can export your active session from your local GUI machine and import it on the remote machine:

1. **On your local machine (with GUI)**:
   * Sign in to generate a session:
     ```bash
     python -m deepseek.auth
     ```
   * Export the active session details as a JSON block:
     ```bash
     python -m deepseek.auth --export
     ```
   * Copy the JSON output from the terminal.

2. **On your remote machine (headless/no GUI)**:
   * Run the import command:
     ```bash
     python -m deepseek.auth --import
     ```
   * Paste the JSON block you copied and press `Ctrl+D` (on Linux) or `Ctrl+Z` (on Windows) then `Enter` to save the session.
   * Make sure to set `SERVER_INTERACTIVE_LOGIN=false` in your `.env` to prevent the server from trying to launch a browser remote-side.

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

`conversation_id`, `thinking`, and `search` are non-OpenAI extras. A thread's
model is fixed at creation, so `model` can't be combined with `conversation_id`
on resume. Unknown model names return a `404` (no silent fallback). See
[server/config.py](server/config.py).

---

## Tool / Function Calling (Emulated)

The server supports emulated OpenAI-style **Tool/Function Calling**. Since the web version of DeepSeek does not natively support tool registration via the API, the proxy handles this transparently:

1. **System Instruction Injection:** When you pass a `tools` list in the OpenAI request, the proxy formats their descriptions and parameters, appending them as a system instruction at the start of the prompt.
2. **JSON Parsing:** If the model outputs a JSON block requesting a tool call, the proxy intercepts the response, extracts the call, and maps it back into the standard OpenAI `tool_calls` structure.

This allows any OpenAI-compatible client (including LangChain, n8n, etc.) to trigger local tools automatically.

Example using the OpenAI SDK:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_dolar_rate",
            "description": "Get the current USD to VEF exchange rate.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What is the current dollar rate?"}],
    tools=tools
)

# If the model requested the tool:
message = resp.choices[0].message
if message.tool_calls:
    for tc in message.tool_calls:
        print(f"Tool request: {tc.function.name} with args {tc.function.arguments}")
```

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
| [server/](server/) | The FastAPI OpenAI-compatible server |
| [examples/](examples/) | Runnable examples for every feature ([examples/README.md](examples/README.md)) |
| [app.py](app.py) | Starts the server |

---

## Notes & limitations

- **Sign in once, then reuse.** The cached session refreshes automatically; you only re-sign-in if it fully expires.
- **Be reasonable.** Please use it in moderation, and don't spam or hammer it with automated bulk requests.
- **No real token counts.** `usage` in responses is a rough ~4-chars/token estimate.
- **Most OpenAI params are accepted but ignored** (`temperature`, `top_p`, `max_tokens`); only `model`, `messages`, `stream`, `conversation_id`, `thinking`, `search`, and `tools` do anything.
- **Vision is deferred.** It needs image-upload plumbing that isn't built yet.
- **Your session is private.** Everything in `session/` (cookies + token) stays on your machine and is git-ignored.

## License

Released under the [MIT License](LICENSE). As this is an unofficial project, you remain responsible for complying with DeepSeek's terms of service.

---

