# OrcaRouter provider

This project normally answers through your own DeepSeek web session. It can also
answer through **OrcaRouter**, an OpenAI-compatible AI gateway that routes many
providers behind one endpoint. The two are independent: adding OrcaRouter does
not change the DeepSeek path, and you can use either one, or both.

- [Authentication](#authentication)
- [Origins](#origins)
- [Command line](#command-line)
- [Using it](#using-it)
- [Models](#models)
- [Multimodal input](#multimodal-input)
- [Credential lifecycle](#credential-lifecycle)

---

## Authentication

There are **two ways in**, and they are separately usable. Both end in the same
thing: an ordinary OrcaRouter API key that belongs to you, is billed to your
account, and is revocable by you at any time.

| Way in | Command | When to use it |
| --- | --- | --- |
| Existing API key | `python -m server.orcarouter key` | You already have an `sk-orca-…` key. |
| Sign in (OAuth 2.0 + PKCE) | `python -m server.orcarouter connect` | You would rather authorize this app than copy a key. |

An `ORCAROUTER_API_KEY` in the environment or in `.env` also works and takes
precedence over a stored key.

Neither path involves a client secret. PKCE binds the authorization code to
this process: the challenge is sent to the consent screen and the verifier
never leaves the machine, so an intercepted code cannot be redeemed by anyone
else. There is nothing to pre-register and no redirect URI to apply for.

### `connect`: two flows

```bash
python -m server.orcarouter connect               # Flow A: browser + loopback callback
python -m server.orcarouter connect --flow oob     # Flow B: paste the code back
python -m server.orcarouter connect --flow device  # Flow C: RFC 8628 device grant
```

**Flow A (loopback)** is the default. The client binds `127.0.0.1` on a random
free port *before* opening the browser, so the callback address is known and
nothing races. One click in the browser and the code comes back automatically.

**Flow B (out-of-band)** exists for the headless case — a VPS, a container, a
machine where the browser is somewhere else. The consent screen shows the code
and you paste it into the terminal.

**Flow C (device grant)** is a convenience for keyboard-only hosts. It is not a
replacement for PKCE and is only used when asked for explicitly.

Every flow sends `code_challenge_method=S256`. That is not optional even for
Flow A, because the consent screen lets you choose "show me a code" — which
puts the code into human hands, and under `plain` the challenge *is* the
verifier.

---

## Origins

Authentication and inference are deliberately different origins:

| Purpose | Origin |
| --- | --- |
| Consent screen | `https://www.orcarouter.ai/auth` |
| Code exchange | `https://www.orcarouter.ai/api/v1/auth/keys` |
| Inference | `https://api.orcarouter.ai/v1/chat/completions` |
| Model catalog | `https://api.orcarouter.ai/v1/models` |

`https://api.orcarouter.ai/v1/auth/keys` is a **404** — the relay is at `/v1`,
the auth endpoints are not. Neither origin here is derived from the other by
swapping a hostname.

To override them:

| Variable | Meaning |
| --- | --- |
| `ORCA_AUTH_BASE_URL` | Auth origin. Takes precedence over the shared value. |
| `ORCA_API_BASE_URL` | Inference base, including its `/v1`. Takes precedence. |
| `ORCA_BASE_URL` | Shared self-hosted base, used for both when neither override is set. |

Explicit overrides always win. Remote origins must be HTTPS; plain HTTP is
accepted only for loopback (`localhost`, `127.0.0.1`, `[::1]`). A typo in an
override is reported rather than silently routed to the public service.

---

## Command line

```bash
python -m server.orcarouter status     # what is configured, redacted
python -m server.orcarouter key        # save an existing API key (prompt, hidden)
python -m server.orcarouter key --clear
python -m server.orcarouter connect    # sign in with OAuth 2.0 + PKCE
python -m server.orcarouter logout     # forget the stored credential
python -m server.orcarouter models     # the live model catalog
python -m server.orcarouter models --capability multimodal --modality image
```

`status` shows both methods and their state, and never prints a key — only a
redacted form such as `sk-orca-abc…wxyz`. The API-key prompt reads from the
terminal rather than taking the secret as an argument, so it does not land in
your shell history.

---

## Using it

Start the server as usual and name an OrcaRouter model:

```bash
python app.py
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek/deepseek-v4-pro",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

A request routes to OrcaRouter when either:

- the model is prefixed with the provider — `orcarouter:deepseek/deepseek-v4-pro`; or
- the model id is one the live OrcaRouter catalog advertises, e.g.
  `deepseek/deepseek-v4-pro`.

Anything else keeps using the DeepSeek bridge, so existing clients are
unaffected. `GET /v1/models` backwards-compatible: it returns the DeepSeek ids
first and appends the OrcaRouter catalog when a credential is configured.

There is also a small control surface for the provider itself:

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/orcarouter/status` | Both authentication methods, the active credential (redacted), and login state. |
| `GET /v1/orcarouter/models` | The catalog for one capability, already filtered. Metadata only. |
| `POST /v1/orcarouter/connect/start` | Save a pasted key, or begin a PKCE login. |
| `POST /v1/orcarouter/connect/code` | Deliver an out-of-band code. |
| `POST /v1/orcarouter/connect/cancel` | Explicit cancel. |
| `POST /v1/orcarouter/connect/pagehide` | The page is going away; release the login lock. |
| `POST /v1/orcarouter/logout` | Forget the stored credential. |

The catalog and status endpoints return model metadata and a redacted
fingerprint only — a browser never receives an OrcaRouter key.

---

## Models

The model list comes from `GET https://api.orcarouter.ai/v1/models` on the
configured inference origin, and the vendor namespace is preserved exactly as
advertised (`deepseek/deepseek-v4-pro`, never `deepseek-v4-pro`). When a
credential is present the request is made with it, so the list reflects what
your workspace can actually call.

Entries are filtered per entry point, from the metadata the catalog supplies —
`supported_endpoint_types` and `architecture.input_modalities` — never from the
shape of a model name:

| Entry point | Accepted |
| --- | --- |
| Text chat / agent | `?capability=chat`, and an endpoint type among `openai`, `anthropic`, `gemini`, `openai-response`. Models specialised for `image-generation`, `openai-video`, `jina-rerank`, or `embeddings` are excluded. |
| Multimodal understanding | The chat rules, plus the attachment's modality must be **explicitly declared** in `architecture.input_modalities`. |
| Embedding / image / video / rerank | The matching capability or endpoint type exactly. |

A model that does not declare a capability is not offered for it; absence of a
declaration is not permission.

If discovery fails, the last known-good catalog is reused, and failing that a
small verified seed. Both are reported as degraded (`degraded: true` plus the
reason) so a caller can say so — a failed catalog never becomes a free-text
model field.

The seed is deliberately small and every entry carries only metadata that can
be attributed:

| Model | Notes |
| --- | --- |
| `openai/gpt-5.5` | Reasoning ladder `low`/`medium`/`high`/`xhigh`. |
| `anthropic/claude-opus-4.8` | |
| `google/gemini-3.5-flash` | |
| `deepseek/deepseek-v4-pro` | `context_length` and endpoint types observed live. |
| `orcarouter/auto` | Observed live. |

---

## Multimodal input

Vision is not part of the DeepSeek bridge — see "Notes & limitations" in the
main README. The OrcaRouter path does not change that, but it *does* support
image input for models that declare it, and it enforces it twice:

1. the model list for an image-bearing request contains only chat models that
   declare `image` in `architecture.input_modalities`;
2. a request that attaches an image to a model that did not declare it is
   refused with `400` before anything is sent upstream.

A remembered model that is no longer compatible — because an attachment changed
the required capability — is reported as incompatible and must be re-selected;
it is never silently kept.

---

## Credential lifecycle

A PKCE login returns a **durable API key**, not a refreshable token pair. There
is no refresh endpoint and this project does not invent one: the key is reused
until it is revoked from
[the OrcaRouter console](https://www.orcarouter.ai/console/authorized-apps).

Two consequences are handled explicitly:

- **A revoked key is terminal.** A `401` from the relay marks the exact account
  and credential generation that made the request as needing reauthorization,
  and the request fails once with a clear message. There is no retry loop and
  no fake refresh.
- **A late failure cannot poison a new login.** Because the mark is scoped to a
  generation, a `401` from a request made with an older key never marks a
  freshly authorized one as broken.

Authorization is also rate limited upstream: **at most 10 PKCE-issued keys per
user per 24 hours**. The stored key is reused across restarts, so a normal
install authorizes once.

The credential is stored alongside the DeepSeek session in the project's own
git-ignored `session/` directory, with owner-only permissions, and is never
written to a log, an error message, or a URL.

---

## Testing

```bash
python -m pytest -q                    # the whole suite
python -m pytest -q -m "not live"      # offline only
```

The `live`-marked tests are the only ones that touch the network; they run the
implemented provider path against the real service and skip when
`ORCAROUTER_API_KEY` is unset. Everything else uses fake credentials and a local
fake authorization server.
