# Examples

Runnable examples for both ways to use this project. Run each from the **project
root** (e.g. `python examples/01_direct_chat.py`).

## Directly, in Python (`DeepSeekClient`)

No server needed. On the first run, sign-in opens a browser window automatically
— sign in by hand and solve the human-check; it continues once a session is
captured.

| File | Shows |
| --- | --- |
| [01_direct_chat.py](01_direct_chat.py) | The simplest one-shot chat (plus `model` / `thinking` / `search`) |
| [02_direct_conversation.py](02_direct_conversation.py) | Multi-turn — continue with `conversation_id` |
| [03_direct_stream.py](03_direct_stream.py) | Stream the reply as it's generated |

## Over HTTP (the OpenAI-compatible server)

Start the server first in another terminal: `python app.py`

| File | Shows |
| --- | --- |
| [04_server_http.py](04_server_http.py) | Plain HTTP with `requests` (multi-turn via `messages`) |
| [05_server_stream.py](05_server_stream.py) | Streaming over Server-Sent Events |
| [06_server_openai_sdk.py](06_server_openai_sdk.py) | The official `openai` SDK (`pip install openai`) |
| [07_server_tool_calls.py](07_server_tool_calls.py) | Tool / function calling — pass `tools`, receive `tool_calls`, send results back |

The server limits requests per client IP (default 30/min, set via
`RATE_LIMIT_PER_MINUTE`); over the limit you get HTTP `429` with a `Retry-After`
header.

The `model` name selects the model — `deepseek-chat` (fast) or `deepseek-expert`
(stronger, slower); see [server/config.py](../server/config.py). DeepThink and web
search are separate toggles, requested via the `thinking` and `search` booleans in
the request body — or through `extra_body` with the OpenAI SDK.

Tool calling is a prompt-based shim (DeepSeek has no native tool channel), so the
model can occasionally return a call as prose or malformed JSON. Parse
defensively. See [07_server_tool_calls.py](07_server_tool_calls.py) for the full
loop.
