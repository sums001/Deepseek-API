"""Example 7 — tool / function calling against the server.

Pass OpenAI-style `tools` and the server returns standard `tool_calls`, so a
normal function-calling loop works: the model asks for a function, you run it,
and you send the result back.

DeepSeek's web protocol has no native tool channel, so the server renders your
tools into a prompt instruction and parses the JSON tool call back out — see
server/openai_format.py.

Install the SDK first:

    pip install openai

Start the server in another terminal:

    python app.py

Then run this from the project root:

    python examples/07_server_tool_calls.py
"""

import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# 1. Describe the functions the model may call (standard OpenAI schema).
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Auckland"},
            },
            "required": ["city"],
        },
    },
}]

# A real implementation of the tool. The model never runs this — you do.
def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 18, "sky": "cloudy"}

DISPATCH = {"get_weather": get_weather}

user_message = {"role": "user", "content": "What's the weather in Auckland?"}

# 2. First call: the model decides whether to call a function.
first = client.chat.completions.create(
    model="deepseek-chat",
    messages=[user_message],
    tools=tools,
    tool_choice="auto",   # "none" | "required" | {"type": "function", "function": {"name": ...}}
)

reply = first.choices[0].message
print("finish_reason:", first.choices[0].finish_reason)

if not reply.tool_calls:
    # The model answered directly — nothing to dispatch.
    print(reply.content)
    raise SystemExit

# 3. Run each requested function and collect the results.
tool_messages = [reply]
for call in reply.tool_calls:
    name = call.function.name
    args = json.loads(call.function.arguments or "{}")
    print(f"-> tool call: {name}({args})")
    result = DISPATCH[name](**args)
    tool_messages.append({
        "role": "tool",
        "tool_call_id": call.id,          # ties the result to the specific call
        "content": json.dumps(result),
    })

# 4. Second call: send the tool results back so the model can finish its answer.
final = client.chat.completions.create(
    model="deepseek-chat",
    messages=[user_message, *tool_messages],
    tools=tools,
)

print("answer:", final.choices[0].message.content)
