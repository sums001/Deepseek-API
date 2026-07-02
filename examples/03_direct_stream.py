"""Example 3 — stream the reply as it is generated, in-process.

client.stream() yields (type, text) tuples as they arrive; type is "thinking" for
DeepThink internal reasoning and "content" for the actual response.  Good for
showing output live (like a chat UI typing).

Run it from the project root:

    python examples/03_direct_stream.py
"""

# Make the project importable when this file is run directly.
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from deepseek import DeepSeekClient

client = DeepSeekClient()

# Omit conversation_id to start fresh. Each chunk is a (type, text) tuple; the
# conversation_id is filled in once the stream finishes.
stream = client.stream("Tell me a short, clean joke.", thinking=True)
for chunk_type, chunk in stream:
    if chunk_type == "content":
        print(chunk, end="", flush=True)

print()  # newline after the streamed text
print("conversation_id:", stream.conversation_id)
if stream.reasoning_text:
    print("reasoning:", stream.reasoning_text[:120] + "...")
client.close()
