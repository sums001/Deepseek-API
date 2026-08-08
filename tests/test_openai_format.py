import unittest
from server.schemas import ChatMessage, ChatCompletionRequest
from server.openai_format import (
    messages_to_prompt,
    extract_tool_calls_and_clean_text,
    completion_response,
    stream_chunks,
)

class TestOpenAIFormat(unittest.TestCase):
    def test_messages_to_prompt_with_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Execute bash command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }
        ]
        messages = [ChatMessage(role="user", content="cria um arquivo txt ai")]
        prompt = messages_to_prompt(messages, tools=tools)
        self.assertIn("System: # Tools", prompt)
        self.assertIn('"name": "Bash"', prompt)
        self.assertIn("User: cria um arquivo txt ai", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_messages_to_prompt_with_tool_calls_and_responses(self):
        messages = [
            ChatMessage(role="user", content="cria um arquivo txt ai"),
            ChatMessage(
                role="assistant",
                content="Vou criar o arquivo.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command": "touch /home/salada/arquivo.txt"}'}
                    }
                ]
            ),
            ChatMessage(role="tool", tool_call_id="call_123", content="Command executed successfully")
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn('Assistant: Vou criar o arquivo. <tool_call name="Bash">', prompt)
        self.assertIn('Tool Output (call_123): Command executed successfully', prompt)

    def test_extract_tool_calls_xml(self):
        raw_output = (
            "The user is asking me to create a txt file. They said \"cria um arquivo txt ai\". "
            "Let me create one in the current working directory.\n"
            '<tool_call name="Bash">\n'
            '<parameter name="command" string="true">touch /home/salada/arquivo.txt</parameter>\n'
            '<parameter name="description" string="true">Cria um arquivo txt vazio</parameter>\n'
            '</tool_call>'
        )
        content, tool_calls = extract_tool_calls_and_clean_text(raw_output)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "Bash")
        self.assertIn("touch /home/salada/arquivo.txt", tool_calls[0]["function"]["arguments"])
        self.assertIn("The user is asking me to create a txt file.", content)
        self.assertNotIn("<tool_call", content)

    def test_completion_response_tool_calls(self):
        raw_output = (
            '<tool_call name="Bash">\n'
            '<parameter name="command" string="true">ls -la</parameter>\n'
            '</tool_call>'
        )
        res = completion_response("deepseek-chat", raw_output, "ls prompt")
        choice = res["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIn("tool_calls", choice["message"])
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "Bash")
        self.assertIsNone(choice["message"]["content"])

    def test_stream_chunks_tool_calls(self):
        chunks = [
            "Vou listar os arquivos.\n",
            '<tool_call name="Bash">\n',
            '<parameter name="command" string="true">ls</parameter>\n',
            '</tool_call>'
        ]
        sse_lines = list(stream_chunks("deepseek-chat", chunks))
        full_sse = "".join(sse_lines)
        self.assertIn("tool_calls", full_sse)
        self.assertIn("finish_reason\": \"tool_calls\"", full_sse)

if __name__ == "__main__":
    unittest.main()
