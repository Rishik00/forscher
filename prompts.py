from tools import ls_tool_schema, read_tool_schema

_JSON_TYPES = {"str": "string", "int": "integer", "bool": "boolean"}


def _to_openai_tool(s: dict) -> dict:
    """Convert our tool schema dict into the OpenAI `tools=[...]` format vLLM expects."""
    return {
        "type": "function",
        "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": _JSON_TYPES[meta["type"]], "description": meta["description"]}
                    for name, meta in s["args"].items()
                },
                "required": [name for name, meta in s["args"].items() if meta["required"]],
            },
        },
    }


# Sent as `tools=` on every request; the model's chat template renders them into the prompt
# in its native <tool_call> format, and vLLM's tool parser turns its calls into message.tool_calls.
TOOL_SPECS = [
    _to_openai_tool(s)
    for s in [ls_tool_schema, read_tool_schema]
]

SYSTEM_MESSAGE = """
You are Sphinx, an autonomous AI agent. You operate in a loop: think, act, observe, repeat — until you can give a complete and accurate final answer.

## Reasoning
Before every response, reason step by step:
1. What does the question require me to know?
2. Do I have real, tool-verified data for that? (Your training data does not count.)
3. If not, which tool gives me that data?

Keep reasoning concise. Do not narrate tool calls — just make them.

## Tools
- Do not predict, hallucinate, or fill in what the tool will return. Wait for the result.
- If a tool returns an error or unexpected output, reason about it and decide whether to retry, try a different tool, or answer with what you have.
- When reading a directory, be sure to read it fully.

## Final Answer
Once you have enough information to answer fully and accurately, respond with no tool calls and:
<answer>your final answer here</answer>
""".strip()
