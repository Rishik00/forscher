# forscher

*German for "explorer".* A small, hand-written agent loop that explores a codebase with an LLM and reports what the project does, its main components and where they live. No agent framework: call the model, run its tools, send the results back, repeat. See [PLAN.md](PLAN.md) for where this is going.

The default model is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), served by vLLM on a Modal GPU. Any OpenAI-compatible endpoint with tool calling works too.

## Install

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install openai python-dotenv
pip install modal                # only if you serve the model on Modal
```

## Run with Modal

1. Log in once: `modal setup`.
2. Start the server (it stops when you Ctrl-C it):
   ```bash
   modal serve modal_qwen.py
   ```
   It prints a URL. Containers start on the first request: expect ~2–3 min the first time in a while, and ~5 min the very first time while the weights download into a Modal volume. The GPU (L40S, ~$2/hr) stays up for 15 min after the last request, so stop the server when you're done.
3. Create `.env` next to `loop.py`:
   ```
   BASE_URL=https://<your-workspace>--forscher-qwen-serve-dev.modal.run/v1
   ```
4. Run the agent on a codebase:
   ```bash
   python loop.py --dir ../some-repo
   ```

For a server that outlives your terminal, use `modal deploy modal_qwen.py` instead of `modal serve` (the URL changes; update `.env`).

## Run without Modal

`loop.py` only needs an OpenAI-compatible `/v1` endpoint that supports `tools=`. Set these in `.env`:

| Variable | Meaning | Default |
|---|---|---|
| `BASE_URL` | Endpoint, ending in `/v1` | required |
| `MODEL_NAME` | Model name to send in requests | `qwen` |
| `API_KEY` | Bearer token, if the endpoint needs one | `EMPTY` |

**Your own GPU with vLLM** (needs ~24 GB of VRAM for the 9B in bf16):

```bash
pip install vllm
vllm serve Qwen/Qwen3.5-9B --served-model-name qwen --max-model-len 32768 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```
```
BASE_URL=http://localhost:8000/v1
```

**A hosted API** such as OpenRouter:

```
BASE_URL=https://openrouter.ai/api/v1
API_KEY=sk-or-...
MODEL_NAME=qwen/qwen3.5-9b   # check the exact id on openrouter.ai/models
```

Then run `python loop.py --dir ../some-repo` as above.

## Options

```bash
python loop.py --dir ../safetensors              # codebase to explore (default: current directory)
python loop.py --dir ../safetensors --tag no-readme   # suffix for the trace filename
python loop.py --prompt "..."                    # replace the default task
```

The agent sees `--dir` as `.`. Its tools are `ls_tool` and `read_tool` (300 lines or 16k characters per call, with `start_line` for more). Lockfiles, READMEs and `.git`/`.venv`/`node_modules`-style folders are hidden from it, so it has to read code rather than docs.

## Traces

Every run writes `runs/<HHMMSS>-<YYYYMMDD>-<model>[-<tag>].jsonl`, e.g. `runs/215625-20261014-Qwen3.5-9B.jsonl`, and prints a one-line summary of each step as it goes. One JSON object per line:

| Event | Fields |
|---|---|
| `run_start` | model, endpoint, directory, system prompt, tool schemas, prompt |
| `model_turn` | latency, tokens in/out, finish reason, thinking, content, tool calls |
| `tool_call` / `tool_result` | tool name, arguments, full result |
| `parse_error` | what was wrong with the model's reply |
| `answer` | the final answer |
| `run_end` | stop reason, totals, files read, files the answer names but never read |

`runs/` is git-ignored: traces include your endpoint URL.

## Files

| File | What it does |
|---|---|
| `loop.py` | The agent loop and trace logging |
| `prompts.py` | System prompt and tool schemas for the API |
| `tools/` | The agent's tools and the hidden-file rules |
| `config.py` | Dataclasses for parsed tool calls and answers |
| `modal_qwen.py` | vLLM server for Modal |
