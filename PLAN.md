# forscher

A small harness for exploring codebases with LLMs. The model gets only a few read-only tools and a budget, crawls a repo, and reports what matters. Its report is scored against an answer key written by a human who studied the repo.

**Question:** how far can a cheap model get with a good harness?

## Design

- **Loop:** written by hand, no agent framework. Call the model, run its tools, append the results, repeat. Stop when the model calls `submit_report` or runs out of budget, with a warning when the budget is nearly gone.
- **Tools:**
  - `read(path, offset, limit)`: numbered lines, capped output
  - `bash(cmd)`: read-only commands (`ls`, `rg`, `find`, `wc`), truncated output
  - `submit_report(key_files, summary)`: ranked files with a one-line reason each, plus an architecture summary
- **Sandbox:** Docker with the repo mounted read-only. If Docker isn't available, fall back to an allowlist of commands.
- **Providers:** two adapters.
  - Anthropic SDK for Claude
  - OpenAI-compatible API for vLLM on Modal, DeepSeek, Mistral and similar
- **Logging:** one JSONL trace per run with every call, its arguments, output size, tokens, cost and time.
- **Cost guard:** compute dollars from `usage` after every call and abort the run past a cap.

## Scoring

- **Answer key:** `truth/<repo>.yaml` holds my top 10 files (ranked) and a short architecture summary. Written **before** looking at any model output.
- **Main metric:** how many of my top 10 the model's report contains, at tool-call budgets of 5 / 10 / 20.
- **Secondary metrics:** calls until the first correct file, total tokens, cost, failed tool calls.
- **Summary quality:** compared by hand for now.

## Things to vary (one at a time)

- Model
- Tools: read only, or read + bash
- Starting context: no directory tree, or a tree up front
- Prompt: whether the model must plan before exploring
- Truncation size
- Effort level, for models that have one

## Models

- **Development:** a scripted fake model that replays fixed tool calls. Free.
- **Main runs:** open-weights models on Modal via vLLM, paid with the $30/month free credits. Candidates: Qwen3.6-35B-A3B, GLM-4.7-Flash, Gemma 4, gpt-oss-20b.
- **API comparison:** Claude Haiku 5.5 (`claude-haiku-5-5`, $0.10 / $0.50 per million tokens), plus maybe one other cheap API.

## Layout

```
forscher/
  loop.py        # agent loop
  tools.py       # read, bash, submit_report
  providers.py   # anthropic + openai-compatible adapters, scripted fake
  eval.py        # score a report against truth/
  truth/         # answer keys
  runs/          # JSONL traces
```

## Milestones

1. [ ] Write `truth/safetensors.yaml` by hand
2. [ ] Loop, tools, logging and scoring working against the scripted model ($0)
3. [ ] vLLM server on Modal, first real run on `safetensors`, read the trace (~free)
4. [ ] Budget sweep plus 2–3 harness variants (~$2–5 or credits)
5. [ ] Haiku 5.5 comparison run
6. [ ] Blog post: chart plus trace excerpts
7. [ ] Repeat on more repos from `project_index_everything`

## Open questions

- Is Docker installed?
- Is a Modal workspace set up for this?
