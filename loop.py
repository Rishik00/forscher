"""forscher: a hand-written agent loop that explores a codebase and reports on it.

Run:   python loop.py --dir ../some-repo [--tag note]
Trace: runs/<HHMMSS>-<YYYYMMDD>-<model>[-<tag>].jsonl, one JSON event per line, written as it happens.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import tools
from config import Answer, ToolCallResult
from prompts import SYSTEM_MESSAGE, TOOL_SPECS

# Tool calls arrive parsed in message.tool_calls (vLLM --tool-call-parser); only the answer is ours.
# Greedy: the answer itself may quote "<answer>...</answer>", so run to the last closing tag.
ANSWER_PATTERN = re.compile(r"<answer>(.*)</answer>", re.DOTALL)
FILE_MENTION = re.compile(r"[\w./\\-]+\.(?:py|md|txt|toml|ya?ml|json|cfg|ini|sh|rs)\b")
DOTENV_PATH = Path(__file__).with_name(".env")
RUNS_DIR = Path(__file__).with_name("runs")
MAX_ITERS: int = 50
MAX_TOKENS: int = 8192  # per turn; thinking counts against this

load_dotenv(DOTENV_PATH)

# Any OpenAI-compatible endpoint: the Modal vLLM server by default, OpenRouter etc. by swapping .env.
MODEL_NAME: str = os.environ.get("MODEL_NAME", "qwen")

TOOLS = {
    "ls_tool": tools.ls_tool,
    "read_tool": tools.read_tool,
}


# ---------------------------------------------------------------- logging


class Trace:
    """Appends events to a JSONL file and prints a one-line summary of each to the console.

    Every line is flushed, so a crashed run still leaves its trace behind.
    """

    def __init__(self, model: str, tag: str | None = None, runs_dir: Path = RUNS_DIR):
        runs_dir.mkdir(exist_ok=True)
        now = datetime.now()
        name = f"{now:%H%M%S}-{now:%Y%m%d}-{_slug(model)}" + (f"-{_slug(tag)}" if tag else "")
        self.path = runs_dir / f"{name}.jsonl"
        self._f = self.path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()

    def log(self, event: str, **fields) -> None:
        record = {"t": round(time.monotonic() - self._t0, 2), "event": event, **fields}
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._f.flush()
        line = _summary(record)
        if line:
            print(line, flush=True)

    def close(self) -> None:
        self._f.close()


def _slug(text: str) -> str:
    """'Qwen/Qwen3.5-9B' -> 'Qwen3.5-9B': safe for a filename."""
    return re.sub(r"[^\w.-]+", "-", text.split("/")[-1]).strip("-")


def _short(text, n: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _summary(r: dict) -> str | None:
    """Console line for a trace record; None to keep it out of the console."""
    e = r["event"]
    if e == "run_start":
        return f"model={r['model']}  root={r['root']}\ntrace={r['trace']}"
    if e == "model_turn":
        head = (
            f"[{r['step']}] {r['latency_s']}s  in={r['prompt_tokens']} out={r['completion_tokens']}"
            f"  {r['finish_reason']}"
        )
        return head + (f"\n    think: {_short(r['thinking'])}" if r.get("thinking") else "")
    if e == "tool_call":
        return f"    -> {r['name']}({json.dumps(r['args'], ensure_ascii=False)})"
    if e == "tool_result":
        result = r["result"]
        return f"    <- {_short(result)}" if result.startswith("Error") else f"    <- {len(result)} chars"
    if e == "parse_error":
        return f"    !! {r['error']}"
    if e == "run_end":
        return (
            f"\n{r['stop_reason']}: {r['turns']} turns, {r['prompt_tokens']} in / "
            f"{r['completion_tokens']} out tokens, {len(r['files_read'])} files read"
            + (f"\nnamed but never read: {', '.join(r['unread_mentions'])}" if r["unread_mentions"] else "")
        )
    return None


def _norm(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def unread_mentions(answer: str, files_read: set[str]) -> list[str]:
    """File paths the answer talks about that the agent never successfully read."""
    read = {_norm(p) for p in files_read}
    mentioned = {_norm(m) for m in FILE_MENTION.findall(answer)}
    # "read_tool.py" counts as read if "tools/read_tool.py" was read.
    return sorted(m for m in mentioned if not any(r == m or r.endswith("/" + m) for r in read))


# ---------------------------------------------------------------- agent loop


def parse_result(message) -> list[ToolCallResult] | Answer:
    """Turn a chat completion message into its tool calls (there can be several) or an Answer."""
    if message.tool_calls:
        return [
            ToolCallResult(
                function_name=tc.function.name,
                args=json.loads(tc.function.arguments or "{}"),
                call_id=tc.id,
            )
            for tc in message.tool_calls
        ]

    # Thinking text can mention tags; only parse what comes after it.
    response = (message.content or "").split("</think>")[-1]
    answer_match = ANSWER_PATTERN.search(response)
    if answer_match:
        return Answer(final_answer=answer_match.group(1).strip())

    msg = "Response did not contain a tool call or answer tag."
    raise ValueError(msg)


def _get_base_url() -> str:
    """Return the model server URL from the environment, or exit with an error."""
    base_url: str | None = os.environ.get("BASE_URL")

    if base_url:
        return base_url

    msg = (
        f"Missing BASE_URL. Set it in {DOTENV_PATH} to the Modal URL + /v1 "
        "(printed by `modal serve modal_qwen.py`)."
    )
    raise SystemExit(msg)


def _model_id(client: OpenAI) -> str:
    """The real model behind MODEL_NAME: vLLM reports it as `root` (e.g. Qwen/Qwen3.5-9B)."""
    try:
        for m in client.models.list().data:
            if m.id == MODEL_NAME:
                return getattr(m, "root", None) or m.id
    except Exception:
        pass
    return MODEL_NAME


def _run_tool(call: ToolCallResult) -> str:
    """Run a tool call, returning errors as text so the model can recover."""
    func = TOOLS.get(call.function_name)
    if func is None:
        return f"Error: unknown tool {call.function_name!r}. Available: {', '.join(TOOLS)}"
    try:
        return str(func(**(call.args or {})))
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def run(prompt: str, tag: str | None = None) -> None:
    """Run the agent loop for a given prompt, tracing every step."""
    client = OpenAI(
        base_url=_get_base_url(),
        api_key=os.environ.get("API_KEY", "EMPTY"),  # vLLM ignores it unless started with --api-key
        timeout=15 * 60,  # first request waits out the cold start
    )
    print(f"Connecting to {client.base_url} (a cold Modal server takes a few minutes)...", flush=True)
    model = _model_id(client)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    trace = Trace(model, tag)
    files_read: set[str] = set()
    stats = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0}
    final_answer = ""
    stop_reason = "max_iters"
    trace.log(
        "run_start",
        model=model,
        served_as=MODEL_NAME,
        base_url=str(client.base_url),
        root=os.getcwd(),
        trace=str(trace.path),
        max_iters=MAX_ITERS,
        max_tokens=MAX_TOKENS,
        system_prompt=SYSTEM_MESSAGE,
        tools=TOOL_SPECS,
        prompt=prompt,
    )

    try:
        final_answer = _loop(client, messages, trace, files_read, stats)
        if final_answer:
            stop_reason = "answer"
    except BaseException as e:
        stop_reason = f"crash: {type(e).__name__}: {e}"
        raise
    finally:
        trace.log(
            "run_end",
            stop_reason=stop_reason,
            **stats,
            files_read=sorted(files_read),
            unread_mentions=unread_mentions(final_answer, files_read),
            # An answer naming no files at all is a red flag for an exploration task.
            answer_mentions_files=bool(FILE_MENTION.search(final_answer)),
        )
        trace.close()

    if final_answer:
        print("\n" + final_answer)
    print(f"\nTrace: {trace.path}")


def _loop(client, messages, trace, files_read, stats) -> str:
    """The agent loop proper: model turn → parse → tool or answer. Returns the answer, or ""."""
    for step in range(MAX_ITERS):
        started = time.monotonic()
        completion = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=TOOL_SPECS, max_tokens=MAX_TOKENS
        )
        choice = completion.choices[0]
        response = choice.message.content or ""
        raw_calls = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in choice.message.tool_calls or []
        ]

        usage = completion.usage
        if usage:
            stats["prompt_tokens"] += usage.prompt_tokens
            stats["completion_tokens"] += usage.completion_tokens
        stats["turns"] = step + 1
        trace.log(
            "model_turn",
            step=step,
            latency_s=round(time.monotonic() - started, 2),
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            # vLLM's reasoning parser puts the thinking here (field name varies by version).
            thinking=getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None),
            content=response,
            tool_calls=raw_calls,
        )

        try:
            parse = parse_result(choice.message)
        except (ValueError, json.JSONDecodeError) as e:
            # Malformed output: tell the model exactly what to fix instead of crashing.
            # A generic "reply with a tool call or answer" made Qwen forget the task, so be specific.
            trace.log("parse_error", step=step, error=f"{type(e).__name__}: {e}")
            messages.append({"role": "assistant", "content": response})
            if isinstance(e, json.JSONDecodeError):
                nudge = f"One of your tool calls had invalid arguments ({e}). Resend your calls."
            else:
                nudge = (
                    "Your reply had no tool call and no <answer> tags. If it was your final "
                    "answer, send the same text again wrapped in <answer></answer>. "
                    "Otherwise, make the tool call you need."
                )
            messages.append({"role": "user", "content": nudge})
            continue

        if isinstance(parse, Answer):
            trace.log("answer", step=step, answer=parse.final_answer)
            return parse.final_answer

        assistant_msg = {"role": "assistant", "content": response, "tool_calls": raw_calls}
        messages.append(assistant_msg)
        # Run every call in the reply; each result goes back as its own `tool` message.
        for call in parse:
            trace.log("tool_call", step=step, name=call.function_name, args=call.args)
            result = _run_tool(call)
            trace.log("tool_result", step=step, name=call.function_name, result=result)
            if call.function_name == "read_tool" and not result.startswith("Error"):
                files_read.add(str((call.args or {}).get("file_path", "")))
            messages.append({"role": "tool", "tool_call_id": call.call_id, "content": result})

    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Codebase to explore; tools see it as '.'.")
    parser.add_argument("--tag", help="Optional suffix for the trace filename, e.g. 'no-readme'.")
    parser.add_argument(
        "--prompt",
        default=(
            "Explore the codebase in the current directory ('.'). It may be too big to read "
            "everything: use ls_tool to see the layout, then read_tool only on the key files. "
            "For every component you report, list its directory and read at least one of its "
            "source files before answering. Only "
            "name paths you have seen in an ls_tool or read_tool result. Answer with what the "
            "project does, its main components, and the file paths where each one lives."
        ),
    )
    args = parser.parse_args()

    os.chdir(args.dir)  # traces and .env use absolute paths, so this only moves the tools
    run(args.prompt, args.tag)
