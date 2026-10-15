"""Serve Qwen on a Modal GPU as an OpenAI-compatible vLLM server.

Dev (stops when you Ctrl-C):  modal serve modal_qwen.py
Long-lived:                   modal deploy modal_qwen.py

Both print a URL; put `<url>/v1` in .env as BASE_URL for loop.py.
The container stays warm for SCALEDOWN after the last request, so only the
first call of a session pays the ~3 min startup.
"""

import subprocess

import modal

MODEL = "Qwen/Qwen3.5-9B"  # official; was the community distill empero-ai/Qwen3.8-9B-Distill
SERVED_NAME = "qwen"  # what clients pass as `model=`
PORT = 8000
SCALEDOWN = 15 * 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm")
    # FlashInfer's sampler JIT-compiles with nvcc, which debian_slim doesn't have.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("forscher-qwen")


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/root/.cache/huggingface": hf_cache,  # weights download once, then cached
        "/root/.cache/vllm": vllm_cache,  # torch.compile artifacts, reused across runs
    },
    scaledown_window=SCALEDOWN,
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", "32768",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',  # text only: skip vision encoder
        # Moves the thinking trace into `reasoning_content`, so `content` is just the reply.
        "--reasoning-parser", "qwen3",
        # Parses Qwen's native <tool_call><function=...> XML into message.tool_calls.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    subprocess.Popen(cmd)
