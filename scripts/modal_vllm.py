"""Serve Qwen3-VL-4B (+ optional LoRA) with vLLM on Modal, OpenAI-compatible.  `modal serve scripts/modal_vllm.py`
Then set verifier.base_url to the printed URL + '/v1'. Keep-warm so the demo never cold-starts."""
import modal

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("vllm>=0.10", "transformers>=4.57", "huggingface_hub", "xgrammar")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
app = modal.App("sentinel-vlm")
vol = modal.Volume.from_name("sentinel-models", create_if_missing=True)   # put LoRA adapter here


@app.function(image=image, gpu="L4", timeout=60 * 60, scaledown_window=15 * 60, min_containers=1,
              volumes={"/models": vol})
@modal.web_server(8000, startup_timeout=15 * 60)
def serve():
    import os
    import subprocess
    cmd = ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
           "--dtype", "bfloat16", "--max-model-len", "8192", "--limit-mm-per-prompt", '{"image":8}',
           "--gpu-memory-utilization", "0.9", "--max-num-seqs", "4"]
    if os.path.isdir("/models/lora"):
        cmd += ["--enable-lora", "--lora-modules", "sentinel=/models/lora", "--max-lora-rank", "32"]
    subprocess.Popen(cmd)
