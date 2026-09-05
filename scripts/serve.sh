#!/usr/bin/env bash
# Start the verifier VLM behind an OpenAI-compatible endpoint. Pick ONE backend.
# All of them are reachable by sentinel via configs/default.yaml -> verifier.base_url
set -euo pipefail
BACKEND="${1:-vllm}"
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
LORA="${LORA:-}"                       # e.g. output/qwen3vl4b-sentinel-lora  (vLLM only)
PORT="${PORT:-8000}"

case "$BACKEND" in
  vllm)
    # T4 (16 GB, no bf16): --dtype half. L4/A10G: bf16 is fine.
    ARGS=(--model "$MODEL" --port "$PORT" --dtype half --max-model-len 8192
          --limit-mm-per-prompt '{"image":8}' --gpu-memory-utilization 0.90 --max-num-seqs 4
          --guided-decoding-backend xgrammar)
    if [[ -n "$LORA" ]]; then
      ARGS+=(--enable-lora --lora-modules "sentinel=$LORA" --max-lora-rank 32)
      echo ">> use verifier.model: sentinel  in configs/default.yaml"
    fi
    exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
    ;;
  llamacpp)
    # Edge / Jetson demo: GGUF Q4_K_M + mmproj. `pip install llama-cpp-python` or use the llama-server binary.
    GGUF="${GGUF:-models/Qwen3-VL-4B-Instruct-Q4_K_M.gguf}"
    MMPROJ="${MMPROJ:-models/mmproj-Qwen3-VL-4B-Instruct-f16.gguf}"
    exec llama-server -m "$GGUF" --mmproj "$MMPROJ" --port "$PORT" -ngl 99 -c 8192 --parallel 2
    ;;
  modal)
    exec modal serve scripts/modal_vllm.py
    ;;
  *) echo "unknown backend $BACKEND (vllm|llamacpp|modal)"; exit 1;;
esac
