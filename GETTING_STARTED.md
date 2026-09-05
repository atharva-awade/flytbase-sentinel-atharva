# Getting started — exact steps (for someone new to VLMs)

You have three machines that matter today. Use each for what it is good at:

| Machine | Use it for | Don't use it for |
|---|---|---|
| **Your laptop (RTX 4050, 6 GB VRAM)** | Running the SENTINEL app + gate + a **4-bit** verifier for the live demo; all code; the arena uploads | Fine-tuning (6 GB is not enough) |
| **Kaggle (2× T4, 16 GB each, free)** | LoRA fine-tuning; building the normal bank; batch-scoring the 28 private videos with fp16 vLLM | The live demo (no webcam, notebook UI) |
| **Modal (L4/A10G, $30 free)** | Optional: a warm vLLM endpoint for the demo if the laptop VLM is too slow | Training (Kaggle is free) |

---

## 0. Is the RTX 4050 laptop enough? — Yes, with the 4-bit route

- **SigLIP2 gate:** ~0.4 GB. Runs at a few ms/frame on the 4050. ✔
- **Qwen3-VL-4B fp16 (~9 GB)** does **not** fit in 6 GB. ✘
- **Qwen3-VL-4B Q4_K_M GGUF (~3.3 GB + ~0.6 GB mmproj)** via **llama.cpp** fits with room for images. ✔ Expect ~2–4 s per 6-frame verification. This *is* the "edge deployability" story — say so proudly on stage.
- Fallback if even that is tight: `Qwen3-VL-2B-Instruct` GGUF (~1.7 GB).

### 0.1 Laptop setup (Windows or Linux, ~20 min)

```bash
# 1) clone the hub
git clone https://github.com/atharva-awade/flytbase-sential-atharva.git sentinel && cd sentinel

# 2) python env (3.10–3.12)
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu124     # CUDA build for the 4050
pip install "transformers>=4.57" accelerate

# 3) sanity: everything green, app works with mock models
python -m pytest -q
SENTINEL_MOCK=1 uvicorn app.server:app --port 8080      # Windows PowerShell:  $env:SENTINEL_MOCK=1; uvicorn app.server:app --port 8080
# open http://localhost:8080 → Samples → sample_fire_event.mp4 → Start monitoring
```

### 0.2 Real verifier on the laptop (llama.cpp, 4-bit)

```bash
# 4) llama.cpp server (prebuilt binaries: https://github.com/ggml-org/llama.cpp/releases — pick the CUDA build)
#    models: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF  (download the Q4_K_M .gguf and the mmproj-*.gguf)
mkdir -p models && mv ~/Downloads/Qwen3-VL-4B-Instruct-Q4_K_M.gguf ~/Downloads/mmproj-*.gguf models/
llama-server -m models/Qwen3-VL-4B-Instruct-Q4_K_M.gguf --mmproj models/mmproj-Qwen3-VL-4B-Instruct-f16.gguf \
             --port 8000 -ngl 99 -c 8192 --parallel 2
#    (or: `ollama run qwen3-vl:4b-instruct` and set base_url to http://localhost:11434/v1, model qwen3-vl:4b-instruct)

# 5) tell SENTINEL where the verifier lives — configs/default.yaml
#    verifier.backend: openai
#    verifier.base_url: http://localhost:8000/v1
#    verifier.model: Qwen3-VL-4B-Instruct          (llama.cpp ignores the name; ollama needs the exact tag)
#    verifier.self_consistency: false             (halves VLM calls on the 6 GB card; turn back on for the arena run)
#    gate.device: cuda

# 6) run the real app
uvicorn app.server:app --port 8080
```

First start downloads SigLIP2 (~400 MB) from Hugging Face; if the machine is offline, run it once beforehand.

---

## 1. Connect the dataset

Download the pack (one of the five mirrors in the hackathon doc) and unzip so you have:

```
data/train/<class_name>/videos/*.mp4  + videos.csv + ground_truth.csv
data/test/videos/*.mp4                + videos.csv + ground_truth.csv     (34 public videos)
```

Then, on any GPU machine:

```bash
python -m sentinel.cli derive-priors      --train-dir data/train         # per-class median durations → configs/class_priors.yaml
python -m sentinel.cli build-normal-bank  --train-dir data/train         # SigLIP2 normal memory → banks/normal.npy
python -m sentinel.cli run --gt data/test/ground_truth.csv --videos data/test/videos --out submissions/public.json
python -m sentinel.cli sweep --gt data/test/ground_truth.csv --videos data/test/videos --trace-from submissions/public.trace.json
cp configs/class_priors.sweep.yaml configs/class_priors.yaml               # adopt the tuned thresholds
```

---

## 2. Kaggle, step by step (fine-tuning + heavy runs)

1. **Account:** kaggle.com → sign up → *Settings → Phone verification* (GPU stays locked until verified).
2. **Dataset:** *Datasets → New Dataset → upload* the unzipped `train/` and `test/` folders (or add via Google Drive link tools if you prefer). Name it `ahc-vad`. This takes a while — start it first.
3. **Notebook:** *Code → New Notebook*. Right sidebar → *Session options*: Accelerator **GPU T4 x2**, Internet **On**, Persistence **Files only**. Add the `ahc-vad` dataset with *Add Input*.
4. **Cell 1 — code + deps**
   ```python
   !git clone https://github.com/atharva-awade/flytbase-sential-atharva.git /kaggle/working/sentinel
   %cd /kaggle/working/sentinel
   !pip install -q -e . "transformers>=4.57" accelerate
   !pip install -q "unsloth[colab-new]" trl peft bitsandbytes
   import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())
   !ln -s /kaggle/input/ahc-vad data
   ```
5. **Cell 2 — SFT dataset** (CPU-bound, ~10–20 min for the whole train set; use `--max-per-class 150` if time is short)
   ```python
   !python scripts/make_sft_dataset.py --train-dir data/train --out /kaggle/working/sft --max-per-class 200
   ```
6. **Cell 3 — LoRA** (T4 = fp16, 4-bit; ~40–60 min for ~2,000 samples, 1 epoch)
   ```python
   !python scripts/train_lora_unsloth.py --data /kaggle/working/sft --out /kaggle/working/lora --epochs 1 --max-steps 400
   ```
   Checkpoints land in `/kaggle/working/lora/ckpt` every 100 steps; if the session dies, resume from the last one.
7. **Cell 4 — does it help?** Serve the base model with vLLM on the *other* T4 and score the public set, then repeat with `LORA=/kaggle/working/lora`:
   ```python
   import subprocess, os
   srv = subprocess.Popen("CUDA_VISIBLE_DEVICES=1 bash scripts/serve.sh vllm", shell=True)   # wait ~3 min, watch for 'Uvicorn running'
   !python -m sentinel.cli run --gt data/test/ground_truth.csv --videos data/test/videos --out submissions/public_base.json
   ```
   Then `LORA=/kaggle/working/lora bash scripts/serve.sh vllm` (set `verifier.model: sentinel` in the config) and run again to `public_lora.json`. Keep whichever scores higher on the local scorer — **that is the adapter you ship**.
8. **Export for the laptop:** `--export-gguf` in the trainer writes a Q4_K_M GGUF of the merged model → download from *Output* and point llama.cpp at it.
9. **Private run:** put `manifest.json` (downloaded from the arena) and the private videos in the dataset or upload them; run
   `python -m sentinel.cli run --manifest data/manifest.json --videos data/private --levels 1 --out submissions/L1.json`
   then `submit-check`, download the JSON from *Output*, upload in the arena. Repeat for `--levels 2`, `--levels 3`.

Kaggle gives 30 GPU-hours/week; a full day of T4×2 use is ~16 h — fine.

---

## 3. The day, in order

| When | What | Command |
|---|---|---|
| now | app works in mock; repo cloned on laptop & Kaggle | §0.1 |
| data arrives | priors, normal bank, public run, sweep | §1 |
| immediately after | **upload L1** | `make run-private LEVELS=1 OUT=submissions/L1.json && make check FILE=submissions/L1.json` |
| in parallel (Kaggle) | SFT set → LoRA → compare | §2 steps 5–7 |
| then | L2, L3 uploads (re-run only changed videos if the LoRA wins) | `make run-private LEVELS=2 …`, `LEVELS=3 …` |
| last 90 min | demo on the laptop with the real 4-bit verifier; write-up; final submission section | `uvicorn app.server:app` |

**Rule of thumb for uploads:** never upload a sheet `submit-check` refused; run `mark-uploaded` after every accepted upload so the tool knows what the arena holds.

---

## 4. Demo checklist (laptop)

- `llama-server` running (`curl localhost:8000/v1/models` answers).
- `uvicorn app.server:app --port 8080` running with `SENTINEL_MOCK` **unset**.
- Three clips ready in `data/samples/`: a normal one (silence), an accident/fire one (alert + action card), a long one (two events, one interval each).
- Webcam tab: point the camera at a phone playing drone footage — a live source, no file.
- Press **Mark scene normal** on a new location before showing the stopped-car-in-parking-bay vs. highway-shoulder contrast.
- Keep the browser at 125 % zoom on the projector; the UI is designed for 1440–1560 px wide.
