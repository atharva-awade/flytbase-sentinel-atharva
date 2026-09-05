# SENTINEL — Build Blueprint
## AHC / FlytBase Visual Intelligence Hackathon · Real-Time Drone Video Anomaly Detection · 05 Sep 2026

> **Who this is for:** any engineer or coding agent building this from zero. Read top to bottom once, then use §7 (repo layout) and §13 (execution order) as the working checklist. Every design decision below is traced back to a line in the problem statement or the scoring rules, so if you have to cut scope you know exactly what you are giving up.

---

## 0. Thirty-second summary

We are building **SENTINEL**: a *cascaded* video anomaly detector where a cheap, always-on perception stage watches every frame and a small vision-language model (Qwen3-VL-4B, LoRA-tuned) is *woken only when something looks off*. A temporal decoder turns noisy per-second scores into **one clean interval per real event**, biased toward silence, because the scoring punishes false alarms harder than misses. Every detected event ships with a one-sentence explanation and a suggested responder action.

The three numbers that drive every choice:

| Fact (from the rules / live arena) | Consequence for design |
|---|---|
| L1 = 25 pts, L2 = 35, L3 = 40, Speed +5, Reasoning +5 | **75% of points are temporal.** Timing quality, not clip labels, wins. |
| Normal L2/L3 video + any prediction = **0** for that video | Default to *silence*. Only fire when confident. Calibrated abstention is a feature. |
| Match requires class correct **and** IoU ≥ 0.5; fragments hurt | Never emit fragments. Merge aggressively. One interval per event. |

---

## 1. Understanding the problem (what they actually want)

### 1.1 The task in one paragraph
A drone flies over highways, streets, parks, stations, terminals, gatherings and utility sites — day and night. We receive its video and must decide, while it is still overhead, whether one of eleven *situations* is happening, and for the harder tiers, *when* it starts and ends. Object detection is explicitly declared insufficient: "a stationary car is unremarkable in a parking bay and a problem on a highway shoulder." The anomaly is a **relation between objects, place and time**, which is why they push toward vision-language models. But large VLMs are too slow and costly per drone feed, so the hackathon question is literally: *can a small VLM do this reliably in real time on limited GPU?*

### 1.2 What the organisers reward (reading between the lines)
- **Runtime independence from hosted giants.** "Larger hosted models can be used during development, for comparison, or to generate training data, but they cannot be part of what makes the detector work at runtime." → Any pipeline calling NIM/Gemini/GPT at inference time is *against the rules* — not just unimpressive. This settles the API-vs-local question (§3).
- **False alarms matter as much as misses.** Stated in the PS and hard-coded in scoring (normal L2/L3 → 0).
- **Events have different temporal shapes.** "An accident is over in about a second, congestion builds gradually, a stopped vehicle only becomes an anomaly once it has been stationary for some time." → per-class temporal priors (§5.5) are not a nicety; they are the difference between IoU 0.3 and 0.6.
- **Open set.** "This is not a fixed list. Participants are welcome to cover other events." → A VLM-based design can claim open-vocabulary coverage; make it demoable (§10).
- **Economy across many drones.** They care about *cost per feed*. Report a Real-Time Factor (RTF) and a "VLM wake ratio" (§10) — show that we watch 100% of the video with the cheap stage and spend the expensive model on ~10–20%.

### 1.3 The eleven classes, grouped by temporal shape (this grouping drives the decoder)
| Shape | Classes | Typical duration | Decoder behaviour |
|---|---|---|---|
| **Impulse** | `traffic_accident`, `fighting_or_violence` (onset) | 1–10 s onset, consequences linger | Detect onset sharply; extend end to include immediate aftermath but cap length |
| **Ramp / state** | `traffic_congestion`, `waterlogging_or_flood`, `smoke`, `fire` | tens of seconds → whole clip | Hysteresis with long hold; merge gaps generously |
| **Dwell** | `stalled_or_broken_down_vehicle`, `vehicle_blocking_traffic`, `loitering_or_suspicious_presence` | Becomes anomalous *after* N seconds stationary | Require persistence before start; start time = when stationarity became evident, not first frame |
| **Trajectory** | `wrong_way_driving` | 2–15 s | Motion-direction cue (§5.2) + VLM confirm |
| **Static condition** | `road_spill_or_debris` | Often entire clip | Treat as state; if present from frame 0, start = 0 |

### 1.4 The data we have
- `train/<class>/videos/*.mp4` + `videos.csv` + `ground_truth.csv` (with `description_summary` sometimes filled) — 12 folders including `normal`. Sources: CCTV, dashcam, drone; day/night/weather.
- `test/` — **34 public videos, ~56 min, with ground truth.** This is our calibration and regression set. Never tune thresholds on the private manifest; tune here.
- Private manifest: 28 videos across 3 levels (`manifest.json` from the arena).
- Unannotated drone footage (urban, night) — pseudo-label with a big model during development (allowed) to distill.

---

## 2. How the arena scores us (operational rules we must engineer around)

1. **Latest upload wins; no best-of.** → Never upload without passing the local scorer gate (§8.3). Keep a `submissions/` history with local scores so we can re-upload the best sheet at the end.
2. **Partial uploads update only mentioned videos.** → Submit per level: L1 batch first (fast, 25 pts locked in), then L2, then L3. Re-submit individual videos when improved.
3. **Unanswered = normal.** → For L2/L3 videos where we are unsure, *omit or send `events: []`*. This is a free, rule-sanctioned abstention.
4. **Rejected file ≠ used run**, and the error table names the field. → Still validate locally first (§8.1); rejections waste minutes we don't have.
5. **`runtime_metadata` required per video; latency bonus = total reported processing time ÷ total video duration** (RTF). → Instrument honestly (§6.4). Speed bonus is only 5 pts — never trade accuracy for it, but a cascade gets it almost for free.
6. **Reasoning bonus (+5) via optional `explanation` 20–500 chars.** → Generate one for every event. Free points that most teams skip.
7. **L1: one label per clip; repeating a class earns nothing.** → Emit exactly one event with `null` timestamps, or `[]`.
8. **Class `normal` must be sent as `events: []`.**
9. **`average_time_ms` must equal `total/count` within 2%; `call_times_ms` length must equal `call_count`.** → Compute, never hand-type.

---

## 3. The strategic decision: fine-tune a small VLM vs. hosted VLM API

### 3.1 Verdict
**Hosted APIs (NVIDIA NIM, Gemini Flash, the gpt-5.6-luna key) are development-time tools only. Runtime must be a local small model.** This is dictated by the rules, and it is also what wins: judges from FlytBase build drone systems; a demo that dies without Wi-Fi is not a drone product.

### 3.2 Where each option is used
| Component | Choice | Why |
|---|---|---|
| **Runtime verifier VLM** | `Qwen/Qwen3-VL-4B-Instruct` + LoRA (ours). Fallback: `Qwen3-VL-2B-Instruct` for the "edge" story | Best ≤4B open VLM with native multi-frame/video input, Apache-2.0, vLLM + llama.cpp/GGUF support, Unsloth fine-tune path. Competitors on the live board are already on it — we must out-*system* them, not out-model them. |
| **Runtime always-on gate** | SigLIP2-B/16 (image-text) + frame-difference motion energy + embedding-drift stationarity | ~5 ms/frame on T4, INT8/TensorRT-able for Jetson. Gives zero-shot class scores *and* a normality distance. |
| **Teacher / labeler (dev only)** | NIM `cosmos-reason2` or Gemini 2.5 Flash (video-native, free tier) | Pseudo-label unannotated drone footage; write `description_summary`-style captions for distillation; generate explanations for training. 40 RPM NIM / free-tier Gemini is plenty for a few hundred clips. |
| **Question-bank author (dev only)** | Any strong LLM | Generate the fine-grained yes/no question bank (§5.3) per class per viewpoint. |

### 3.3 Should we fine-tune at all in one day?
**Yes — but as a parallel track, never as the critical path.** The zero-shot ASK-HINT-style pipeline (§5.3) must be producing valid submissions by build-hour 2. LoRA runs on Kaggle T4×2 / Modal L4 in the background using `description_summary` + our JSON schema. If it beats zero-shot on the 34-video public set, swap the adapter in; if not, we still have a full system. What the LoRA buys: (a) the model learns *our output JSON* reliably, (b) drone-viewpoint vocabulary (tiny cars from above), (c) the 11-class taxonomy boundaries (`smoke` vs `fire`, `stalled` vs `blocking`).

Recipe (Unsloth, fp16 because T4 has no bf16):
```python
model, tok = FastVisionModel.from_pretrained("unsloth/Qwen3-VL-4B-Instruct", load_in_4bit=True)
model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False,
        finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0)
# SFTConfig: fp16=True, bf16=False, max_length=None (else image tokens truncate silently),
# per_device_train_batch_size=1, grad_accum=4, lr=1e-4, 1 epoch, gradient_checkpointing="unsloth"
# processor: min_pixels=128*28*28, max_pixels=384*28*28 ; 6 frames per sample as a 2x3 mosaic
# Build dataset with a list comprehension, NOT dataset.map (breaks on multi-image).
```
Training sample = 6-frame mosaic (or 6 images) from a GT window (or a normal window) → target JSON `{"anomaly": true, "class_name": "...", "explanation": "..."}`. Balance: ≥40% normal windows, including *hard normals* (dense-but-flowing traffic, parked cars in bays) so the model learns the "context" the PS talks about.

---

## 4. Architecture

```
                 ┌──────────────────────────────────────────────────────────────────────┐
  video.mp4 ───▶ │ S0  DECODE & SAMPLE   ffmpeg/PyAV · 2 fps (L1) · 1 fps (L2/L3) · 448px │
                 └──────────────┬───────────────────────────────────────────────────────┘
                                ▼  frames[t]
                 ┌──────────────────────────────────────────────────────────────────────┐
                 │ S1  ALWAYS-ON GATE  (≈5 ms/frame, runs on 100% of frames)            │
                 │  • SigLIP2 image embedding e[t]                                       │
                 │  • zero-shot class scores  s_c[t] = softmax(e·T_c)  (prompt bank)     │
                 │  • normality distance    d[t] = 1 - max cos(e[t], NormalBank)         │
                 │  • motion energy m[t] (frame diff) · stationarity σ[t] (‖e[t]-e[t-k]‖) │
                 │  → suspicion score  u[t] = w1·max_c s_c + w2·d + w3·f(m,σ)            │
                 └──────────────┬───────────────────────────────────────────────────────┘
                                ▼  candidate windows W = {[a,b] : u > τ_gate, padded ±3 s}
                 ┌──────────────────────────────────────────────────────────────────────┐
                 │ S2  VLM VERIFIER  Qwen3-VL-4B (+LoRA), only on W (≈10–25% of video)   │
                 │  • 6–8 frame mosaic per window (+ motion-mask tint on moving regions)  │
                 │  • fine-grained yes/no question bank → per-class evidence              │
                 │  • structured JSON: class, confidence, explanation, onset_frame_idx    │
                 │  • self-consistency: 2 samples; disagree → abstain (normal)            │
                 └──────────────┬───────────────────────────────────────────────────────┘
                                ▼  per-window verdicts v_w
                 ┌──────────────────────────────────────────────────────────────────────┐
                 │ S3  TEMPORAL DECODER                                                 │
                 │  • fuse u[t] & v_w → per-class curve p_c[t]                          │
                 │  • class-shape priors (impulse / ramp / dwell / trajectory / static)  │
                 │  • hysteresis (τ_on > τ_off) · min-duration · gap-merge · 1-per-event │
                 │  • video-level abstention: if max evidence < τ_video → events = []     │
                 └──────────────┬───────────────────────────────────────────────────────┘
                                ▼
                 ┌──────────────────────────────────────────────────────────────────────┐
                 │ S4  EMIT  events[] + explanation + responder action + runtime_metadata │
                 │      → submission JSON (validated) · dashboard · latency ledger        │
                 └──────────────────────────────────────────────────────────────────────┘
```

**Why a cascade (and why it is the correct answer to *their* question):** the S1 gate is what keeps the system real-time on limited GPU; the S2 VLM is what gives context-awareness that YOLO cannot; S3 is what converts model outputs into the *interval geometry* the scorer rewards. Each stage is independently testable (§8).

Level-specific behaviour:
- **L1** — whole clip is one window. S1 gives a prior over classes; S2 runs once on a 8-frame mosaic spanning the clip; output one class or `[]`.
- **L2 / L3** — full cascade. L3 differs only in length and difficulty: increase gap-merge tolerance, allow multiple events, and run S2 with a sliding 12 s window / 6 s stride inside candidate regions.

---

## 5. Component design (detailed)

### 5.1 S0 — Decoding & sampling (`sentinel/ingest.py`)
- Use PyAV (or `ffmpeg -vf fps=…,scale=-2:448`) to decode to RGB at fixed fps; record `frames_processed`.
- Keep timestamps from the container (`pts * time_base`), not `i / fps` — variable-frame-rate drone clips exist.
- L1: 2 fps capped at 16 frames total (uniform). L2/L3: 1 fps for S1; S2 re-samples 6–8 frames within each window.
- Night footage: apply CLAHE / gamma lift *only* for the S1 embedding path if mean luminance < 40; the VLM sees the original.

### 5.2 S1 — Always-on gate (`sentinel/gate.py`)
- **Model:** `google/siglip2-base-patch16-224` (or `-256`). fp16 on GPU; export ONNX → TensorRT INT8 for the Jetson story. Batch frames (B=32).
- **Prompt bank** (`configs/prompt_bank.yaml`): 6–10 text prompts per class *per viewpoint* (aerial / CCTV / dashcam), e.g. `traffic_accident`: "aerial view of a car crash with vehicles collided on a road", "overturned vehicle on a highway seen from a drone", …; `normal`: "aerial view of traffic flowing normally on a highway", "empty street at night", "cars parked in a parking lot", …. Text embeddings precomputed once → `T_c` (mean of the class's prompt embeddings, L2-normalised).
- **Normality bank:** embed ~3,000 frames sampled from `train/normal/` (+ frames outside GT intervals of other classes). Store as FAISS/NumPy matrix `N` (fp16). `d[t] = 1 − max_j cos(e[t], N_j)`.
- **Motion:** `m[t] = mean(|gray[t] − gray[t−1]|)` on a 112px downscale. **Stationarity:** `σ[t] = ‖e[t] − e[t−5]‖`; low motion + low σ + vehicle-class prompt score = "dwell" evidence (stalled/blocking). **Direction (for wrong-way):** dense optical flow (Farneback, 112px) dominant-angle histogram; a persistent sub-population moving anti-parallel to the dominant flow raises `s_wrong_way`.
- **Output:** per-second arrays `u`, `s_c`, `d`, `m`, `σ` saved to `cache/<video_id>.npz` for reuse and for the dashboard timeline.
- **Gate threshold τ_gate:** tuned on the public test set to achieve ≥95% recall of GT event seconds while sending ≤25% of seconds to S2 (see §8.2 sweep).

### 5.3 S2 — VLM verifier (`sentinel/verifier.py`)
- **Serving:** vLLM (`--max-model-len 8192 --limit-mm-per-prompt image=8 --dtype half`) on T4/L4; llama.cpp with `Qwen3-VL-4B-Instruct-Q4_K_M.gguf` + mmproj for the edge/Jetson demo. Both behind one `VLMClient` interface with an OpenAI-compatible `/v1/chat/completions` call, so swapping is a config change.
- **Input construction:** 6–8 frames from the window as separate images (preferred; native temporal encoding) or a labelled 2×3/2×4 mosaic with timestamp captions burned in (`t=12s`). Cerberus trick: tint moving regions (from frame-diff mask) with a faint red overlay in a *second* copy of the key frame so attention goes to what moved. Resize to ≤448 on the long side to hold latency at ~1.5–3 s per call on a T4.
- **Prompting — fine-grained questions (ASK-HINT style), not "is there an anomaly?":**
  System: role = drone-operations analyst; taxonomy with one-line definitions; "answer strictly in JSON".
  User: frames + `"Answer each question yes/no with a 0-1 confidence, then give a final verdict."` Questions come from `configs/question_bank.yaml`, ~2 per class filtered to the top-4 classes suggested by S1 (keeps prompt short), plus always: "Is traffic moving normally for this kind of road?" and "Is this scene ordinary for the location?" (explicit *normal* evidence).
  Output schema:
  ```json
  {"answers":[{"q":"...","yes":true,"conf":0.8}], "class_name":"traffic_accident"|"normal",
   "confidence":0.0-1.0, "onset_frame":3, "explanation":"<20-500 chars, concrete, visual>"}
  ```
  Use vLLM guided decoding (`response_format` json_schema) so parsing never fails.
- **Self-consistency:** temperature 0 answer + one temperature 0.6 answer. If classes disagree or min confidence < 0.55 → verdict `normal` for this window. This is our calibrated abstention.
- **Class disambiguation rules (post-hoc, deterministic):** `fire` requires visible flame answer yes; else `smoke`. `vehicle_blocking_traffic` requires "other vehicles queued/deviating around it" yes; else `stalled_or_broken_down_vehicle`. `traffic_congestion` requires "many vehicles, slow or stopped, across lanes" yes AND no single-vehicle obstruction.
- **Explanation:** the model's `explanation` is post-edited: clamp 20–500 chars, strip newlines, prepend nothing. Template fallback if empty: `"{class label}: {top yes-answer question rephrased}. Observed at ~{t}s from aerial view."`

### 5.4 Fusion (`sentinel/fusion.py`)
Per second `t`, per class `c`:
`p_c[t] = α · s_c[t] + β · Σ_w∈W(t) v_w.conf · 1[v_w.class = c] · K(t − center_w)` where `K` is a triangular kernel over the window; α≈0.3, β≈0.7 (sweep). Seconds never sent to S2 keep only α-scaled S1 evidence, which is below τ_on by construction → *no event can be emitted without VLM confirmation.*

### 5.5 S3 — Temporal decoder (`sentinel/decoder.py`)  ← where most L2/L3 points live
Per class curve `p_c[t]`:
1. **Smooth:** median filter width 3 s (impulse classes) / 7 s (ramp/state).
2. **Hysteresis:** enter when `p_c ≥ τ_on` (≈0.6), stay while `p_c ≥ τ_off` (≈0.35).
3. **Min duration:** impulse 2 s; dwell 8 s (and start = first second of the stationary streak, per §1.3); ramp 5 s.
4. **Gap merge:** merge same-class intervals separated by < `g_c` (impulse 4 s, ramp 15 s, dwell 10 s, L3: ×1.5). *This is the anti-fragment rule.*
5. **Cross-class NMS:** overlapping intervals of different classes → keep higher mean `p`, unless the pair is a known co-occurrence (`fire`+`smoke`, `traffic_accident`+`traffic_congestion` aftermath) — keep both only if each has independent VLM votes.
6. **Interval shaping for IoU ≥ 0.5:** if the interval is shorter than the class's typical GT length (learned from `train/*/ground_truth.csv` medians), extend symmetrically to the median length (stay inside clip). Being *slightly too long* is safe up to 2× GT; being too short is fatal below 0.5×. Bias long.
7. **Video-level abstention:** if `max_t max_c p_c[t] < τ_video` (≈0.65) → `events = []`. Tune τ_video on the public set to maximise `(hits) − λ·(false alarms on normal videos)` with λ = 2 (a false alarm on a normal video costs a whole video; a miss costs part of one).
8. **L1 path:** skip 1–7; `class = argmax over classes of mean p_c` if its value ≥ τ_L1 else `[]`.

### 5.6 S4 — Emission (`sentinel/emit.py`)
Build the prediction object exactly to the schema in §6; attach `explanation`; attach *extra* ignored-but-impressive fields: `confidence`, `responder_action` (e.g. `"dispatch: traffic police + tow; alert ambulance if occupants visible"`), `evidence_frames` (timestamps). Unknown fields are ignored by the scorer, so they cost nothing and read well in the write-up.

### 5.7 Responder action map (`configs/actions.yaml`) — ties to "so that someone can act"
| class | action |
|---|---|
| traffic_accident | Alert ambulance + traffic police; hold drone on station; stream to control room |
| fire / smoke | Alert fire service; estimate spread direction from wind/plume; keep safe standoff |
| waterlogging_or_flood | Notify municipal drainage; push road-closure advisory |
| stalled / blocking | Dispatch tow / patrol; push navigation warning |
| wrong_way_driving | Immediate police alert; track vehicle |
| road_spill_or_debris | Dispatch road maintenance; lane-closure advisory |
| fighting_or_violence | Police dispatch; keep recording as evidence |
| loitering_or_suspicious_presence | Flag to security; continue observation (no action if it resolves) |
| traffic_congestion | Signal-timing adjustment; traffic advisory |

---

## 6. Data contracts

### 6.1 Ground truth (input) — `ground_truth.csv`
`video_id, level, is_anomaly, class_name, start_time_sec, end_time_sec, description_summary` — one row per event; normal = one row, class `normal`, empty times.

### 6.2 Manifest (input) — `manifest.json` from arena
Assume `[{"video_id": "E001", "level": 1, ...}, ...]`. Write `sentinel/manifest.py` to load *either* a list or `{"videos":[...]}` and to read `level` from any of `level|difficulty|tier`. Log the raw keys on first load; adapt within minutes if the shape differs.

### 6.3 Submission (output) — exactly this
```json
{
  "schema_version": "1.0",
  "submission_id": "sentinel-run-03",
  "model_name": "sentinel-cascade-siglip2-qwen3vl4b-lora",
  "run_metadata": {"total_wall_time_ms": 0, "hardware": "1x T4 16GB", "vlm_wake_ratio": 0.18},
  "predictions": [
    {"video_id": "E001", "events": [],
     "runtime_metadata": {"frames_processed": 90, "chunks_processed": 1,
                          "end_to_end_internal_time_ms": 4820, "model_runtimes": []}},
    {"video_id": "E009", "events": [{"class_name": "fire", "start_time_sec": null, "end_time_sec": null,
                                     "explanation": "Orange flames and a dark plume rise from a roadside structure; nearby vehicles have stopped."}],
     "runtime_metadata": {...}},
    {"video_id": "E021", "events": [{"class_name": "traffic_accident", "start_time_sec": 20.0, "end_time_sec": 41.0,
                                     "explanation": "..."}],
     "runtime_metadata": {"frames_processed": 600, "chunks_processed": 24, "end_to_end_internal_time_ms": 91000,
       "model_runtimes": [{"model_name": "siglip2-b16", "call_count": 600, "total_time_ms": 3100, "average_time_ms": 5.17, "p50_time_ms": 5.0, "p95_time_ms": 7.2, "max_time_ms": 12.0},
                          {"model_name": "qwen3-vl-4b-lora", "call_count": 5, "total_time_ms": 12400, "average_time_ms": 2480.0, "p50_time_ms": 2400.0, "p95_time_ms": 3100.0, "max_time_ms": 3150.0}]}}
  ]
}
```
Hard rules encoded in `sentinel/validate.py`: `predictions` present; each video once; `video_id ∈ manifest`; class ∈ 11 (never `normal`); L1 → both times `null`; L2/L3 → `start ≥ 0`, `end > start`; explanation 20–500 chars if present; `average_time_ms ≈ total/count` (±2%); `len(call_times_ms) == call_count` if present; file < 5 MB.

### 6.4 Runtime metadata — honesty rules
Start the clock when the video file is opened; stop when the event list is final. Include decode, S1, S2, S3. Exclude model load (warm up on a dummy clip first). `model_runtimes` from a `Ledger` context manager (`with ledger.time("qwen3-vl-4b-lora"): …`) that records every call; stats computed, never typed.

---

## 7. Repository layout

```
sentinel/
├── README.md                      # how to run in 5 commands; architecture diagram (PNG + mermaid)
├── PLAN.md                        # this file
├── pyproject.toml                 # deps: torch, transformers, open_clip/siglip via transformers, av, opencv-python-headless,
│                                  #       numpy, pydantic, typer, rich, fastapi, uvicorn, faiss-cpu, pyyaml, vllm (extra), unsloth (extra)
├── Dockerfile                     # CUDA 12.x base; `make serve` starts vLLM + API
├── Makefile                       # setup / gate-cache / run / score / validate / submit-check / demo
├── configs/
│   ├── default.yaml               # fps, thresholds, class priors, paths, model ids
│   ├── prompt_bank.yaml           # SigLIP text prompts per class × viewpoint
│   ├── question_bank.yaml         # VLM yes/no questions per class
│   ├── actions.yaml               # responder actions
│   └── class_priors.yaml          # median durations, gap-merge, min-dur (auto-filled by scripts/derive_priors.py)
├── sentinel/
│   ├── __init__.py
│   ├── schema.py                  # pydantic: Event, Prediction, Submission, RuntimeMetadata, ModelRuntime
│   ├── manifest.py                # load manifest / videos.csv / ground_truth.csv
│   ├── ingest.py                  # S0
│   ├── gate.py                    # S1 (SigLIP2 + motion + normality bank)
│   ├── verifier.py                # S2 (VLMClient: vllm | llamacpp | mock)
│   ├── prompts.py                 # builds system/user prompts from question bank; mosaic + motion tint
│   ├── fusion.py                  # S1+S2 → p_c[t]
│   ├── decoder.py                 # S3 temporal decoding
│   ├── emit.py                    # S4 → Prediction objects, explanations, actions
│   ├── ledger.py                  # timing ledger → runtime_metadata
│   ├── validate.py                # arena field rules (mirrors §6.3)
│   ├── scorer.py                  # LOCAL replica of arena scoring (§8.2)
│   └── pipeline.py                # orchestrates per video / per level; caching; resume
├── scripts/
│   ├── build_normal_bank.py       # embeds train/normal frames → banks/normal.npy
│   ├── derive_priors.py           # GT medians per class → configs/class_priors.yaml
│   ├── make_sft_dataset.py        # train/ → JSONL (6-frame mosaics + JSON targets), balanced with hard normals
│   ├── pseudo_label.py            # dev-only: NIM/Gemini teacher on unannotated drone clips → JSONL
│   ├── train_lora_unsloth.py      # Kaggle/Modal; saves adapter + merged fp16 + GGUF export
│   ├── sweep_thresholds.py        # grid over τ_gate/τ_on/τ_off/τ_video/gap on public test → best config
│   ├── run_level.py               # run pipeline for a level subset → submissions/<ts>_L{n}.json
│   ├── merge_submissions.py       # compose the final sheet from best per-video answers
│   └── export_edge.py             # SigLIP2 → ONNX/TensorRT INT8; Qwen3-VL → GGUF Q4_K_M; Jetson runbook
├── dashboard/
│   ├── app.py                     # FastAPI: serves UI + SSE stream of pipeline events
│   └── static/index.html          # single-file UI (video, score timeline, gate/VLM indicators, latency HUD, RTF gauge)
├── tests/
│   ├── test_schema_validate.py    # every gotcha in §2 has a failing fixture
│   ├── test_scorer.py             # hand-computed IoU cases; normal-video zero rule; fragment penalty
│   ├── test_decoder.py            # synthetic curves → expected intervals (hysteresis, merge, min-dur, shaping)
│   ├── test_ledger.py             # avg = total/count within 2%
│   └── test_pipeline_mock.py      # end-to-end on 3 synthetic videos with MockVLM → valid submission
├── data/                          # (gitignored) train/, test/, manifest.json, unannotated/
├── banks/                         # normal.npy, text_bank.npy
├── cache/                         # per-video npz from S1, S2 JSON verdicts
├── submissions/                   # every generated sheet + local score sidecar
└── writeup/
    ├── architecture.md            # → export to PDF/HTML for the final submission
    └── diagram.svg
```

**Coding standards:** Python 3.11, type hints everywhere, pydantic models at every boundary, `typer` CLIs, `rich` logging, no notebooks in the repo (notebooks only for Kaggle training; export the script). Every stage is a pure function `(inputs, config) → outputs` and caches to disk keyed by `video_id + config_hash`, so re-running after a threshold change costs seconds, not GPU minutes. `MockVLM` (returns scripted verdicts) makes the whole pipeline runnable on CPU in CI.

---

## 8. Testing & validation

### 8.1 Schema validation (`sentinel/validate.py`) — run before *every* upload
Fixtures in `tests/` for each gotcha: `class_name: normal`; L1 with timestamps; `end <= start`; duplicate `video_id`; unknown `video_id`; explanation 12 chars; `average_time_ms` off by 5%; `call_times_ms` length mismatch; missing `runtime_metadata`. All must be *rejected* locally with the same field name the arena would print.

### 8.2 Local scorer (`sentinel/scorer.py`) — replica of arena rules
- **L1:** pooled over L1 videos: `0.5 · acc(anomaly vs normal) + 0.5 · acc(class)` (class acc over anomalous videos; a wrong class on a normal video counts against both).
- **L2/L3, per video:** GT normal → `1` if no events else `0`. GT has events → `w_alert · 1[any event] + w_match · (matched / |GT|) + w_time · mean IoU over matched`, with one-to-one greedy matching on IoU ≥ 0.5 & same class; unmatched predictions subtract a penalty `w_fp · (unmatched / (|pred|))`. Weights: L2 `(0.3, 0.4, 0.3)`, L3 `(0.2, 0.3, 0.5)`, `w_fp=0.25`. *The exact arena weights are unknown; the point is to be directionally identical so threshold sweeps transfer. Log our L2/L3 components separately as the arena does.*
- **Speed:** RTF = Σ `end_to_end_internal_time_ms` / Σ video duration; bonus rises as RTF falls below 1.0.
- **Reasoning:** count of events with valid explanations / events.
- Report: per-level score, per-video breakdown, confusion matrix, fragment count, false-alarm count on normal videos. Used by `sweep_thresholds.py`.

### 8.3 Upload gate (`make submit-check FILE=…`)
Refuses to mark a sheet "ready" unless (a) validate passes, (b) local public-set score ≥ previous best − 0.5 pts, (c) zero false alarms on public normal L2/L3 videos, (d) fragment count ≤ |events|. Prints a one-line diff vs. the last uploaded sheet (which videos changed).

### 8.4 Unit & integration tests
`pytest -q` must be green before hour 3 and stay green. `test_pipeline_mock.py` generates three synthetic videos with OpenCV (moving rectangles; one "stops" — dwell; one "flashes red" — impulse; one plain — normal) and asserts a valid, correctly-shaped submission from the MockVLM path.

### 8.5 Public-set protocol
Split the 34 public videos 24/10 by level-stratified hash. Sweep on 24, confirm on 10. Never sweep on all 34 or we overfit the private set's siblings. Report both numbers in the write-up (judges respect this).

---

## 9. Compute plan

| Where | What | Notes |
|---|---|---|
| Kaggle T4×2 (30 h/wk) | LoRA training (one GPU), S1 bank building (other GPU) | Save adapter to `/kaggle/working` every 100 steps |
| Modal ($30) L4 / A10G | vLLM serving of Qwen3-VL-4B for the 28-video inference run and the demo | Keep a warm container; `modal serve` with `keep_warm=1` during demo |
| Local laptop / CPU | Everything else: S3, scorer, validator, dashboard, tests via MockVLM | |
| NIM (40 RPM) / Gemini free | `pseudo_label.py` on ≤300 unannotated drone clips; question-bank generation | Dev only |

If GPU is scarce: the *entire* system also runs on a T4 alone — SigLIP2 (0.4 GB) + Qwen3-VL-4B fp16 (≈9 GB) + KV cache fits in 16 GB with `--gpu-memory-utilization 0.9`.

---

## 10. Novelty & wow factors (what sets us apart in a room of Qwen3-VL LoRAs)

Every team will show a fine-tuned small VLM with a decent accuracy number. We win on **system thinking demonstrated live**. Ranked by impact-per-hour:

1. **Cascade with a live "energy budget" HUD.** The dashboard shows the cheap gate scanning 100% of frames and the VLM lighting up on ~15–20%. A live RTF gauge and "VLM wake ratio" tile answer the organisers' central question ("can a small VLM do this in real time, economically?") visually, in one glance. → Directly targets the Speed bonus and the FlytBase judges' cost-per-drone concern.
2. **Calibrated abstention ("silence is a decision").** Self-consistency voting + video-level abstention threshold tuned against the normal-video-zero rule. Show a *normal* clip where a naive VLM cries wolf and SENTINEL stays quiet, with the explanation "traffic is dense but flowing; no obstruction". → Targets the biggest point-killer in the scoring.
3. **Temporal event grammar.** Per-class shape priors (impulse/ramp/dwell/trajectory/static) and interval shaping for IoU ≥ 0.5 — explained with one diagram showing why "stalled vehicle" starts *after* it stops. Most teams emit raw VLM windows; this is where L2/L3 points come from.
4. **Site normality memory.** The normal bank can be *re-seeded from the first 30 s of any new drone feed* (`--adapt-normal`): the system learns what is ordinary *for this location* and flags deviation from it — the parking-bay-vs-highway-shoulder problem from the PS, solved literally. Demo: same stopped car, two contexts, two verdicts.
5. **Every alert is actionable.** Explanation + responder action + evidence frame timestamps on each event. Reads like an operations product, not a classifier output. Also harvests the Reasoning bonus.
6. **Open-vocabulary "watchlist."** `--extra-events "open drain,crowd surge,animal on carriageway"` adds classes at runtime via the prompt bank + question bank with no retraining (kept out of the submission, shown in the demo). Directly answers "this is not a fixed list."
7. **Motion-mask attention (Cerberus).** A faint tint on moving regions in the VLM's key frame; show side-by-side that it fixes a small-object miss from aerial view.
8. **Teacher→student distillation flywheel.** `pseudo_label.py` uses a big hosted model on the *unannotated* drone footage (explicitly allowed) to mint drone-viewpoint training data for the LoRA. Show the loop diagram; it is how this would scale after the hackathon.
9. **Edge deployability proof.** GGUF Q4 of the tuned model on llama.cpp + SigLIP2 INT8 ONNX; a table with measured T4 numbers and projected Jetson Orin numbers (with methodology stated). Brings the "run on drone hardware" claim from slideware to a runbook.
10. **Honest, reproducible, one-command.** `make demo VIDEO=…` from a clean clone; tests green; latency ledger auto-generated; public-set score reported on a held-out split. In a hackathon, engineering hygiene is itself a differentiator.

**Demo script (6 minutes):** (1) 30 s: the question and the three scoring facts; (2) live run on a drone clip with the HUD — gate scanning, VLM waking on the accident, interval forming, explanation + action card appearing; (3) the normal-clip abstention; (4) the two-context stopped-car; (5) open-vocab watchlist adding "open drain"; (6) the edge table + flywheel slide; (7) held-out score, RTF, what we'd do with a week.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Fine-tune doesn't finish / degrades | Zero-shot path is the critical path; LoRA is a plug-in adapter behind a config flag |
| Manifest shape differs from assumption | `manifest.py` is tolerant; log raw keys; 10-minute fix budget |
| VLM latency too high on T4 | Cut frames 8→6, res 448→384, disable self-consistency for L1; cascade already limits calls |
| Over-firing on normal videos | τ_video sweep with λ=2 penalty; abstention on disagreement; manual review of every L2/L3 event before upload (28 videos — feasible) |
| Fragments | Gap-merge + shaping; `submit-check` blocks fragment count > events |
| Wrong `average_time_ms` math | Ledger computes; validator checks |
| Uploading a worse sheet | Local gate + per-video merge tool; last upload of the day = best-known sheet, re-uploaded deliberately |
| Night footage embeddings collapse | Luminance-conditional CLAHE for S1; night prompts in the bank; night normals in the bank |

---

## 12. Final submission package
- **Code repo:** public GitHub, README with the 5-command run, architecture PNG, results table (public held-out), hardware, RTF.
- **Architecture write-up (PDF/HTML ≤ 25 MB):** 4–6 pages: the diagram from §4; per-frame vs per-window vs per-video table; the temporal grammar figure; the abstention figure; the energy-budget figure; edge table; flywheel; limitations; "with more time" (GRPO à la TAU-R1, tracker-based dwell timing, audio).
- **Notes:** limitations honestly stated; how to run; what to do with more time.

---

## 13. Execution order (for the builder — whichever agent or human)

**Phase A — Skeleton that can already submit (≈ 90 min)**
1. Repo scaffold, `schema.py`, `validate.py`, `manifest.py`, `ledger.py`, `scorer.py`, tests. `MockVLM`. Generate a valid all-normal submission for the manifest. *Milestone: a validated file exists.*
2. `ingest.py` + `gate.py` (SigLIP2 zero-shot + motion). Run on public test; cache npz. Quick L1 from S1 alone; score locally.

**Phase B — Real cascade (≈ 2.5 h)**
3. `verifier.py` against vLLM Qwen3-VL-4B-Instruct (zero-shot) with question bank + JSON schema. `prompts.py` mosaics + motion tint.
4. `fusion.py`, `decoder.py` with class priors from `derive_priors.py`. `emit.py` with explanations + actions.
5. `sweep_thresholds.py` on the 24-video sweep split; confirm on 10. **Upload L1 sheet, then L2, then L3** after `submit-check`.

**Phase C — Parallel tracks (start at Phase A step 2, on other machines)**
6. `make_sft_dataset.py` → Kaggle `train_lora_unsloth.py` (fp16, 4-bit, vision frozen). Evaluate adapter vs zero-shot on the public split; swap in if better; re-run only the videos whose verdicts change; re-upload those.
7. `pseudo_label.py` on unannotated drone clips (teacher) → add to SFT set if time; otherwise show as flywheel in write-up.

**Phase D — Wow & wrap (≈ 1.5 h)**
8. `dashboard/` HUD; `--adapt-normal`; `--extra-events`; `export_edge.py` (GGUF + ONNX) with measured numbers.
9. Write-up + README + diagram; final re-upload of the best-known sheet; fill Final submission section.

**Definition of done:** validated sheet uploaded for all 28 videos; zero predicted events on any public normal L2/L3 video; explanations on 100% of events; tests green; write-up and repo links filled in; demo runs from `make demo`.
