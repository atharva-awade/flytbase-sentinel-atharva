# SENTINEL — Architecture write-up (skeleton; fill numbers on the day, export to PDF/HTML ≤ 25 MB)

## 1. The question we answer
Can a small VLM detect context-dependent anomalies in drone video, in real time, on limited GPU, without crying wolf?
Our answer: **yes, if it is not asked to watch everything.**

## 2. System
(Insert the S0–S4 diagram from PLAN.md §4 as an image. Per-frame vs per-window vs per-video table below.)

| Runs per… | Component | Cost (measured) |
|---|---|---|
| frame (1–2 fps) | decode → SigLIP2-B/16 embedding → class scores, novelty, motion, stationarity | ___ ms/frame on T4 |
| candidate window (~__ % of seconds) | Qwen3-VL-4B (+LoRA), 6 frames, fine-grained yes/no questions, JSON | ___ s/call, __ calls per video |
| video | temporal decoder, abstention, emission | < 50 ms |

## 3. Why each stage exists (tie to scoring)
- Gate → real-time factor and cost per drone (Speed bonus; the organisers' central question).
- Verifier → context ("stationary car: parking bay vs highway shoulder"); yes/no questions beat "is this an anomaly?".
- Decoder → IoU ≥ 0.5 geometry, one interval per event, class-shape priors (75 % of marks are temporal).
- Abstention → normal video + any prediction = 0.

## 4. Training (parallel track)
SFT set: __ windows (__ % normal incl. hard negatives), 6 frames each, target JSON. LoRA r=16 on language layers,
vision tower frozen, fp16 4-bit on T4, __ min. Public held-out: zero-shot __ → LoRA __ points.

## 5. Results (public set, 24/10 split)
| | L1 /25 | L2 /35 | L3 /40 | total | FA on normal | RTF | wake ratio |
|---|---|---|---|---|---|---|---|
| zero-shot cascade | | | | | | | |
| + LoRA | | | | | | | |

## 6. Edge deployability
Qwen3-VL-4B Q4_K_M GGUF via llama.cpp: ___ s/call on T4; projected Jetson Orin: ___ (method: ___).
SigLIP2 ONNX INT8: ___ ms/frame.

## 7. Novelty
Energy-budget HUD · calibrated abstention · temporal event grammar · site normality memory · actionable alerts ·
open-vocabulary watchlist · motion-mask attention · teacher→student flywheel.

## 8. Limitations & next steps
Wrong-way driving needs a tracker for direction; dwell timing would improve with per-object tracks (ByteTrack);
GRPO à la TAU-R1 for onset precision; audio for accidents/violence; night-domain LoRA data from the flywheel.
