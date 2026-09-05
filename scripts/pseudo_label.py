"""DEVELOPMENT-TIME ONLY teacher labelling of unannotated drone clips (PLAN.md §3.2, §10 item 8).

Uses a large hosted VLM (NVIDIA NIM or any OpenAI-compatible endpoint) to write the same JSON our verifier
emits, for 12 s windows of unannotated footage. Output feeds make_sft_dataset-style JSONL for distillation.
Never called at inference time.

  export NVIDIA_API_KEY=nvapi-...
  python scripts/pseudo_label.py --videos data/unannotated --out data/pseudo --model <check GET /v1/models> --rpm 30
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel import CLASSES  # noqa: E402
from sentinel.ingest import load_clip  # noqa: E402
from sentinel.prompts import build_messages, load_question_bank, parse_json, select_questions, user_text  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(videos: Path, out: Path = "data/pseudo", base_url: str = "https://integrate.api.nvidia.com/v1",
         model: str = "meta/llama-3.2-90b-vision-instruct", api_key_env: str = "NVIDIA_API_KEY",
         window: float = 12.0, stride: float = 12.0, frames: int = 6, rpm: int = 30, min_conf: float = 0.8,
         question_bank: Path = "configs/question_bank.yaml", limit: int = 300):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
    qb = load_question_bank(question_bank)
    (out / "images").mkdir(parents=True, exist_ok=True)
    fout = open(out / "pseudo.jsonl", "a")
    n = 0
    for v in sorted(videos.glob("*.mp4")):
        clip = load_clip(v.stem, str(v), sample_fps=1.0, max_side=448)
        t = 0.0
        while t < clip.duration_sec and n < limit:
            fr, ts = clip.frames_between(t, min(clip.duration_sec, t + window), frames)
            qs = select_questions(qb, list(CLASSES))          # teacher gets every class's questions
            msgs = build_messages(fr, ts, qs, motion_tint=False, max_side=448)
            try:
                r = client.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=500)
                d = parse_json(r.choices[0].message.content or "{}")
            except Exception as ex:  # noqa: BLE001
                print("teacher error", ex); time.sleep(5); continue
            cls, conf = d.get("class_name", "normal"), float(d.get("confidence", 0) or 0)
            if cls in CLASSES and conf < min_conf:
                t += stride; continue                          # only keep confident anomaly labels
            paths = []
            for i, (f, tt) in enumerate(zip(fr, ts)):
                p = out / "images" / f"{v.stem}_{int(t)}_{i}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR)); paths.append(str(p))
            fout.write(json.dumps({"images": paths, "times": ts, "class_name": cls, "question": user_text(qs, ts, len(fr)),
                                   "answer": json.dumps(d), "teacher": model}) + "\n"); fout.flush()
            n += 1; print(v.stem, f"{t:.0f}s", cls, conf)
            t += stride; time.sleep(60.0 / rpm)
    print(n, "pseudo-labelled windows →", out)


if __name__ == "__main__":
    app()
