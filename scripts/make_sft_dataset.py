"""Build the LoRA SFT dataset from train/ (PLAN.md §3.3).

Each sample = 6 frames sampled inside a GT event window (positive) or a non-event / normal stretch (negative),
saved as JPEGs, plus the target JSON the verifier expects at inference time. Balanced to >=40% normals,
including *hard normals*: non-event stretches of anomaly videos (dense-but-flowing traffic etc.).

  python scripts/make_sft_dataset.py --train-dir data/train --out data/sft --per-video 2 --max-per-class 250

Output: data/sft/{images/*.jpg, train.jsonl, val.jsonl}; each line:
  {"images": [...6 paths...], "times": [...], "question": "<user prompt>", "answer": "<json string>"}
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel import CLASSES  # noqa: E402
from sentinel.ingest import load_clip  # noqa: E402
from sentinel.manifest import load_ground_truth, load_videos_csv  # noqa: E402
from sentinel.prompts import load_question_bank, select_questions, user_text  # noqa: E402

app = typer.Typer(add_completion=False)


def _answer(cls: str, expl: str, onset: int, qs) -> str:
    answers = []
    for i, (qc, _) in enumerate(qs):
        yes = (qc == cls) if cls != "normal" else (qc == "normal")
        answers.append({"q": i, "yes": bool(yes), "conf": 0.9 if yes else 0.85})
    return json.dumps({"answers": answers, "class_name": cls, "confidence": 0.9 if cls != "normal" else 0.95,
                       "onset_frame": onset, "explanation": expl})


def _expl(cls: str, summary: str) -> str:
    if summary and 20 <= len(summary) <= 300:
        return summary
    fallback = {
        "normal": "Traffic and activity look routine for this location; no obstruction, collision, fire or hazard is visible.",
        "traffic_accident": "Vehicles have collided; damaged vehicles sit at unusual angles and traffic behind them has stopped.",
        "traffic_congestion": "Many vehicles are stopped or crawling bumper-to-bumper across the lanes for the whole segment.",
        "stalled_or_broken_down_vehicle": "One vehicle stays stationary on the carriageway while the rest of the traffic flows past it.",
        "vehicle_blocking_traffic": "A stopped vehicle obstructs the lane and other vehicles queue behind it or swerve around it.",
        "fire": "Open orange flames are visible with a rising smoke plume.",
        "smoke": "A grey smoke plume rises and drifts over the area without clearly visible flames.",
        "waterlogging_or_flood": "The road surface is covered by standing water and vehicles move slowly through it.",
        "wrong_way_driving": "A vehicle moves against the direction of the surrounding traffic.",
        "road_spill_or_debris": "Objects or spilled material lie on the road surface and vehicles steer around them.",
        "fighting_or_violence": "People are physically fighting, striking and grappling with each other.",
        "loitering_or_suspicious_presence": "A person lingers in one spot for a long time in an unusual or restricted place.",
    }
    return fallback[cls]


@app.command()
def main(train_dir: Path = "data/train", out: Path = "data/sft", per_video: int = 2, max_per_class: int = 250,
         frames: int = 6, val_frac: float = 0.1, question_bank: Path = "configs/question_bank.yaml", seed: int = 0):
    random.seed(seed)
    qb = load_question_bank(question_bank)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    per_class: dict[str, int] = {}
    for cls_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        gt = load_ground_truth(cls_dir / "ground_truth.csv")
        files = load_videos_csv(cls_dir / "videos.csv")
        by_vid: dict[str, list] = {}
        for e in gt:
            by_vid.setdefault(e.video_id, []).append(e)
        for vid, evs in by_vid.items():
            if per_class.get(cls_dir.name, 0) >= max_per_class:
                break
            path = cls_dir / "videos" / files.get(vid, f"{vid}.mp4")
            if not path.exists():
                continue
            try:
                clip = load_clip(vid, str(path), sample_fps=1.0, max_side=448)
            except Exception as ex:  # noqa: BLE001
                print("skip", path, ex); continue
            D = clip.duration_sec
            windows: list[tuple[float, float, str, str]] = []   # (a, b, class, summary)
            ev_iv = [(e.start_time_sec, e.end_time_sec) for e in evs if e.is_anomaly and e.start_time_sec is not None]
            for e in evs:
                if not e.is_anomaly:
                    for _ in range(per_video):
                        a = random.uniform(0, max(0.0, D - 12)); windows.append((a, min(D, a + 12), "normal", e.description_summary))
                elif e.start_time_sec is None:      # level-1 clip: whole clip is the event
                    windows.append((0.0, D, e.class_name, e.description_summary))
                else:
                    a, b = e.start_time_sec, e.end_time_sec
                    windows.append((a, b, e.class_name, e.description_summary))
                    # hard negative: a stretch of the same video outside all events
                    for _ in range(3):
                        s = random.uniform(0, max(0.0, D - 12)); t = min(D, s + 12)
                        if all(t <= x or s >= y for x, y in ev_iv):
                            windows.append((s, t, "normal", "")); break
            for a, b, cls, summ in windows:
                fr, ts = clip.frames_between(a, b, frames)
                if len(fr) < 2:
                    continue
                paths = []
                for i, (f, t) in enumerate(zip(fr, ts)):
                    p = out / "images" / f"{vid}_{int(a)}_{i}.jpg"
                    cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                    paths.append(str(p))
                top = [cls] + random.sample([c for c in CLASSES if c != cls], 3) if cls != "normal" else random.sample(list(CLASSES), 4)
                qs = select_questions(qb, top)
                onset = 0 if cls == "normal" else int(next((i for i, t in enumerate(ts) if t >= a), 0))
                rows.append({"images": paths, "times": ts, "class_name": cls, "top_classes": top,
                             "question": user_text(qs, ts, len(fr)), "answer": _answer(cls, _expl(cls, summ), onset, qs)})
                per_class[cls] = per_class.get(cls, 0) + 1
    random.shuffle(rows)
    n_val = int(len(rows) * val_frac)
    with open(out / "val.jsonl", "w") as f:
        for r in rows[:n_val]: f.write(json.dumps(r) + "\n")
    with open(out / "train.jsonl", "w") as f:
        for r in rows[n_val:]: f.write(json.dumps(r) + "\n")
    print(json.dumps(per_class, indent=1)); print(f"{len(rows)} samples → {out}")


if __name__ == "__main__":
    app()
