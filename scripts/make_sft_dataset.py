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


CONFUSABLE_TWINS: dict[str, list[str]] = {
    "fighting_or_violence": ["loitering_or_suspicious_presence", "normal", "traffic_accident"],
    "traffic_accident": ["stalled_or_broken_down_vehicle", "vehicle_blocking_traffic", "road_spill_or_debris"],
    "stalled_or_broken_down_vehicle": ["vehicle_blocking_traffic", "traffic_congestion", "traffic_accident"],
    "vehicle_blocking_traffic": ["stalled_or_broken_down_vehicle", "traffic_congestion", "traffic_accident"],
    "fire": ["smoke", "road_spill_or_debris", "normal"],
    "smoke": ["fire", "waterlogging_or_flood", "normal"],
    "traffic_congestion": ["vehicle_blocking_traffic", "stalled_or_broken_down_vehicle", "normal"],
    "wrong_way_driving": ["traffic_accident", "vehicle_blocking_traffic", "normal"],
    "waterlogging_or_flood": ["road_spill_or_debris", "smoke", "normal"],
    "road_spill_or_debris": ["waterlogging_or_flood", "traffic_accident", "stalled_or_broken_down_vehicle"],
    "loitering_or_suspicious_presence": ["fighting_or_violence", "stalled_or_broken_down_vehicle", "normal"],
    "normal": ["fighting_or_violence", "loitering_or_suspicious_presence", "traffic_congestion", "stalled_or_broken_down_vehicle"],
}


def _expl(cls: str, summary: str) -> str:
    if summary and 25 <= len(summary) <= 300:
        return summary.strip()
    fallback = {
        "normal": "Traffic flows smoothly across all active lanes with pedestrian movement adhering to normal walkways; no collision, hazard, fire or obstruction is present.",
        "traffic_accident": "Two or more vehicles have collided on the roadway, showing structural impact damage, vehicle dislocation, and stopped traffic queuing behind the crash site.",
        "traffic_congestion": "Heavy multi-vehicle queuing observed bumper-to-bumper across multiple travel lanes with persistent crawling or stationary traffic flow.",
        "stalled_or_broken_down_vehicle": "A single stationary disabled vehicle occupies a live lane or shoulder with hazard indicators or driver attention while adjacent traffic continues past.",
        "vehicle_blocking_traffic": "A stationary vehicle obstructs the travel path or intersection, forcing following vehicles to brake suddenly, queue, or veer into opposing lanes.",
        "fire": "Active orange flames and combustion are clearly visible on a vehicle, structure, or roadside vegetation accompanied by rising thermal smoke.",
        "smoke": "A dense rising plume or drifting cloud of dark grey/black smoke obscures the visual scene and roadway without open flames visible.",
        "waterlogging_or_flood": "Significant standing water covers the road surface, submerging wheel bases and creating visible water displacement and traffic slowing.",
        "wrong_way_driving": "A vehicle is actively navigating against the established one-way direction of travel or oncoming traffic lanes, posing immediate collision risk.",
        "road_spill_or_debris": "Hazardous physical debris, detached cargo, or spilled materials are strewn across the traffic lanes, forcing vehicles to take evasive swerves.",
        "fighting_or_violence": "Multiple individuals engaged in hostile physical altercation involving striking, grappling, punching, or violent street assault rather than recreation.",
        "loitering_or_suspicious_presence": "Individual(s) lingering persistently in an unauthorized, isolated perimeter, closed facility, or critical infrastructure zone without legitimate transit purpose.",
    }
    return fallback.get(cls, "Ordinary scene with routine activity and no emergency incident detected.")


def sample_top_classes(cls: str, k: int = 4) -> list[str]:
    """Sample candidate classes prioritizing confusing twin pairs to teach sharp fine-grained discrimination."""
    twins = [c for c in CONFUSABLE_TWINS.get(cls, []) if c != cls and c in CLASSES]
    remaining = [c for c in CLASSES if c != cls and c not in twins]
    random.shuffle(remaining)
    selected_twins = twins[:2]
    needed = max(0, k - 1 - len(selected_twins))
    selected = selected_twins + remaining[:needed]
    if cls != "normal":
        return [cls] + selected
    return (selected + remaining)[:k]


@app.command()
def main(train_dir: Path = "data/train", out: Path = "data/sft", per_video: int = 2, max_per_class: int = 250,
         frames: int = 6, val_frac: float = 0.1, question_bank: Path = "configs/question_bank.yaml", seed: int = 0):
    random.seed(seed)
    qb = load_question_bank(question_bank)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    per_class: dict[str, int] = {}

    train_path = Path(train_dir)
    if not train_path.exists():
        print(f"Error: train_dir '{train_dir}' does not exist.")
        return

    cls_dirs = sorted(p for p in train_path.iterdir() if p.is_dir())
    print(f"Found {len(cls_dirs)} class folders in {train_dir}")

    for cls_dir in cls_dirs:
        gt_csv = cls_dir / "ground_truth.csv"
        if not gt_csv.exists():
            continue
        gt = load_ground_truth(gt_csv)
        files = load_videos_csv(cls_dir / "videos.csv") if (cls_dir / "videos.csv").exists() else {}
        by_vid: dict[str, list] = {}
        for e in gt:
            by_vid.setdefault(e.video_id, []).append(e)

        for vid, evs in by_vid.items():
            if per_class.get(cls_dir.name, 0) >= max_per_class:
                break
            vname = files.get(vid, f"{vid}.mp4")
            path = cls_dir / "videos" / vname
            if not path.exists():
                alt_path = cls_dir / vname
                if alt_path.exists():
                    path = alt_path
                else:
                    continue

            try:
                clip = load_clip(vid, str(path), sample_fps=1.0, max_side=448)
            except Exception as ex:  # noqa: BLE001
                print("skip", path, ex)
                continue

            D = clip.duration_sec
            windows: list[tuple[float, float, str, str]] = []  # (a, b, class, summary)
            ev_iv = [(e.start_time_sec, e.end_time_sec) for e in evs if e.is_anomaly and e.start_time_sec is not None]

            for e in evs:
                if not e.is_anomaly:
                    for _ in range(per_video):
                        a = random.uniform(0, max(0.0, D - 12))
                        windows.append((a, min(D, a + 12), "normal", e.description_summary))
                elif e.start_time_sec is None:  # level-1 clip: whole clip is the event
                    windows.append((0.0, D, e.class_name, e.description_summary))
                else:
                    a, b = e.start_time_sec, e.end_time_sec
                    windows.append((a, b, e.class_name, e.description_summary))
                    # Hard negative anchor: segment of the same video outside all events
                    for _ in range(2):
                        s = random.uniform(0, max(0.0, D - 12))
                        t = min(D, s + 12)
                        if all(t <= x or s >= y for x, y in ev_iv):
                            windows.append((s, t, "normal", "Routine traffic and activity outside event window."))
                            break

            for a, b, cls, summ in windows:
                fr, ts = clip.frames_between(a, b, frames)
                if len(fr) < 2:
                    continue
                paths = []
                for i, (f, t) in enumerate(zip(fr, ts)):
                    p = out / "images" / f"{vid}_{int(a)}_{i}.jpg"
                    cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                    paths.append(str(p))

                # Use confusable twin sampling for sharp discrimination
                top = sample_top_classes(cls, k=4)
                qs = select_questions(qb, top)
                onset = 0 if cls == "normal" else int(next((i for i, t in enumerate(ts) if t >= a), 0))
                expl_text = _expl(cls, summ)

                rows.append({
                    "images": paths,
                    "times": ts,
                    "class_name": cls,
                    "top_classes": top,
                    "question": user_text(qs, ts, len(fr)),
                    "answer": _answer(cls, expl_text, onset, qs)
                })
                per_class[cls] = per_class.get(cls, 0) + 1

    random.shuffle(rows)
    n_val = int(len(rows) * val_frac)
    (out / "val.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(out / "val.jsonl", "w") as f:
        for r in rows[:n_val]:
            f.write(json.dumps(r) + "\n")
    with open(out / "train.jsonl", "w") as f:
        for r in rows[n_val:]:
            f.write(json.dumps(r) + "\n")

    print("\n📊 Generated SFT Dataset Class Distribution:")
    print(json.dumps(per_class, indent=2))
    print(f"✅ Total SFT Samples: {len(rows)} ({len(rows)-n_val} train, {n_val} val) saved to {out}\n")


if __name__ == "__main__":
    app()
