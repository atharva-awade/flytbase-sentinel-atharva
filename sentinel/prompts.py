"""Prompt construction for the VLM verifier: system/user text, frame mosaics, motion tint (PLAN.md §5.3)."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from . import CLASSES
from .gate import moving_mask

TAXONOMY = {
    "traffic_accident": "vehicles collided / overturned / crashed",
    "traffic_congestion": "many vehicles stopped or crawling across lanes",
    "stalled_or_broken_down_vehicle": "one vehicle stationary on the roadway or shoulder while traffic flows",
    "vehicle_blocking_traffic": "a stopped vehicle that others must queue behind or swerve around",
    "fire": "visible open flames",
    "smoke": "smoke plume without clearly visible flames",
    "waterlogging_or_flood": "standing or flowing water on the road",
    "wrong_way_driving": "a vehicle moving against the flow of traffic",
    "road_spill_or_debris": "objects, cargo or spilled material on the road surface",
    "fighting_or_violence": "people physically fighting",
    "loitering_or_suspicious_presence": "person(s) lingering without purpose or in a restricted place",
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {"type": "array", "items": {"type": "object", "properties": {
            "q": {"type": "integer"}, "yes": {"type": "boolean"}, "conf": {"type": "number"}},
            "required": ["q", "yes", "conf"]}},
        "class_name": {"type": "string", "enum": list(CLASSES) + ["normal"]},
        "confidence": {"type": "number"},
        "onset_frame": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": ["answers", "class_name", "confidence", "onset_frame", "explanation"],
}

SYSTEM = (
    "You are a drone-operations video analyst supporting emergency responders. You are shown frames sampled "
    "in time order from one video (aerial, CCTV or dashcam). Decide whether one of these situations is happening:\n"
    + "\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items())
    + "\nOtherwise the scene is 'normal'. Dense but flowing traffic, cars parked in bays, and people walking are normal. "
    "False alarms are costly: choose 'normal' unless the visual evidence is clear. Answer strictly as JSON."
)


def load_question_bank(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def select_questions(qb: dict, top_classes: list[str]) -> list[tuple[str, str]]:
    """[(class_or_'normal', question)] — 'always' questions first, then top-K class questions."""
    qs = [("normal", q) for q in qb.get("always", [])]
    for c in top_classes:
        for q in qb["classes"].get(c, []):
            qs.append((c, q))
    return qs


def user_text(questions: list[tuple[str, str]], times: list[float], n_frames: int) -> str:
    qlines = "\n".join(f"{i}. {q}" for i, (_, q) in enumerate(questions))
    tl = ", ".join(f"frame {i} = {t:.0f}s" for i, t in enumerate(times))
    return (
        f"You see {n_frames} frames in time order ({tl}). Answer each question yes/no with a confidence 0-1:\n{qlines}\n\n"
        "Then give: class_name (one of the listed classes or 'normal'), confidence 0-1, onset_frame (index of the "
        "first frame where the situation is visible, or 0), and a one-sentence explanation (20-300 characters) that "
        "names concrete visual evidence (objects, positions, motion). Return only JSON with keys "
        "answers, class_name, confidence, onset_frame, explanation."
    )


def tint_motion(prev: np.ndarray, cur: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Cerberus-style attention: faint red tint on moving regions of `cur`."""
    m = moving_mask(prev, cur) > 0
    out = cur.copy()
    red = np.zeros_like(cur); red[..., 0] = 255
    out[m] = (out[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return out


def burn_timestamp(img: np.ndarray, t: float) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, f"t={t:.0f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, f"t={t:.0f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mosaic(frames: list[np.ndarray], times: list[float], cols: int = 3, tile: int = 336) -> np.ndarray:
    tiles = []
    for f, t in zip(frames, times):
        h, w = f.shape[:2]
        s = tile / max(h, w)
        r = cv2.resize(f, (int(w * s), int(h * s)))
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[: r.shape[0], : r.shape[1]] = r
        tiles.append(burn_timestamp(canvas, t))
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)


def to_data_url(img: np.ndarray, max_side: int = 448, quality: int = 85) -> str:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(frames: list[np.ndarray], times: list[float], questions: list[tuple[str, str]],
                   motion_tint: bool = True, max_side: int = 448, as_mosaic: bool = False) -> list[dict]:
    imgs = [burn_timestamp(f, t) for f, t in zip(frames, times)]
    if motion_tint and len(frames) >= 2:
        k = int(np.argmax([np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).mean() for i in range(1, len(frames))])) + 1
        imgs[k] = tint_motion(frames[k - 1], imgs[k])
    content: list[dict] = []
    if as_mosaic:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(mosaic(imgs, times), max_side * 2)}})
    else:
        for im in imgs:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(im, max_side)}})
    content.append({"type": "text", "text": user_text(questions, times, len(frames))})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction (handles ```json fences and leading prose)."""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    return json.loads(t[a:b + 1]) if a >= 0 and b > a else {}
