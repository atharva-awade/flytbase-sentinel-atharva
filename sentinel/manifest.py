"""Tolerant loaders for manifest.json, videos.csv and ground_truth.csv (PLAN.md §6.1-6.2)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import GTEvent, ManifestEntry

_LEVEL_KEYS = ("level", "difficulty", "tier", "task_level")
_ID_KEYS = ("video_id", "id", "video", "name")
_FILE_KEYS = ("file", "filename", "path", "video_file")


def _first(d: dict[str, Any], keys: tuple[str, ...], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _coerce_level(v) -> int:
    if isinstance(v, str):
        v = v.strip().upper().lstrip("L").lstrip("D")
    return int(v)


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        for k in ("videos", "manifest", "items", "predictions", "entries"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:  # maybe {video_id: {...}} mapping
            if all(isinstance(v, dict) for v in data.values()):
                data = [{"video_id": k, **v} for k, v in data.items()]
    if not isinstance(data, list):
        raise ValueError(f"Unrecognised manifest shape: top-level {type(data).__name__}")
    out: list[ManifestEntry] = []
    for row in data:
        if isinstance(row, str):
            row = {"video_id": row, "level": 1}
        vid = _first(row, _ID_KEYS)
        lvl = _first(row, _LEVEL_KEYS, 1)
        if vid is None:
            raise ValueError(f"Manifest row without id: {row}")
        out.append(ManifestEntry(video_id=str(vid), level=_coerce_level(lvl),
                                 file=_first(row, _FILE_KEYS),
                                 duration_sec=_first(row, ("duration_sec", "duration", "length_sec"))))
    return out


def load_videos_csv(path: str | Path) -> dict[str, str]:
    """video_id -> filename"""
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vid = _first(row, _ID_KEYS)
            fn = _first(row, _FILE_KEYS + ("video", "mp4"))
            if vid and fn:
                out[str(vid)] = str(fn)
    return out


def _float_or_none(v) -> float | None:
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("nan", "none", "null"):
        return None
    return float(v)


def load_ground_truth(path: str | Path) -> list[GTEvent]:
    out: list[GTEvent] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            is_anom = str(row.get("is_anomaly", "")).strip().lower() in ("1", "true", "yes", "y")
            cls = (row.get("class_name") or "normal").strip()
            if cls == "normal":
                is_anom = False
            out.append(GTEvent(
                video_id=str(_first(row, _ID_KEYS)),
                level=_coerce_level(_first(row, _LEVEL_KEYS, 1)),
                is_anomaly=is_anom,
                class_name=cls,
                start_time_sec=_float_or_none(row.get("start_time_sec")),
                end_time_sec=_float_or_none(row.get("end_time_sec")),
                description_summary=(row.get("description_summary") or "").strip(),
            ))
    return out


def gt_by_video(gt: list[GTEvent]) -> dict[str, list[GTEvent]]:
    d: dict[str, list[GTEvent]] = {}
    for e in gt:
        d.setdefault(e.video_id, []).append(e)
    return d


def manifest_from_gt(gt: list[GTEvent]) -> list[ManifestEntry]:
    """Build a manifest from a ground_truth.csv (public test set has no manifest.json)."""
    seen: dict[str, int] = {}
    for e in gt:
        seen.setdefault(e.video_id, e.level)
    return [ManifestEntry(video_id=v, level=l) for v, l in seen.items()]
