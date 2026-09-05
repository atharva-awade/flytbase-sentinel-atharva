"""Local replica of the arena's file validation (PLAN.md §2, §6.3, §8.1).

Returns a list of Problem rows shaped like the arena's rejection table: (video_id, field, message).
An empty list means the file would be accepted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from pydantic import ValidationError

from . import CLASSES, NORMAL
from .schema import ManifestEntry, Submission

MAX_FILE_BYTES = 5 * 1024 * 1024
EXPL_MIN, EXPL_MAX = 20, 500
AVG_TOL = 0.02


@dataclass
class Problem:
    video_id: str
    field: str
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.video_id}] {self.field}: {self.message}"


def _levels(manifest: Optional[Iterable[ManifestEntry]]) -> dict[str, int]:
    return {m.video_id: m.level for m in manifest} if manifest else {}


def validate_submission(
    sub: Submission | dict,
    manifest: Optional[Iterable[ManifestEntry]] = None,
    raw_bytes: Optional[int] = None,
) -> list[Problem]:
    probs: list[Problem] = []
    if isinstance(sub, dict):
        try:
            sub = Submission.model_validate(sub)
        except ValidationError as e:  # surface structural errors in arena style
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                probs.append(Problem("-", loc, err["msg"]))
            return probs

    if raw_bytes is not None and raw_bytes > MAX_FILE_BYTES:
        probs.append(Problem("-", "file", f"file is {raw_bytes} bytes, max {MAX_FILE_BYTES}"))

    levels = _levels(manifest)
    seen: set[str] = set()
    for p in sub.predictions:
        vid = p.video_id
        if vid in seen:
            probs.append(Problem(vid, "video_id", "appears more than once"))
        seen.add(vid)
        if levels and vid not in levels:
            probs.append(Problem(vid, "video_id", "not in manifest"))
        level = levels.get(vid)

        for i, ev in enumerate(p.events):
            f = f"events[{i}]"
            if ev.class_name == NORMAL:
                probs.append(Problem(vid, f + ".class_name", "'normal' is not allowed; use events: []"))
            elif ev.class_name not in CLASSES:
                probs.append(Problem(vid, f + ".class_name", f"unknown class '{ev.class_name}'"))
            if level == 1:
                if ev.start_time_sec is not None or ev.end_time_sec is not None:
                    probs.append(Problem(vid, f + ".start_time_sec", "must be null at Level 1"))
            elif level in (2, 3):
                if ev.start_time_sec is None or ev.end_time_sec is None:
                    probs.append(Problem(vid, f + ".start_time_sec", "required at Levels 2-3"))
                else:
                    if ev.start_time_sec < 0:
                        probs.append(Problem(vid, f + ".start_time_sec", "must be >= 0"))
                    if ev.end_time_sec <= ev.start_time_sec:
                        probs.append(Problem(vid, f + ".end_time_sec", "must be greater than start"))
            else:  # level unknown (no manifest): still enforce internal consistency
                if (ev.start_time_sec is None) != (ev.end_time_sec is None):
                    probs.append(Problem(vid, f + ".end_time_sec", "start/end must both be null or both set"))
                if ev.start_time_sec is not None and ev.end_time_sec is not None and ev.end_time_sec <= ev.start_time_sec:
                    probs.append(Problem(vid, f + ".end_time_sec", "must be greater than start"))
            if ev.explanation is not None:
                n = len(ev.explanation)
                if n < EXPL_MIN or n > EXPL_MAX:
                    probs.append(Problem(vid, f + ".explanation", f"length {n}, must be {EXPL_MIN}-{EXPL_MAX} chars"))

        for j, mr in enumerate(p.runtime_metadata.model_runtimes):
            f = f"runtime_metadata.model_runtimes[{j}]"
            if mr.call_count > 0:
                expected = mr.total_time_ms / mr.call_count
                if expected > 0 and abs(mr.average_time_ms - expected) / expected > AVG_TOL:
                    probs.append(Problem(vid, f + ".average_time_ms",
                                         f"{mr.average_time_ms:.2f} != total/calls={expected:.2f} (±2%)"))
            if mr.call_times_ms is not None and len(mr.call_times_ms) != mr.call_count:
                probs.append(Problem(vid, f + ".call_times_ms",
                                     f"has {len(mr.call_times_ms)} entries, call_count={mr.call_count}"))

    if levels:
        missing = sorted(set(levels) - seen)
        if missing:  # not an error (omitted videos keep previous answers) — informational
            probs.append(Problem("-", "info", f"{len(missing)} manifest videos not in this file (kept/normal): {missing[:6]}..."))
    return probs


def validate_file(path: str | Path, manifest: Optional[Iterable[ManifestEntry]] = None) -> list[Problem]:
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    return validate_submission(data, manifest, raw_bytes=len(raw))


def is_acceptable(problems: list[Problem]) -> bool:
    return all(p.field == "info" for p in problems)
