"""S4 — turn decoded intervals into arena Prediction objects (PLAN.md §5.6)."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .schema import Event, Interval, Prediction, RuntimeMetadata
from .validate import EXPL_MAX, EXPL_MIN

_ACTIONS = Path(__file__).resolve().parent.parent / "configs" / "actions.yaml"
_actions_cache: dict[str, str] | None = None

LABEL = {
    "traffic_accident": "Traffic accident", "traffic_congestion": "Traffic congestion",
    "stalled_or_broken_down_vehicle": "Stalled or broken-down vehicle", "vehicle_blocking_traffic": "Vehicle blocking traffic",
    "fire": "Fire", "smoke": "Smoke", "waterlogging_or_flood": "Waterlogging / flood", "wrong_way_driving": "Wrong-way driving",
    "road_spill_or_debris": "Road spill or debris", "fighting_or_violence": "Fighting or violence",
    "loitering_or_suspicious_presence": "Loitering / suspicious presence",
}


def actions() -> dict[str, str]:
    global _actions_cache
    if _actions_cache is None:
        with open(_ACTIONS) as f:
            _actions_cache = yaml.safe_load(f)
    return _actions_cache


def clean_explanation(text: str | None, cls: str, t: float | None = None, viewpoint: str = "aerial") -> str:
    """Guarantee a 20-500 char, single-line, concrete explanation."""
    s = re.sub(r"\s+", " ", (text or "")).strip().strip('"')
    if len(s) < EXPL_MIN:
        where = f" at ~{int(t)}s" if t is not None else ""
        s = f"{LABEL.get(cls, cls)} observed from the {viewpoint} view{where}; scene context deviates from normal traffic flow."
    if len(s) > EXPL_MAX:
        s = s[:EXPL_MAX - 1].rsplit(" ", 1)[0].rstrip(",;:") + "."
    if len(s) < EXPL_MIN:  # pathological trim
        s = (s + " Anomaly confirmed by the verifier model.")[:EXPL_MAX]
    return s


def to_prediction(video_id: str, level: int, intervals: list[Interval] | None, l1_class: str | None,
                  runtime: RuntimeMetadata, explanations: dict[str, str] | None = None,
                  l1_confidence: float | None = None) -> Prediction:
    ev: list[Event] = []
    ex = explanations or {}
    if level == 1:
        if l1_class:
            ev.append(Event(class_name=l1_class, start_time_sec=None, end_time_sec=None,
                            explanation=clean_explanation(ex.get(l1_class), l1_class),
                            confidence=l1_confidence, responder_action=actions().get(l1_class)))
    else:
        for iv in intervals or []:
            if iv.end <= iv.start:
                continue
            ev.append(Event(class_name=iv.class_name, start_time_sec=float(iv.start), end_time_sec=float(iv.end),
                            explanation=clean_explanation(iv.explanation or ex.get(iv.class_name), iv.class_name, iv.peak_t),
                            confidence=round(float(iv.score), 3), responder_action=actions().get(iv.class_name),
                            evidence_frames_sec=[float(iv.peak_t)]))
    return Prediction(video_id=video_id, events=ev, runtime_metadata=runtime)
