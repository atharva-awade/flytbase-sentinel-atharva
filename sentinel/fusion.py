"""Fuse S1 gate scores and S2 VLM verdicts into per-class per-second curves (PLAN.md §5.4).

curve_c[t] = alpha * s1_c[t] + max_over_windows( beta * conf_w * K_w(t) )
No event can be emitted without VLM confirmation: alpha * max(s1) < tau_on by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CLASSES


@dataclass
class Verdict:
    """One S2 (VLM) verdict for a window [start, end)."""
    start: float
    end: float
    class_name: str            # one of CLASSES or "normal"
    confidence: float          # 0..1
    explanation: str = ""
    onset_sec: float | None = None   # model-estimated onset inside the window, if any
    flat: bool = False               # hold-forward verdicts: constant support across the window


def fuse(s1: dict[str, np.ndarray], verdicts: list[Verdict], T: int,
         alpha: float = 0.3, beta: float = 0.7) -> dict[str, np.ndarray]:
    gate = {c: np.zeros(T, dtype=float) for c in CLASSES}
    vlm = {c: np.zeros(T, dtype=float) for c in CLASSES}
    for c in CLASSES:
        if c in s1 and len(s1[c]):
            v = np.asarray(s1[c], dtype=float)
            n = min(T, len(v))
            gate[c][:n] = v[:n]
    for vd in verdicts:
        if vd.class_name not in CLASSES or vd.confidence <= 0:
            continue
        s, e = int(max(0, np.floor(vd.start))), int(min(T, np.ceil(vd.end)))
        if e <= s:
            continue
        ts = np.arange(s, e, dtype=float)
        if vd.flat:
            k = np.ones_like(ts)
        elif vd.onset_sec is not None and s <= vd.onset_sec < e:
            # support ramps up from the model-estimated onset; seconds before it get little weight
            k = np.clip((ts - vd.onset_sec + 1.0) / 2.0, 0.15, 1.0)
        else:
            center, half = (s + e) / 2.0, max(1.0, (e - s) / 2.0)
            k = 1.0 - 0.5 * np.abs(ts - center) / half   # triangular, floor 0.5
        vlm[vd.class_name][s:e] = np.maximum(vlm[vd.class_name][s:e], beta * vd.confidence * k)
    return {c: np.clip(alpha * gate[c] + vlm[c], 0.0, 1.0) for c in CLASSES}


def best_explanations(verdicts: list[Verdict]) -> dict[str, str]:
    """Highest-confidence explanation per class."""
    out: dict[str, tuple[float, str]] = {}
    for v in verdicts:
        if v.class_name in CLASSES and v.explanation:
            if v.class_name not in out or v.confidence > out[v.class_name][0]:
                out[v.class_name] = (v.confidence, v.explanation)
    return {k: v[1] for k, v in out.items()}
