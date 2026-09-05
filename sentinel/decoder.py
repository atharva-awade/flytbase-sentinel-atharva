"""S3 — temporal decoder: per-class evidence curves -> clean intervals (PLAN.md §5.5).

Input:  curves  {class_name: np.ndarray[T]}  sampled at 1 Hz (index t == second t)
Output: list[Interval]  — at most one interval per real event, shaped for IoU >= 0.5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import median_filter

from . import CLASSES, SHAPE
from .schema import Interval

_DEFAULT_PRIORS = Path(__file__).resolve().parent.parent / "configs" / "class_priors.yaml"


def load_priors(path: str | Path | None = None) -> dict:
    with open(path or _DEFAULT_PRIORS) as f:
        return yaml.safe_load(f)


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width) | 1)  # odd
    return median_filter(x, size=width, mode="nearest") if len(x) >= width else x


def hysteresis(x: np.ndarray, tau_on: float, tau_off: float) -> list[tuple[int, int]]:
    """Return [start, end) index pairs. Enter at >= tau_on, leave when < tau_off."""
    out, inside, s = [], False, 0
    for t, v in enumerate(x):
        if not inside and v >= tau_on:
            inside, s = True, t
        elif inside and v < tau_off:
            out.append((s, t)); inside = False
    if inside:
        out.append((s, len(x)))
    return out


def merge_gaps(iv: list[tuple[int, int]], gap: float) -> list[tuple[int, int]]:
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def shape_interval(s: float, e: float, target: float, T: float) -> tuple[float, float]:
    """Extend a too-short interval symmetrically toward the class's median GT length, inside [0, T].
    Being up to 2x too long is safe under IoU>=0.5; being <0.5x is fatal. Bias long."""
    dur = e - s
    # Don't blindly inflate to the class median: a detection that already spans most of the event would
    # land at ~2x GT (IoU 0.5, razor's edge). Grow to at most 1.5x its own length unless it is a tiny
    # fragment, in which case 70% of the median is the floor.
    target = min(target, max(1.5 * dur, 0.7 * target))
    if dur >= target:
        return s, e
    pad = (target - dur) / 2.0
    s2, e2 = s - pad, e + pad
    if s2 < 0:
        e2 += -s2; s2 = 0.0
    if e2 > T:
        s2 -= e2 - T; e2 = T
    return max(0.0, s2), min(T, e2)


def decode(curves: dict[str, np.ndarray], level: int = 2, priors: dict | None = None,
           duration_sec: float | None = None, explain: dict[str, str] | None = None,
           live: bool = False) -> list[Interval]:
    """live=True: the curve ends at 'now'; intervals still open are NOT shaped (no back-extension) —
    shaping is applied once the event has closed."""
    P = priors or load_priors()
    d = P["defaults"]
    T = float(duration_sec) if duration_sec else float(max((len(v) for v in curves.values()), default=0))
    gap_mult = d["l3_gap_multiplier"] if level == 3 else 1.0
    cands: list[Interval] = []

    for cls, raw in curves.items():
        if cls not in CLASSES or raw is None or len(raw) == 0:
            continue
        sp = P["shapes"][SHAPE[cls]]
        raw = np.asarray(raw, dtype=float)
        x = _smooth(raw, sp["smooth_sec"])
        iv = hysteresis(x, d["tau_on"], d["tau_off"])
        iv = merge_gaps(iv, sp["gap_merge_sec"] * gap_mult)
        iv = [(s, e) for s, e in iv if (e - s) >= sp["min_dur_sec"]]
        if live:
            # fast path for a live alert: the last >=2 s of *raw* evidence above tau_on opens an interval now,
            # without waiting for the smoother / min-duration that shape clean batch intervals.
            n = len(raw); k = n
            while k > 0 and raw[k - 1] >= d["tau_on"]:
                k -= 1
            if n - k >= d.get("live_min_sec", 2) and not any(e >= n - 1 for _, e in iv):
                iv = [(s, e) for s, e in iv if e < k]        # closed ones stay
                iv.append((k, n))
                x = np.maximum(x, raw)                        # so the score reflects the raw tail
        for s, e in iv:
            seg = x[s:e]
            peak = int(s + np.argmax(seg))
            fs, fe = float(s), float(e)
            is_open = live and e >= len(x) - 1
            if sp.get("extend_to_median", True) and not is_open:
                fs, fe = shape_interval(fs, fe, float(P["classes"][cls]["median_dur_sec"]), T)
            cands.append(Interval(class_name=cls, start=round(fs, 2), end=round(fe, 2),
                                  score=float(seg.mean()), peak_t=float(peak),
                                  explanation=(explain or {}).get(cls)))

    # video-level abstention (PLAN §5.5 step 7)
    if not cands or max(c.score for c in cands) < d["tau_video"]:
        return []

    # cross-class NMS with co-occurrence whitelist
    allowed = {frozenset(p) for p in P.get("co_occur", [])}
    cands.sort(key=lambda c: -c.score)
    kept: list[Interval] = []
    for c in cands:
        clash = False
        for k in kept:
            if k.class_name == c.class_name:
                continue
            inter = min(k.end, c.end) - max(k.start, c.start)
            if inter > 0 and frozenset((k.class_name, c.class_name)) not in allowed:
                clash = True; break
        if not clash:
            kept.append(c)
    kept.sort(key=lambda c: c.start)
    return kept


def decode_l1(curves: dict[str, np.ndarray], priors: dict | None = None) -> str | None:
    """L1: one label for the clip or None (normal)."""
    P = priors or load_priors()
    best, best_v = None, -1.0
    for cls, raw in curves.items():
        if cls in CLASSES and len(raw):
            v = float(np.mean(np.sort(np.asarray(raw, dtype=float))[-max(1, len(raw) // 3):]))  # top-third mean
            if v > best_v:
                best, best_v = cls, v
    return best if best_v >= P["defaults"]["tau_l1"] else None
