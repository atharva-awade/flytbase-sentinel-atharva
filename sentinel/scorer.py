"""Local replica of the arena scoring (PLAN.md §8.2).

The arena's exact L2/L3 weights are undisclosed; this scorer encodes every *stated* rule exactly
(normal-video zero, IoU>=0.5 one-to-one matching, pooled L1, fragments penalised) and uses
declared weights for the rest so threshold sweeps transfer directionally.
Component scores are always reported separately, like the arena does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import GTEvent, Prediction

IOU_GATE = 0.5
LEVEL_POINTS = {1: 25.0, 2: 35.0, 3: 40.0}
# (alert, match, timing) weights for videos that contain events
W = {2: (0.30, 0.40, 0.30), 3: (0.20, 0.30, 0.50)}
W_FP = 0.25  # penalty weight on unmatched predictions (fragments / wrong class / hallucinations)


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


@dataclass
class VideoScore:
    video_id: str
    level: int
    gt_normal: bool
    score: float
    alert: float = 0.0
    match: float = 0.0
    timing: float = 0.0
    matched: int = 0
    n_gt: int = 0
    n_pred: int = 0
    false_alarm: bool = False
    fragments: int = 0
    notes: str = ""


@dataclass
class Report:
    level_scores: dict[int, float] = field(default_factory=dict)      # 0..1 per level
    level_points: dict[int, float] = field(default_factory=dict)      # scaled to 25/35/40
    l1_anomaly_acc: float = 0.0
    l1_class_acc: float = 0.0
    videos: list[VideoScore] = field(default_factory=list)
    rtf: float | None = None
    explanation_coverage: float = 0.0
    total_points: float = 0.0

    def summary(self) -> str:
        lines = [f"TOTAL {self.total_points:.1f}/100  " +
                 "  ".join(f"L{l}: {self.level_points.get(l, 0):.1f}/{LEVEL_POINTS[l]:.0f}" for l in (1, 2, 3))]
        lines.append(f"L1 anomaly-acc {self.l1_anomaly_acc:.2f}  class-acc {self.l1_class_acc:.2f}  |  "
                     f"explanations {self.explanation_coverage:.0%}  |  RTF {self.rtf if self.rtf is not None else 'n/a'}")
        fa = [v.video_id for v in self.videos if v.false_alarm]
        fr = sum(v.fragments for v in self.videos)
        lines.append(f"false alarms on normal L2/L3: {len(fa)} {fa}   fragments: {fr}")
        for v in sorted(self.videos, key=lambda x: (x.level, x.video_id)):
            lines.append(f"  L{v.level} {v.video_id:<8} {v.score:5.2f}  gt={'normal' if v.gt_normal else v.n_gt}"
                         f" pred={v.n_pred} matched={v.matched} {v.notes}")
        return "\n".join(lines)


def _score_l1(preds: dict[str, Prediction], gts: dict[str, list[GTEvent]]) -> tuple[float, float, float, list[VideoScore]]:
    anom_hits = cls_hits = n = n_anom = 0
    rows = []
    for vid, g in gts.items():
        n += 1
        gt_normal = all(not e.is_anomaly for e in g)
        gt_classes = {e.class_name for e in g if e.is_anomaly}
        p = preds.get(vid)
        pred_classes = [e.class_name for e in (p.events if p else [])]
        pred_anom = len(pred_classes) > 0
        a_ok = pred_anom != gt_normal
        anom_hits += a_ok
        c_ok = False
        if not gt_normal:
            n_anom += 1
            c_ok = any(c in gt_classes for c in pred_classes)
            cls_hits += c_ok
        rows.append(VideoScore(vid, 1, gt_normal, 0.5 * a_ok + 0.5 * (c_ok if not gt_normal else a_ok),
                               n_gt=len(gt_classes), n_pred=len(pred_classes),
                               notes=("" if a_ok else "WRONG-ANOMALY ") + ("" if (gt_normal or c_ok) else f"WRONG-CLASS {pred_classes}->{sorted(gt_classes)}")))
    a_acc = anom_hits / n if n else 0.0
    c_acc = cls_hits / n_anom if n_anom else 1.0
    return 0.5 * a_acc + 0.5 * c_acc, a_acc, c_acc, rows


def score_video_temporal(vid: str, level: int, p: Prediction | None, g: list[GTEvent]) -> VideoScore:
    gt_events = [e for e in g if e.is_anomaly and e.start_time_sec is not None and e.end_time_sec is not None]
    pred_events = list(p.events) if p else []
    if not gt_events:  # ground truth normal
        fa = len(pred_events) > 0
        return VideoScore(vid, level, True, 0.0 if fa else 1.0, n_pred=len(pred_events), false_alarm=fa,
                          notes="FALSE-ALARM" if fa else "")
    # greedy one-to-one matching on IoU (same class), best pairs first
    pairs = []
    for i, pe in enumerate(pred_events):
        if pe.start_time_sec is None or pe.end_time_sec is None:
            continue
        for j, ge in enumerate(gt_events):
            if pe.class_name != ge.class_name:
                continue
            v = iou(pe.start_time_sec, pe.end_time_sec, ge.start_time_sec, ge.end_time_sec)
            if v >= IOU_GATE:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    ious = []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); ious.append(v)
    alert = 1.0 if pred_events else 0.0
    match = len(ious) / len(gt_events)
    timing = sum(ious) / len(ious) if ious else 0.0
    unmatched = len(pred_events) - len(ious)
    fp_pen = W_FP * (unmatched / len(pred_events)) if pred_events else 0.0
    wa, wm, wt = W[level]
    score = max(0.0, wa * alert + wm * match + wt * timing - fp_pen)
    # fragments: several predictions of the same class overlapping one GT event
    fragments = 0
    for j, ge in enumerate(gt_events):
        n_over = sum(1 for pe in pred_events if pe.class_name == ge.class_name and pe.start_time_sec is not None
                     and iou(pe.start_time_sec, pe.end_time_sec, ge.start_time_sec, ge.end_time_sec) > 0)
        fragments += max(0, n_over - 1)
    notes = []
    if unmatched:
        notes.append(f"unmatched={unmatched}")
    if fragments:
        notes.append(f"fragments={fragments}")
    if not ious:
        notes.append("MISS")
    return VideoScore(vid, level, False, score, alert, match, timing, len(ious), len(gt_events), len(pred_events),
                      fragments=fragments, notes=" ".join(notes))


def score(predictions: list[Prediction], gt: list[GTEvent], video_durations: dict[str, float] | None = None) -> Report:
    preds = {p.video_id: p for p in predictions}
    gts: dict[str, list[GTEvent]] = {}
    for e in gt:
        gts.setdefault(e.video_id, []).append(e)
    rep = Report()
    by_level: dict[int, dict[str, list[GTEvent]]] = {1: {}, 2: {}, 3: {}}
    for vid, g in gts.items():
        by_level[g[0].level][vid] = g

    if by_level[1]:
        s, a, c, rows = _score_l1(preds, by_level[1])
        rep.level_scores[1], rep.l1_anomaly_acc, rep.l1_class_acc = s, a, c
        rep.videos.extend(rows)
    for lvl in (2, 3):
        rows = [score_video_temporal(vid, lvl, preds.get(vid), g) for vid, g in by_level[lvl].items()]
        if rows:
            rep.level_scores[lvl] = sum(r.score for r in rows) / len(rows)
            rep.videos.extend(rows)
    for lvl, s in rep.level_scores.items():
        rep.level_points[lvl] = s * LEVEL_POINTS[lvl]
    rep.total_points = sum(rep.level_points.values())

    all_events = [e for p in predictions for e in p.events]
    with_expl = [e for e in all_events if e.explanation and 20 <= len(e.explanation) <= 500]
    rep.explanation_coverage = len(with_expl) / len(all_events) if all_events else 1.0
    if video_durations:
        proc = sum(p.runtime_metadata.end_to_end_internal_time_ms for p in predictions) / 1000.0
        dur = sum(video_durations.get(p.video_id, 0.0) for p in predictions)
        rep.rtf = round(proc / dur, 3) if dur > 0 else None
    return rep
