"""Online (streaming) SENTINEL: frame in → overlay + live state out.

Same cascade as pipeline.py, executed incrementally so it works on a live camera, an RTSP feed or a file
played in real time:

  push(frame, t)  ──▶ motion boxes (every frame)          ──▶ overlay drawn on the frame
                  ──▶ gate embedding (1 Hz)                ──▶ suspicion / class curves
                  ──▶ VLM job on a hot window (async)      ──▶ verdicts
                  ──▶ online decode (1 Hz)                 ──▶ events (open = still happening)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import CLASSES, NORMAL
from .decoder import decode, load_priors
from .emit import LABEL, actions, clean_explanation
from .fusion import Verdict, best_explanations, fuse
from .gate import GateResult
from .ingest import Clip, night_lift
from .ledger import Ledger
from .schema import Event, Interval, Prediction

# light-theme overlay palette (BGR)
C_INK = (38, 44, 52)
C_BRAND = (206, 120, 38)       # #2678CE blue (BGR)
C_ALERT = (52, 78, 230)        # #E64E34 red
C_OK = (94, 173, 52)           # #34AD5E green
C_VLM = (196, 96, 150)         # purple
C_WHITE = (255, 255, 255)


@dataclass
class LiveEvent:
    class_name: str
    start: float
    end: float
    open: bool
    confidence: float
    explanation: str
    action: str
    peak_t: float


@dataclass
class LiveState:
    t: float = 0.0
    fps_in: float = 0.0
    fps_proc: float = 0.0
    suspicion: float = 0.0
    novelty: float = 0.0
    motion: float = 0.0
    top_classes: list[tuple[str, float]] = field(default_factory=list)
    vlm_busy: bool = False
    vlm_calls: int = 0
    vlm_avg_ms: float = 0.0
    gate_avg_ms: float = 0.0
    wake_ratio: float = 0.0
    rtf: float = 0.0
    frames: int = 0
    events: list[LiveEvent] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)
    active: LiveEvent | None = None
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    curves: dict[str, list[float]] = field(default_factory=dict)
    suspicion_curve: list[float] = field(default_factory=list)
    windows: list[tuple[float, float]] = field(default_factory=list)
    normal_bank_size: int = 0

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("events", "active", "boxes")}
        d["events"] = [e.__dict__ for e in self.events]
        d["active"] = self.active.__dict__ if self.active else None
        d["top_classes"] = [(c, round(float(v), 3)) for c, v in self.top_classes]
        d["curves"] = {c: [round(x, 3) for x in v] for c, v in self.curves.items()}
        d["suspicion_curve"] = [round(x, 3) for x in self.suspicion_curve]
        return d


class StreamProcessor:
    def __init__(self, cfg: dict, gate, verifier, level: int = 3, priors: dict | None = None):
        self.cfg, self.gate, self.verifier, self.level = cfg, gate, verifier, level
        self.priors = priors or load_priors(cfg["paths"].get("priors"))
        gc, vc = cfg["gate"], cfg["verifier"]
        self.tau_gate, self.pad = float(gc["tau_gate"]), float(gc["pad_sec"])
        self.window_sec, self.stride_sec = float(vc.get("window_sec", 12)), float(vc.get("stride_sec", 6))
        self.k_frames = int(vc.get("frames_per_window", 6))
        self.names, self.TB = self.gate._text_bank()
        self.ledger = Ledger()
        self.pool = ThreadPoolExecutor(max_workers=int(vc.get("max_concurrency", 2)))
        self.lock = threading.Lock()
        # history at 1 Hz
        self.sec_probs: list[np.ndarray] = []
        self.sec_nov: list[float] = []
        self.sec_mot: list[float] = []
        self.sec_stat: list[float] = []
        self.embs: deque = deque(maxlen=64)
        self.ring: deque = deque(maxlen=int(self.window_sec * 2 * 4))   # (t, frame) ~4 fps worth of 2 windows
        self.verdicts: list[Verdict] = []
        self.windows: list[tuple[float, float]] = []
        self.jobs: list[tuple[float, float, Future]] = []
        self.last_window_end = -1e9
        self.prev_small: np.ndarray | None = None
        self.last_gate_t = -1e9
        self.last_decode_t = -1e9
        self.t0_wall = time.perf_counter()
        self.state = LiveState()
        self._fps_hist: deque = deque(maxlen=30)
        self._last_push_wall = None
        self.intervals: list[Interval] = []
        self.suspicion_hist: list[float] = []

    # ------------------------------------------------------------------ helpers
    def _motion(self, frame: np.ndarray) -> tuple[float, list[tuple[int, int, int, int]], np.ndarray | None]:
        small = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_RGB2GRAY)
        if self.prev_small is None:
            self.prev_small = small
            return 0.0, [], None
        d = cv2.absdiff(small, self.prev_small)
        self.prev_small = small
        energy = float(np.clip(d.mean() / 12.0, 0, 1))
        _, m = cv2.threshold(d, 22, 255, cv2.THRESH_BINARY)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = frame.shape[:2]
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 30:
                continue
            boxes.append((int(x * W / 224), int(y * H / 224), int(w * W / 224), int(h * H / 224)))
        boxes.sort(key=lambda b: -b[2] * b[3])
        return energy, boxes[:12], m

    def _gate_step(self, frame: np.ndarray, t: float, motion: float) -> None:
        f = night_lift(frame, self.cfg["gate"].get("night_lum_thresh", 40))
        t0 = time.perf_counter()
        e = self.gate.embed_images([f], None).astype(np.float32)[0]
        self.ledger.record(getattr(self.gate, "name", "gate"), (time.perf_counter() - t0) * 1000)
        e = e / (np.linalg.norm(e) + 1e-8)
        logits = 100.0 * self.TB @ e
        p = np.exp(logits - logits.max()); p /= p.sum()
        nb = self.gate.normal_bank
        if nb is not None and len(nb):
            nov = float(np.clip((1.0 - float((nb @ e).max()) - 0.05) / 0.35, 0, 1))
        else:
            nov = float(1.0 - p[self.names.index(NORMAL)]) if NORMAL in self.names else 0.0
        stat = 1.0
        if len(self.embs) >= 5:
            stat = float(1.0 - np.clip(np.linalg.norm(e - self.embs[-5]) / 0.6, 0, 1))
        self.embs.append(e)
        self.sec_probs.append(p); self.sec_nov.append(nov); self.sec_mot.append(motion); self.sec_stat.append(stat)

    def _gate_result(self) -> GateResult:
        T = len(self.sec_probs)
        P = np.stack(self.sec_probs)
        cs = {n: P[:, i] for i, n in enumerate(self.names)}
        return GateResult(T=T, class_scores=cs, novelty=np.asarray(self.sec_nov), motion=np.asarray(self.sec_mot),
                          stationarity=np.asarray(self.sec_stat), suspicion=np.asarray(self.suspicion_hist))

    def _suspicion(self) -> float:
        p = self.sec_probs[-1]
        anomaly_max = max(p[self.names.index(c)] for c in CLASSES if c in self.names)
        gc = self.cfg["gate"]
        dwell = self.sec_stat[-1] * (1 - self.sec_mot[-1]) * max(p[self.names.index("stalled_or_broken_down_vehicle")],
                                                                 p[self.names.index("vehicle_blocking_traffic")])
        u = gc["w_class"] * anomaly_max + gc["w_novelty"] * self.sec_nov[-1] + gc["w_motion"] * max(self.sec_mot[-1] * anomaly_max, dwell)
        return float(np.clip(u, 0, 1))

    def _maybe_wake_vlm(self, t: float) -> None:
        """Onset-aware wake policy.
        * onset (gate just crossed tau):      short window ending now  -> fast first verdict
        * hot but last verdict was 'normal':  re-check every ~3 s with a recent-biased window
        * anomaly confirmed:                  standard window every stride_sec to keep the hold alive
        """
        u = self.suspicion_hist
        if u[-1] < self.tau_gate:
            return
        onset = len(u) < 2 or u[-2] < self.tau_gate
        with self.lock:
            last_anom = any(v.class_name in CLASSES for v in self.verdicts[-2:])
        if onset:
            a, b, min_gap = max(0.0, t - 5.0), t + 1, 0.0
        elif not last_anom:
            a, b, min_gap = max(0.0, t - 7.0), t + 1, 3.0
        else:
            a, b, min_gap = max(0.0, t - self.window_sec + 1), t + 1, self.stride_sec
        if not onset and t - self.last_window_end < min_gap:
            return
        if self.jobs and not self.jobs[-1][2].done() and t - self.last_window_end < 2.0:
            return                                  # don't pile up while a verdict is in flight
        frames = [(tt, fr) for tt, fr in self.ring if a <= tt <= b]
        if len(frames) < 2:
            return
        pick = np.unique(np.linspace(0, len(frames) - 1, min(self.k_frames, len(frames))).round().astype(int))
        fr = [frames[i][1] for i in pick]; ts = [frames[i][0] for i in pick]
        self.last_window_end = b
        self.windows.append((a, b))
        gr = self._gate_result()
        fut = self.pool.submit(self._vlm_job, fr, ts, a, b, gr)
        self.jobs.append((a, b, fut))

    def _vlm_job(self, frames, times, a, b, gr: GateResult) -> Verdict:
        clip = Clip(video_id="live", path="", duration_sec=b, fps_native=0, frames=frames, times=times)
        v = self.verifier.verify_window(clip, gr, a, b, self.ledger)
        with self.lock:
            self.verdicts.append(v)
        return v

    def _decode(self, T: int) -> None:
        gr = self._gate_result()
        fc = self.cfg["fusion"]
        with self.lock:
            vs = list(self.verdicts)
        # hold-forward: a recent anomaly verdict keeps supporting the present until the next verdict
        # (which arrives every stride_sec) confirms or contradicts it — otherwise live alerts stutter.
        hold = self.stride_sec + 1.5
        latest = max((v for v in vs), key=lambda v: v.end, default=None)
        held = []
        for v in vs:
            if v.class_name in CLASSES and v.end >= T - hold and v.end < T and (latest is None or latest.end <= v.end
                                                                                   or latest.class_name == v.class_name):
                held.append(Verdict(v.end, float(T), v.class_name, v.confidence * 0.9, v.explanation, flat=True))
        curves = fuse(gr.class_scores, vs + held, T, fc["alpha"], fc["beta"])
        self.intervals = decode(curves, self.level, self.priors, float(T), best_explanations(vs), live=True)
        self.state.curves = {c: curves[c].tolist() for c in CLASSES if curves[c].max() > 0.05}

    # ------------------------------------------------------------------ public
    def adapt_normal(self) -> int:
        """Site normality memory: declare what we've seen so far as normal for this location."""
        if len(self.embs):
            self.gate.adapt_normal(np.stack(list(self.embs)))
        return 0 if self.gate.normal_bank is None else len(self.gate.normal_bank)

    def push(self, frame_rgb: np.ndarray, t: float) -> np.ndarray:
        """Process one frame at media time t (seconds). Returns the frame with overlays (RGB)."""
        wall = time.perf_counter()
        if self._last_push_wall is not None:
            self._fps_hist.append(1.0 / max(wall - self._last_push_wall, 1e-6))
        self._last_push_wall = wall
        self.state.frames += 1
        self.state.t = t
        motion, boxes, _ = self._motion(frame_rgb)
        self.state.boxes = boxes
        self.state.motion = motion
        if len(self.ring) == 0 or t - self.ring[-1][0] >= 0.24:
            self.ring.append((t, frame_rgb))
        # 1 Hz gate + decode
        if t - self.last_gate_t >= 1.0 - 1e-6:
            self.last_gate_t = t
            self._gate_step(frame_rgb, t, motion)
            self.suspicion_hist.append(self._suspicion())
            self._maybe_wake_vlm(t)
            self._decode(len(self.sec_probs))
            p = self.sec_probs[-1]
            self.state.top_classes = sorted(((c, float(p[self.names.index(c)])) for c in CLASSES), key=lambda kv: -kv[1])[:3]
            self.state.novelty = self.sec_nov[-1]
            self.state.suspicion = self.suspicion_hist[-1]
            self.state.suspicion_curve = list(self.suspicion_hist)
        self._refresh_state(t)
        return self.draw(frame_rgb, t)

    def _refresh_state(self, t: float) -> None:
        s = self.state
        s.fps_proc = float(np.mean(self._fps_hist)) if self._fps_hist else 0.0
        s.vlm_busy = any(not f.done() for _, _, f in self.jobs)
        rt = {m.model_name: m for m in self.ledger.model_runtimes()}
        vname = getattr(self.verifier.client, "name", "vlm")
        if vname in rt:
            s.vlm_calls, s.vlm_avg_ms = rt[vname].call_count, rt[vname].average_time_ms
        gname = getattr(self.gate, "name", "gate")
        if gname in rt:
            s.gate_avg_ms = rt[gname].average_time_ms
        covered, last_end = 0.0, -1e9                      # union of VLM windows
        for a, b in sorted(self.windows):
            a = max(a, last_end); covered += max(0.0, b - a); last_end = max(last_end, b)
        s.wake_ratio = min(1.0, covered / max(t, 1e-6))
        proc = (time.perf_counter() - self.t0_wall)
        s.rtf = proc / max(t, 1e-6)
        s.windows = list(self.windows)
        with self.lock:
            s.verdicts = [dict(start=v.start, end=v.end, class_name=v.class_name, confidence=round(v.confidence, 3),
                               explanation=v.explanation) for v in self.verdicts]
        evs = []
        for iv in self.intervals:
            is_open = iv.end >= t - 1.5
            evs.append(LiveEvent(iv.class_name, iv.start, iv.end, is_open, round(iv.score, 3),
                                 clean_explanation(iv.explanation, iv.class_name, iv.peak_t),
                                 actions().get(iv.class_name, ""), iv.peak_t))
        s.events = evs
        s.active = next((e for e in evs if e.start <= t <= e.end + 1.5), None)
        s.normal_bank_size = 0 if self.gate.normal_bank is None else len(self.gate.normal_bank)

    # ------------------------------------------------------------------ overlay
    def draw(self, frame_rgb: np.ndarray, t: float) -> np.ndarray:
        img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        s = self.state
        alert = s.active is not None
        col = C_ALERT if alert else C_BRAND
        # moving-object boxes
        for (x, y, w, h) in s.boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 1, cv2.LINE_AA)
            cv2.line(img, (x, y), (x + min(10, w), y), col, 2); cv2.line(img, (x, y), (x, y + min(10, h)), col, 2)
        fs = max(0.38, min(0.6, W / 1100))          # font scale follows frame width
        # compact telemetry (top-right); the page chips carry the NORMAL/ALERT state
        right = f"t={t:.1f}s  u={s.suspicion:.2f}  " + ("VLM verifying" if s.vlm_busy else f"VLM x{s.vlm_calls}")
        (tw, th), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        pad = 6
        box = img[6:6 + th + 2 * pad, W - tw - 2 * pad - 8:W - 8].astype(np.float32)
        img[6:6 + th + 2 * pad, W - tw - 2 * pad - 8:W - 8] = (0.2 * box + 0.8 * 255).astype(np.uint8)
        cv2.putText(img, right, (W - tw - pad - 8, 6 + pad + th), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    C_VLM if s.vlm_busy else C_INK, 1, cv2.LINE_AA)
        # suspicion bar (bottom)
        cv2.rectangle(img, (0, H - 6), (W, H), (235, 235, 235), -1)
        cv2.rectangle(img, (0, H - 6), (int(W * s.suspicion), H), col, -1)
        thr = int(W * self.tau_gate); cv2.line(img, (thr, H - 8), (thr, H), C_INK, 1)
        # alert banner with explanation
        if alert:
            e = s.active
            txt = e.explanation
            lines, cur = [], ""
            for w in txt.split():
                if len(cur) + len(w) + 1 > max(30, int(W / (16 * fs))):
                    lines.append(cur); cur = w
                else:
                    cur = (cur + " " + w).strip()
            lines.append(cur)
            lh = int(30 * fs) + 4
            hb = int(44 * fs) + lh * len(lines) + int(30 * fs)
            y0 = max(0, H - 12 - hb)
            box = img[y0:H - 12, 8:W - 8].astype(np.float32)
            img[y0:H - 12, 8:W - 8] = (0.2 * box + 0.8 * 255).astype(np.uint8)
            cv2.rectangle(img, (8, y0), (W - 8, H - 12), C_ALERT, 1)
            cv2.putText(img, f"{LABEL.get(e.class_name, e.class_name)}  {e.start:.0f}s -> {'now' if e.open else f'{e.end:.0f}s'}   conf {e.confidence:.2f}",
                        (16, y0 + int(30 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs + 0.05, C_ALERT, 1, cv2.LINE_AA)
            for i, ln in enumerate(lines):
                cv2.putText(img, ln, (16, y0 + int(44 * fs) + lh * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, C_INK, 1, cv2.LINE_AA)
            cv2.putText(img, "> " + e.action[:int(W / (10 * fs))], (16, y0 + hb - 8), cv2.FONT_HERSHEY_SIMPLEX, fs - 0.03, C_BRAND, 1, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------ export
    def to_prediction(self, video_id: str, level: int | None = None) -> Prediction:
        level = level or self.level
        T = len(self.sec_probs)
        rt = self.ledger.runtime_metadata(frames_processed=self.state.frames, chunks_processed=max(1, len(self.windows)),
                                          end_to_end_ms=(time.perf_counter() - self.t0_wall) * 1000)
        evs = []
        if level == 1:
            if self.intervals:
                best = max(self.intervals, key=lambda i: i.score)
                evs.append(Event(class_name=best.class_name, explanation=clean_explanation(best.explanation, best.class_name),
                                 confidence=round(best.score, 3), responder_action=actions().get(best.class_name)))
        else:
            for iv in self.intervals:
                evs.append(Event(class_name=iv.class_name, start_time_sec=float(iv.start), end_time_sec=float(min(iv.end, T)),
                                 explanation=clean_explanation(iv.explanation, iv.class_name, iv.peak_t), confidence=round(iv.score, 3),
                                 responder_action=actions().get(iv.class_name), evidence_frames_sec=[float(iv.peak_t)]))
        return Prediction(video_id=video_id, events=evs, runtime_metadata=rt)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
