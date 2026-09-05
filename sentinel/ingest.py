"""S0 — decode & sample frames with container timestamps (PLAN.md §5.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Clip:
    video_id: str
    path: str
    duration_sec: float
    fps_native: float
    frames: list[np.ndarray] = field(default_factory=list)      # RGB uint8, long side <= max_side
    times: list[float] = field(default_factory=list)            # seconds, container-derived
    n_decoded: int = 0

    def frames_between(self, t0: float, t1: float, k: int) -> tuple[list[np.ndarray], list[float]]:
        """k frames uniformly spread over [t0, t1] from what we already sampled."""
        idx = [i for i, t in enumerate(self.times) if t0 <= t <= t1]
        if not idx:
            i = int(np.argmin(np.abs(np.asarray(self.times) - (t0 + t1) / 2)))
            idx = [i]
        pick = np.unique(np.linspace(0, len(idx) - 1, min(k, len(idx))).round().astype(int))
        sel = [idx[p] for p in pick]
        return [self.frames[i] for i in sel], [self.times[i] for i in sel]


def _resize(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def night_lift(img: np.ndarray, lum_thresh: float = 40.0) -> np.ndarray:
    """CLAHE on L channel if the frame is dark. Used for the gate path only (the VLM sees the original)."""
    if img.mean() >= lum_thresh:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_clip(video_id: str, path: str, sample_fps: float = 1.0, max_side: int = 448,
              max_frames: int | None = None) -> Clip:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n / fps if n > 0 else 0.0
    step = max(1, int(round(fps / sample_fps)))
    clip = Clip(video_id=video_id, path=path, duration_sec=duration, fps_native=fps)
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= 0 and i > 0:
                t = i / fps
            clip.frames.append(_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side))
            clip.times.append(float(t))
        i += 1
    cap.release()
    clip.n_decoded = len(clip.frames)
    if clip.duration_sec <= 0 and clip.times:
        clip.duration_sec = clip.times[-1] + 1.0 / sample_fps
    if max_frames and len(clip.frames) > max_frames:   # L1: uniform subsample to a cap
        pick = np.linspace(0, len(clip.frames) - 1, max_frames).round().astype(int)
        clip.frames = [clip.frames[p] for p in pick]
        clip.times = [clip.times[p] for p in pick]
    return clip
