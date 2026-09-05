"""Synthetic drone-like videos for CPU tests and demos: grey road, moving 'cars', optional fire burst."""
from __future__ import annotations

import cv2
import numpy as np


def make_video(path: str, duration: int, fps: int = 10, size=(320, 240), fire_windows: list[tuple[int, int]] | None = None,
               stall_at: int | None = None, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    w, h = size
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cars = [[rng.integers(0, w), 60 + 30 * i, rng.integers(2, 5)] for i in range(4)]
    for f in range(duration * fps):
        t = f / fps
        img = np.full((h, w, 3), 90, np.uint8)                       # asphalt
        cv2.rectangle(img, (0, 40), (w, 200), (70, 70, 70), -1)      # road
        for i, c in enumerate(cars):
            if not (stall_at is not None and i == 0 and t >= stall_at):
                c[0] = (c[0] + c[2]) % w
            cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[0]) + 18, int(c[1]) + 10), (200, 200, 200), -1)
        if fire_windows and any(a <= t < b for a, b in fire_windows):
            cx, cy = w // 2, 120
            r = 25 + int(8 * np.sin(t * 7))
            cv2.circle(img, (cx, cy), r, (0, 60, 255), -1)          # BGR: orange-red blob
            cv2.circle(img, (cx, cy - 10), r // 2, (0, 200, 255), -1)
        out.write(img)
    out.release()
