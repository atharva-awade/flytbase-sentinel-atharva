"""S1 — always-on gate: SigLIP2 zero-shot class scores + normality distance + motion cues (PLAN.md §5.2).

Runs on 100% of sampled frames. Output is a GateResult with 1 Hz arrays that (a) feed fusion and
(b) select the candidate windows the VLM is woken for.

A `MockGate` (no model download) is provided so the pipeline runs on CPU/CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import CLASSES, NORMAL
from .ingest import Clip, night_lift
from .ledger import Ledger


@dataclass
class GateResult:
    T: int                                     # seconds
    class_scores: dict[str, np.ndarray]        # per class (incl. normal) softmax prob at 1 Hz, in [0,1]
    novelty: np.ndarray                        # 1 - max cos(e_t, normal bank), in [0,1]
    motion: np.ndarray                         # frame-diff energy, normalised
    stationarity: np.ndarray                   # 1 - ||e_t - e_{t-5}||, high = scene not changing
    suspicion: np.ndarray                      # fused u[t] in [0,1]
    embeddings: np.ndarray | None = None       # (N, D) fp16 — kept for --adapt-normal and the dashboard
    frame_times: list[float] = field(default_factory=list)

    def windows(self, tau: float, pad: float, max_wake_ratio: float, min_len: float = 4.0) -> list[tuple[float, float]]:
        """Candidate windows [a,b) in seconds. Caps total covered seconds at max_wake_ratio * T."""
        u = self.suspicion
        hot = u >= tau
        if hot.sum() > max_wake_ratio * self.T:      # too many: keep only the hottest seconds
            k = int(max_wake_ratio * self.T)
            thr = np.sort(u)[-k] if k > 0 else 1.1
            hot = u >= thr
        iv: list[list[float]] = []
        for t in np.where(hot)[0]:
            a, b = max(0, t - pad), min(self.T, t + 1 + pad)
            if iv and a <= iv[-1][1]:
                iv[-1][1] = max(iv[-1][1], b)
            else:
                iv.append([float(a), float(b)])
        return [(a, max(b, min(self.T, a + min_len))) for a, b in iv]


def _per_second(values: np.ndarray, times: list[float], T: int) -> np.ndarray:
    """Max-pool per-frame values into 1 Hz bins; forward-fill empty bins."""
    out = np.full(T, np.nan)
    for v, t in zip(values, times):
        i = min(T - 1, int(t))
        out[i] = v if np.isnan(out[i]) else max(out[i], v)
    if np.all(np.isnan(out)):
        return np.zeros(T)
    idx = np.where(~np.isnan(out))[0]
    out[: idx[0]] = out[idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        out[a + 1: b] = out[a]
    out[idx[-1]:] = out[idx[-1]]
    return out


def motion_energy(frames: list[np.ndarray], side: int = 112) -> np.ndarray:
    g = [cv2.cvtColor(cv2.resize(f, (side, side)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    m = np.zeros(len(g))
    for i in range(1, len(g)):
        m[i] = np.abs(g[i] - g[i - 1]).mean()
    if m.max() > 0:
        m = m / (np.percentile(m, 95) + 1e-6)
    return np.clip(m, 0, 1)


def red_fraction(f: np.ndarray) -> float:
    """Fraction of strongly red-dominant pixels (used only by the mock components)."""
    r, g, b = f[..., 0].astype(int), f[..., 1].astype(int), f[..., 2].astype(int)
    return float(np.mean((r > 150) & (r > g + 60) & (r > b + 60)))


def moving_mask(prev: np.ndarray, cur: np.ndarray, thresh: int = 25) -> np.ndarray:
    """Binary mask of moving pixels (for the Cerberus-style attention tint)."""
    a = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    d = cv2.absdiff(a, b)
    _, m = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
    return cv2.dilate(m, np.ones((7, 7), np.uint8))


class _BaseGate:
    def __init__(self, cfg: dict, prompt_bank_path: str | Path):
        self.cfg = cfg
        with open(prompt_bank_path) as f:
            self.prompts: dict[str, list[str]] = yaml.safe_load(f)
        self.normal_bank: np.ndarray | None = None
        nb = Path(cfg.get("normal_bank", "banks/normal.npy"))
        if nb.exists():
            self.normal_bank = np.load(nb).astype(np.float32)

    # --- to implement ---
    def embed_images(self, frames: list[np.ndarray], ledger: Ledger | None) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    # --- shared ---
    def _text_bank(self) -> tuple[list[str], np.ndarray]:
        if not hasattr(self, "_tb"):
            names, mats = [], []
            for name, ps in self.prompts.items():
                e = self.embed_texts(ps)
                e = e / np.linalg.norm(e, axis=1, keepdims=True)
                mats.append(e.mean(0) / np.linalg.norm(e.mean(0)))
                names.append(name)
            self._tb = (names, np.stack(mats))
        return self._tb

    def adapt_normal(self, embeddings: np.ndarray, keep: int = 2000) -> None:
        """Site normality memory: seed/extend the normal bank from frames the operator declares normal
        (e.g. the first 30 s of a new feed). PLAN.md §10 item 4."""
        e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normal_bank = e if self.normal_bank is None else np.concatenate([self.normal_bank, e])[-keep:]

    def run(self, clip: Clip, ledger: Ledger | None = None, logit_scale: float = 100.0) -> GateResult:
        T = max(1, int(np.ceil(clip.duration_sec)))
        frames_g = [night_lift(f, self.cfg.get("night_lum_thresh", 40)) for f in clip.frames]
        E = self.embed_images(frames_g, ledger).astype(np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        names, TB = self._text_bank()
        logits = logit_scale * E @ TB.T
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        class_scores = {n: _per_second(probs[:, i], clip.times, T) for i, n in enumerate(names)}

        if self.normal_bank is not None and len(self.normal_bank):
            sims = E @ self.normal_bank.T
            nov = 1.0 - sims.max(1)
            nov = np.clip((nov - 0.05) / 0.35, 0, 1)      # empirical squash; sweep on public set
        else:
            nov = 1.0 - probs[:, names.index(NORMAL)] if NORMAL in names else np.zeros(len(E))
        novelty = _per_second(nov, clip.times, T)

        mot = _per_second(motion_energy(clip.frames), clip.times, T)
        k = 5
        stat_f = np.ones(len(E))
        if len(E) > k:
            stat_f[k:] = 1.0 - np.clip(np.linalg.norm(E[k:] - E[:-k], axis=1) / 0.6, 0, 1)
        stationarity = _per_second(stat_f, clip.times, T)

        anomaly_max = np.max(np.stack([class_scores[c] for c in CLASSES if c in class_scores]), axis=0)
        # dwell cue: something vehicle-like scored + scene static + little motion
        dwell = stationarity * (1 - mot) * np.maximum(class_scores.get("stalled_or_broken_down_vehicle", 0),
                                                       class_scores.get("vehicle_blocking_traffic", 0))
        u = (self.cfg.get("w_class", 0.5) * anomaly_max + self.cfg.get("w_novelty", 0.3) * novelty
             + self.cfg.get("w_motion", 0.2) * np.maximum(mot * anomaly_max, dwell))
        return GateResult(T=T, class_scores=class_scores, novelty=novelty, motion=mot, stationarity=stationarity,
                          suspicion=np.clip(u, 0, 1), embeddings=E.astype(np.float16), frame_times=list(clip.times))


class SigLIP2Gate(_BaseGate):
    """Real gate. Requires torch + transformers (>=4.50) with SigLIP2 weights."""

    def __init__(self, cfg: dict, prompt_bank_path: str | Path):
        super().__init__(cfg, prompt_bank_path)
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(cfg["model_id"], torch_dtype=dtype).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(cfg["model_id"])
        self.bs = int(cfg.get("batch_size", 32))

    @property
    def name(self) -> str:
        return self.cfg["model_id"].split("/")[-1]

    def embed_images(self, frames, ledger=None):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(frames), self.bs):
                batch = self.proc(images=frames[i:i + self.bs], return_tensors="pt").to(self.device)
                if self.device == "cuda":
                    batch["pixel_values"] = batch["pixel_values"].half()
                if ledger:
                    with ledger.call(self.name):
                        e = self.model.get_image_features(**batch)
                        if self.device == "cuda":
                            self.torch.cuda.synchronize()
                else:
                    e = self.model.get_image_features(**batch)
                if hasattr(e, "pooler_output") and e.pooler_output is not None:
                    e = e.pooler_output
                out.append(e.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def embed_texts(self, texts):
        with self.torch.no_grad():
            batch = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            e = self.model.get_text_features(**batch)
            if hasattr(e, "pooler_output") and e.pooler_output is not None:
                e = e.pooler_output
            return e.float().cpu().numpy()


class MockGate(_BaseGate):
    """CPU-only stand-in: colour/motion heuristics projected into a fake embedding space.
    Lets the whole pipeline (windows -> VLM -> decoder -> submission) run without weights."""
    name = "mock-gate"

    def __init__(self, cfg: dict, prompt_bank_path: str | Path):
        super().__init__(cfg, prompt_bank_path)
        rng = np.random.default_rng(0)
        self._text = {n: rng.normal(size=(len(ps), 32)) for n, ps in self.prompts.items()}
        self.normal_bank = None

    def embed_texts(self, texts):
        for n, ps in self.prompts.items():
            if ps == texts:
                return self._text[n]
        return np.random.default_rng(1).normal(size=(len(texts), 32))

    def embed_images(self, frames, ledger=None):
        # Fake but *structured*: red-dominant frames look like fire, very static+dark like loitering, etc.
        names, TB = self._text_bank()
        E = []
        for f in frames:
            if ledger:
                with ledger.call("mock-gate"):
                    pass
            red = red_fraction(f)
            k = min(1.0, red / 0.004)          # >=0.4% strongly-red pixels -> fully 'fire'
            e = (1.0 - k) * TB[names.index(NORMAL)] + k * TB[names.index("fire")]
            E.append(e + np.random.default_rng(int(red * 1e5)).normal(scale=0.02, size=TB.shape[1]))
        return np.stack(E) if E else np.zeros((0, TB.shape[1]))


def build_gate(cfg: dict, prompt_bank_path: str | Path, mock: bool = False) -> _BaseGate:
    return MockGate(cfg, prompt_bank_path) if mock else SigLIP2Gate(cfg, prompt_bank_path)
