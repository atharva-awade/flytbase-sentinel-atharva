"""Orchestrates S0-S4 per video and per level; caches; builds the Submission (PLAN.md §4, §13)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from . import CLASSES
from .decoder import decode, decode_l1, load_priors
from .emit import to_prediction
from .fusion import Verdict, best_explanations, fuse
from .gate import GateResult, build_gate
from .ingest import load_clip
from .ledger import Ledger
from .schema import ManifestEntry, Prediction, RunMetadata, Submission
from .verifier import Verifier


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Pipeline:
    def __init__(self, cfg: dict, mock: bool = False, gate=None, verifier=None):
        self.cfg = cfg
        self.mock = mock
        if mock:
            cfg["verifier"]["backend"] = "mock"
        self.gate = gate or build_gate(cfg["gate"], cfg["paths"]["prompt_bank"], mock=mock)
        self.verifier = verifier or Verifier(cfg["verifier"], cfg["paths"]["question_bank"])
        self.priors = load_priors(cfg["paths"].get("priors"))
        self.cache = Path(cfg["paths"]["cache_dir"]); self.cache.mkdir(parents=True, exist_ok=True)
        self.trace: dict[str, dict] = {}   # per-video debug info for the dashboard / write-up

    # ---------- single video ----------
    def run_video(self, entry: ManifestEntry, path: str) -> Prediction:
        L = Ledger()
        ic, gc, vc, fc = self.cfg["ingest"], self.cfg["gate"], self.cfg["verifier"], self.cfg["fusion"]
        with L.video():
            if entry.level == 1:
                clip = load_clip(entry.video_id, path, ic["sample_fps_l1"], ic["max_side"], ic["max_frames_l1"])
            else:
                clip = load_clip(entry.video_id, path, ic["sample_fps_l23"], ic["max_side"])
            gate: GateResult = self.gate.run(clip, L)
            T = gate.T

            if entry.level == 1:
                v = self.verifier.verify_clip_l1(clip, gate, L)
                verdicts = [v]
                windows = [(0.0, float(clip.duration_sec))]
                curves = fuse(gate.class_scores, verdicts, T, fc["alpha"], fc["beta"])
                l1 = decode_l1(curves, self.priors)
                # L1 abstention: verdict normal wins over gate prior
                if v.class_name == "normal":
                    l1 = None
                intervals = None
                l1_conf = v.confidence if l1 else None
            else:
                windows = gate.windows(gc["tau_gate"], gc["pad_sec"], gc["max_wake_ratio"])
                verdicts = self.verifier.verify_windows(clip, gate, windows, L) if windows else []
                curves = fuse(gate.class_scores, verdicts, T, fc["alpha"], fc["beta"])
                intervals = decode(curves, entry.level, self.priors, clip.duration_sec, best_explanations(verdicts))
                l1, l1_conf = None, None
            chunks = max(1, len(windows))
        rt = L.runtime_metadata(frames_processed=clip.n_decoded, chunks_processed=chunks)
        pred = to_prediction(entry.video_id, entry.level, intervals, l1, rt, best_explanations(verdicts), l1_conf)

        covered = sum(b - a for a, b in windows) if entry.level != 1 else clip.duration_sec
        self.trace[entry.video_id] = {
            "level": entry.level, "duration_sec": clip.duration_sec, "T": T,
            "suspicion": gate.suspicion.round(3).tolist(), "novelty": gate.novelty.round(3).tolist(),
            "motion": gate.motion.round(3).tolist(),
            "curves": {c: curves[c].round(3).tolist() for c in CLASSES if curves[c].max() > 0.05},
            "windows": windows, "verdicts": [asdict(v) for v in verdicts],
            "wake_ratio": min(1.0, covered / max(clip.duration_sec, 1e-6)),
            "events": [e.model_dump(exclude_none=True) for e in pred.events],
            "runtime_ms": rt.end_to_end_internal_time_ms,
        }
        np.savez_compressed(self.cache / f"{entry.video_id}.npz", suspicion=gate.suspicion,
                            **{f"cls_{c}": v for c, v in gate.class_scores.items()})
        return pred

    # ---------- many videos ----------
    def run(self, entries: list[ManifestEntry], files: dict[str, str], levels: set[int] | None = None,
            submission_id: str | None = None, progress=None) -> Submission:
        t0 = time.perf_counter()
        preds: list[Prediction] = []
        for e in entries:
            if levels and e.level not in levels:
                continue
            if e.video_id not in files:
                continue
            p = self.run_video(e, files[e.video_id])
            preds.append(p)
            if progress:
                progress(e, p, self.trace[e.video_id])
        ec = self.cfg["emit"]
        wake = [t["wake_ratio"] for t in self.trace.values() if t["level"] != 1]
        sid = submission_id or f"{ec['submission_id_prefix']}-{time.strftime('%H%M%S')}"
        return Submission(submission_id=sid, model_name=ec["model_name"],
                          run_metadata=RunMetadata(total_wall_time_ms=round((time.perf_counter() - t0) * 1000, 1),
                                                   hardware=ec["hardware"],
                                                   vlm_wake_ratio=round(float(np.mean(wake)), 3) if wake else None),
                          predictions=preds)

    def save_trace(self, path: str | Path) -> None:
        def _np(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError(type(o))
        Path(path).write_text(json.dumps(self.trace, default=_np))


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
