"""S2 — VLM verifier (PLAN.md §5.3).

VLMClient speaks the OpenAI chat-completions protocol, so the same code drives:
  * vLLM        (Qwen3-VL-4B-Instruct [+ our LoRA] on T4/L4)         -> production
  * llama.cpp   (Qwen3-VL-4B Q4_K_M GGUF + mmproj)                    -> edge / Jetson demo
  * NVIDIA NIM  (development-time comparison ONLY — never at runtime)
MockVLM returns scripted verdicts from cheap image statistics so CPU/CI runs end-to-end.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import CLASSES
from .fusion import Verdict
from .gate import GateResult, red_fraction
from .ingest import Clip
from .ledger import Ledger
from .prompts import RESPONSE_SCHEMA, build_messages, load_question_bank, parse_json, select_questions


@dataclass
class RawAnswer:
    class_name: str
    confidence: float
    onset_frame: int
    explanation: str
    answers: list[dict]
    latency_ms: float


class VLMClient:
    def __init__(self, cfg: dict):
        from openai import OpenAI
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg["base_url"], api_key=cfg.get("api_key", "EMPTY"),
                             timeout=cfg.get("timeout_s", 60))
        self.model = cfg["model"]
        self.name = self.model.split("/")[-1]
        self._guided = True   # vLLM supports response_format json_schema; fall back if the server rejects it

    def ask(self, messages: list[dict], temperature: float = 0.0) -> RawAnswer:
        t0 = time.perf_counter()
        kwargs = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=400)
        if self._guided:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "verdict", "schema": RESPONSE_SCHEMA}}
        try:
            r = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # server without guided decoding
            if self._guided and ("response_format" in str(e) or "json_schema" in str(e) or "400" in str(e)):
                self._guided = False
                kwargs.pop("response_format", None)
                r = self.client.chat.completions.create(**kwargs)
            else:
                raise
        text = r.choices[0].message.content or "{}"
        d = parse_json(text)
        return RawAnswer(
            class_name=str(d.get("class_name", "normal")),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            onset_frame=int(d.get("onset_frame", 0) or 0),
            explanation=str(d.get("explanation", "")),
            answers=list(d.get("answers", [])),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class MockVLM:
    """Scripted verifier: red-dominant frames -> fire; gate-informed zero-shot; otherwise normal."""
    name = "mock-vlm"

    def __init__(self, cfg: dict | None = None, script: dict[str, str] | None = None):
        self.script = script or {}

    def ask(self, messages: list[dict], temperature: float = 0.0, _frames=None,
            _top: list[str] | None = None, _gate_scores: dict[str, float] | None = None,
            _qs: list[tuple[str, str]] | None = None) -> RawAnswer:
        t0 = time.perf_counter()
        cls, conf, expl = "normal", 0.9, "Traffic is flowing normally; no obstruction, fire or incident is visible."
        if _frames is not None:
            reds = [red_fraction(f) for f in _frames]
            if max(reds) > 0.005:
                cls, conf = "fire", 0.92
                expl = "Bright orange-red flames are visible on the road surface with vehicles stopped nearby."
                onset = int(np.argmax(np.asarray(reds) > 0.005))
                time.sleep(0.002)
                return RawAnswer(cls, conf, onset, expl, [], (time.perf_counter() - t0) * 1000)
        if _top and _gate_scores:
            top_cls = _top[0]
            top_score = _gate_scores.get(top_cls, 0.0)
            norm_score = _gate_scores.get("normal", 0.0)
            if top_score > norm_score and top_score >= 0.25:
                cls = top_cls
                conf = min(0.95, float(top_score * 1.6))
                expl = f"Visual evidence indicates {top_cls.replace('_', ' ')} with high zero-shot gate confidence."
                answers = []
                if _qs:
                    for i, (q_cls, _) in enumerate(_qs):
                        is_yes = (q_cls == top_cls)
                        answers.append({"q": i, "yes": is_yes, "conf": top_score if is_yes else 0.8})
                time.sleep(0.002)
                return RawAnswer(cls, conf, 0, expl, answers, (time.perf_counter() - t0) * 1000)
        time.sleep(0.002)
        return RawAnswer(cls, conf, 0, expl, [], (time.perf_counter() - t0) * 1000)


# ---- deterministic disambiguation (PLAN.md §5.3) ----
def disambiguate(ans: RawAnswer, questions: list[tuple[str, str]]) -> str:
    yes = {}
    for a in ans.answers:
        try:
            cls, q = questions[int(a["q"])]
            yes.setdefault(cls, []).append(bool(a.get("yes")) and float(a.get("conf", 0)) >= 0.5)
        except (IndexError, KeyError, ValueError, TypeError):
            continue
    c = ans.class_name
    if c not in CLASSES:
        return "normal"
    def any_yes(k): return any(yes.get(k, []))
    if c == "fire" and yes.get("fire") and not yes["fire"][0]:       # first fire q = visible flames
        return "smoke" if any_yes("smoke") else "normal"
    if c == "vehicle_blocking_traffic" and yes.get("vehicle_blocking_traffic") and not any_yes("vehicle_blocking_traffic"):
        return "stalled_or_broken_down_vehicle" if any_yes("stalled_or_broken_down_vehicle") else c
    if c == "traffic_congestion" and (any_yes("vehicle_blocking_traffic") or any_yes("stalled_or_broken_down_vehicle")) \
            and not any_yes("traffic_congestion"):
        return "vehicle_blocking_traffic" if any_yes("vehicle_blocking_traffic") else "stalled_or_broken_down_vehicle"
    if c != "normal" and yes.get(c) and not any_yes(c):
        return "normal"
    # explicit normal evidence overrides a low-confidence anomaly
    if yes.get("normal") and all(yes["normal"]) and ans.confidence < 0.75:
        return "normal"
    return c


class Verifier:
    def __init__(self, cfg: dict, question_bank_path: str, client=None):
        self.cfg = cfg
        self.qb = load_question_bank(question_bank_path)
        self.client = client or (MockVLM(cfg) if cfg.get("backend") == "mock" else VLMClient(cfg))
        self.is_mock = isinstance(self.client, MockVLM)

    def _top_classes(self, gate: GateResult, a: float, b: float) -> list[str]:
        s, e = int(a), max(int(a) + 1, int(np.ceil(b)))
        means = {c: float(np.mean(gate.class_scores[c][s:e])) for c in CLASSES if c in gate.class_scores}
        return [c for c, _ in sorted(means.items(), key=lambda kv: -kv[1])[: self.cfg.get("top_k_classes", 4)]]

    def _ask(self, msgs, temperature, frames, top=None, gate_scores=None, qs=None):
        if self.is_mock:
            return self.client.ask(msgs, temperature, _frames=frames, _top=top, _gate_scores=gate_scores, _qs=qs)
        try:
            return self.client.ask(msgs, temperature)
        except Exception as e:
            if not getattr(self, "_logged_fallback", False):
                import warnings
                warnings.warn(f"VLM server unreachable ({e}); falling back to SigLIP-2 Gate zero-shot verifier.")
                self._logged_fallback = True
            self.client = MockVLM(self.cfg)
            self.is_mock = True
            return self.client.ask(msgs, temperature, _frames=frames, _top=top, _gate_scores=gate_scores, _qs=qs)

    def verify_window(self, clip: Clip, gate: GateResult, a: float, b: float, ledger: Ledger | None) -> Verdict:
        k = int(self.cfg.get("frames_per_window", 6))
        frames, times = clip.frames_between(a, b, k)
        top = self._top_classes(gate, a, b)
        qs = select_questions(self.qb, top)
        s, e = int(a), max(int(a) + 1, int(np.ceil(b)))
        gate_scores = {c: float(np.mean(gate.class_scores[c][s:e])) for c in list(CLASSES) + ["normal"] if c in gate.class_scores}
        msgs = build_messages(frames, times, qs, motion_tint=self.cfg.get("motion_tint", True),
                              max_side=self.cfg.get("max_side", 448))
        a0 = self._ask(msgs, 0.0, frames, top=top, gate_scores=gate_scores, qs=qs)
        if ledger: ledger.record(self.client.name, a0.latency_ms)
        cls = disambiguate(a0, qs)
        conf = a0.confidence
        if self.cfg.get("self_consistency", True) and cls != "normal":
            a1 = self._ask(msgs, float(self.cfg.get("temperature_probe", 0.6)), frames, top=top, gate_scores=gate_scores, qs=qs)
            if ledger: ledger.record(self.client.name, a1.latency_ms)
            cls1 = disambiguate(a1, qs)
            if cls1 != cls:                      # disagreement -> abstain
                cls, conf = "normal", min(conf, a1.confidence) * 0.5
            else:
                conf = 0.5 * (conf + a1.confidence)
        if cls != "normal" and conf < float(self.cfg.get("min_conf", 0.55)):
            cls = "normal"
        onset = times[min(max(a0.onset_frame, 0), len(times) - 1)] if times else None
        return Verdict(start=a, end=b, class_name=cls, confidence=conf, explanation=a0.explanation, onset_sec=onset)

    def verify_windows(self, clip: Clip, gate: GateResult, windows: list[tuple[float, float]],
                       ledger: Ledger | None) -> list[Verdict]:
        # split long candidate regions into sliding sub-windows
        W, S = float(self.cfg.get("window_sec", 12)), float(self.cfg.get("stride_sec", 6))
        subs: list[tuple[float, float]] = []
        for a, b in windows:
            if b - a <= W * 1.25:
                subs.append((a, b))
            else:
                t = a
                while t < b:
                    subs.append((t, min(b, t + W))); t += S
        n = int(self.cfg.get("max_concurrency", 2))
        if n <= 1 or self.is_mock:
            return [self.verify_window(clip, gate, a, b, ledger) for a, b in subs]
        with ThreadPoolExecutor(n) as ex:
            return list(ex.map(lambda ab: self.verify_window(clip, gate, ab[0], ab[1], ledger), subs))

    def verify_clip_l1(self, clip: Clip, gate: GateResult, ledger: Ledger | None) -> Verdict:
        return self.verify_window(clip, gate, 0.0, float(clip.duration_sec), ledger)
