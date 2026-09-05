"""SENTINEL live application server.

  uvicorn app.server:app --host 0.0.0.0 --port 8080            (real models: SigLIP2 + vLLM per configs/default.yaml)
  SENTINEL_MOCK=1 uvicorn app.server:app --port 8080            (CPU mock models — UI/flow identical)

Sources: uploaded video · local file · webcam index · RTSP/HTTP URL. Each session runs the streaming cascade
(sentinel.stream.StreamProcessor) in a thread, publishes MJPEG with overlays and a JSON state WebSocket,
and can export an arena-format prediction for the processed video.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sentinel.gate import build_gate
from sentinel.pipeline import load_config
from sentinel.schema import Submission
from sentinel.stream import StreamProcessor
from sentinel.verifier import Verifier

ROOT = Path(__file__).resolve().parent.parent
MOCK = os.environ.get("SENTINEL_MOCK", "0") == "1"
CFG = load_config(os.environ.get("SENTINEL_CONFIG", ROOT / "configs" / "default.yaml"))
if MOCK:
    CFG["verifier"]["backend"] = "mock"
UPLOADS = ROOT / "data" / "uploads"; UPLOADS.mkdir(parents=True, exist_ok=True)
SAMPLES = ROOT / "data" / "samples"; SAMPLES.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SENTINEL", version="0.2.0")

_models_lock = threading.Lock()
_GATE = None
_VERIFIER = None


def models():
    """Shared, lazily-loaded gate + verifier (one copy of SigLIP2 per process)."""
    global _GATE, _VERIFIER
    with _models_lock:
        if _GATE is None:
            _GATE = build_gate(CFG["gate"], ROOT / CFG["paths"]["prompt_bank"], mock=MOCK)
            _VERIFIER = Verifier(CFG["verifier"], str(ROOT / CFG["paths"]["question_bank"]))
    return _GATE, _VERIFIER


class Session:
    def __init__(self, source: str, kind: str, name: str, level: int, realtime: bool, loop: bool, speed: float):
        self.id = uuid.uuid4().hex[:8]
        self.source, self.kind, self.name, self.level = source, kind, name, level
        self.realtime, self.loop, self.speed = realtime, loop, speed
        gate, verifier = models()
        self.proc = StreamProcessor(CFG, gate, verifier, level=level)
        self.jpeg: bytes | None = None
        self.state: dict = {}
        self.status = "starting"
        self.error = ""
        self.duration = 0.0
        self.width = self.height = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "webcam":
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source {self.source!r}")
        return cap

    def _run(self) -> None:
        try:
            cap = self._open()
        except Exception as ex:  # noqa: BLE001
            self.status, self.error = "error", str(ex); return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = n / fps if n > 0 else 0.0
        self.width, self.height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_live = self.kind in ("webcam", "url")
        self.status = "running"
        t_media, wall0, i = 0.0, time.perf_counter(), 0
        proc_every = 1 if is_live else max(1, int(round(fps / 12.0)))     # process ≤12 fps of overlays from files
        while not self._stop.is_set():
            ok, bgr = cap.read()
            if not ok:
                if self.kind == "file" and self.loop:
                    cap.release(); cap = self._open(); i = 0; wall0 = time.perf_counter() - t_media / self.speed; continue
                break
            if is_live:
                t_media = time.perf_counter() - wall0
            else:
                t_media = i / fps
            i += 1
            if not is_live and i % proc_every:
                if self.realtime:
                    target = wall0 + t_media / self.speed
                    d = target - time.perf_counter()
                    if d > 0: time.sleep(min(d, 0.05))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if max(h, w) > 960:
                s = 960 / max(h, w); rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            out = self.proc.push(rgb, t_media)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                self.jpeg = buf.tobytes()
            st = self.proc.state.to_json()
            st.update(session=self.id, status=self.status, duration=self.duration, name=self.name, level=self.level,
                      kind=self.kind, mock=MOCK)
            self.state = st
            if not is_live and self.realtime:
                target = wall0 + t_media / self.speed
                d = target - time.perf_counter()
                if d > 0: time.sleep(min(d, 0.1))
        cap.release()
        self.status = "stopped" if self._stop.is_set() else "finished"
        if self.state:
            self.state["status"] = self.status
        self.proc.close()

    def stop(self) -> None:
        self._stop.set()


SESSIONS: dict[str, Session] = {}


def _ensure_samples() -> None:
    if any(SAMPLES.glob("*.mp4")):
        return
    import sys
    sys.path.insert(0, str(ROOT))
    from tests.synth import make_video
    make_video(str(SAMPLES / "sample_normal_traffic.mp4"), 40, size=(640, 360))
    make_video(str(SAMPLES / "sample_fire_event.mp4"), 60, size=(640, 360), fire_windows=[(20, 35)])
    make_video(str(SAMPLES / "sample_two_events.mp4"), 120, size=(640, 360), fire_windows=[(10, 30), (80, 100)])


# ------------------------------------------------------------------ API
@app.get("/api/health")
def health():
    return {"ok": True, "mock": MOCK, "gate": CFG["gate"]["model_id"], "verifier": CFG["verifier"]["model"],
            "backend": CFG["verifier"]["backend"]}


@app.get("/api/samples")
def samples():
    _ensure_samples()
    return [{"name": p.name, "path": str(p)} for p in sorted(SAMPLES.glob("*.mp4"))]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    dest = UPLOADS / f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest), "name": Path(file.filename).name}


@app.post("/api/sessions")
def create_session(kind: str = Form(...), source: str = Form(...), level: int = Form(3), realtime: bool = Form(True),
                   loop: bool = Form(False), speed: float = Form(1.0), name: str = Form("")):
    if kind == "file" and not Path(source).exists():
        raise HTTPException(404, f"file not found: {source}")
    s = Session(source, kind, name or Path(source).name, level, realtime, loop, speed)
    SESSIONS[s.id] = s
    return {"id": s.id}


@app.get("/api/sessions")
def list_sessions():
    return [{"id": s.id, "name": s.name, "kind": s.kind, "status": s.status, "level": s.level} for s in SESSIONS.values()]


@app.post("/api/sessions/{sid}/stop")
def stop_session(sid: str):
    SESSIONS[sid].stop(); return {"ok": True}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    s = SESSIONS.pop(sid, None)
    if s: s.stop()
    return {"ok": True}


@app.post("/api/sessions/{sid}/adapt-normal")
def adapt_normal(sid: str):
    n = SESSIONS[sid].proc.adapt_normal()
    return {"normal_bank_size": n}


@app.get("/api/sessions/{sid}/state")
def state(sid: str):
    s = SESSIONS[sid]
    return s.state or {"status": s.status, "error": s.error}


@app.get("/api/sessions/{sid}/export")
def export(sid: str, video_id: str | None = None, level: int | None = None):
    s = SESSIONS[sid]
    pred = s.proc.to_prediction(video_id or Path(s.name).stem, level or s.level)
    sub = Submission(submission_id=f"sentinel-live-{sid}", model_name=CFG["emit"]["model_name"], predictions=[pred])
    return JSONResponse(json.loads(sub.to_json()))


@app.get("/stream/{sid}.mjpg")
async def mjpeg(sid: str):
    s = SESSIONS[sid]

    async def gen():
        last = None
        while sid in SESSIONS:
            if s.jpeg is not None and s.jpeg is not last:
                last = s.jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(last)).encode() + b"\r\n\r\n" + last + b"\r\n"
            elif s.status in ("finished", "stopped", "error") and s.jpeg is last:
                await asyncio.sleep(0.5)
                if s.status == "error":
                    break
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/{sid}")
async def ws(sock: WebSocket, sid: str):
    await sock.accept()
    try:
        while sid in SESSIONS:
            s = SESSIONS[sid]
            await sock.send_text(json.dumps(s.state or {"status": s.status, "error": s.error}))
            if s.status in ("finished", "stopped", "error") and s.state:
                await asyncio.sleep(1.0)
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return


STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
