"""Streaming cascade on a synthetic clip with mock models: live alert appears, export validates."""
import cv2

from sentinel.gate import build_gate
from sentinel.pipeline import load_config
from sentinel.schema import ManifestEntry, Submission
from sentinel.stream import StreamProcessor
from sentinel.validate import is_acceptable, validate_submission
from sentinel.verifier import Verifier
from tests.synth import make_video


def test_stream_live_alert_and_export(tmp_path):
    p = tmp_path / "v.mp4"
    make_video(str(p), 50, size=(640, 360), fire_windows=[(15, 30)])
    cfg = load_config("configs/default.yaml"); cfg["verifier"]["backend"] = "mock"
    g = build_gate(cfg["gate"], "configs/prompt_bank.yaml", mock=True)
    v = Verifier(cfg["verifier"], "configs/question_bank.yaml")
    sp = StreamProcessor(cfg, g, v, level=3)
    cap = cv2.VideoCapture(str(p)); i = 0; saw_live = False; first_alert_t = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 2 == 0:
            out = sp.push(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), i / 10.0)
            assert out.shape == f.shape
            if sp.state.active and sp.state.active.open:
                saw_live = True
                first_alert_t = first_alert_t if first_alert_t is not None else i / 10.0
        i += 1
    # let the async VLM jobs finish, then one more decode
    sp.pool.shutdown(wait=True)
    sp._decode(len(sp.sec_probs)); sp._refresh_state(50.0)
    # headless pushes frames far faster than real time, so async verdicts land "late" in media time; in real-time
    # playback the alert follows onset by ~1 gate tick + 1 VLM call (~3-5 s).
    assert saw_live and 15 <= first_alert_t <= 35, first_alert_t
    assert [e.class_name for e in sp.state.events] == ["fire"]
    pred = sp.to_prediction("V1", 3)
    sub = Submission(predictions=[pred])
    assert is_acceptable(validate_submission(sub, [ManifestEntry(video_id="V1", level=3)]))
    e = pred.events[0]
    assert e.start_time_sec <= 20 and e.end_time_sec >= 28 and 0 < sp.state.wake_ratio < 1
