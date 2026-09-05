import time

from sentinel.ledger import Ledger
from sentinel.schema import ManifestEntry, Submission, Prediction
from sentinel.validate import is_acceptable, validate_submission


def test_ledger_average_matches_total_over_count():
    L = Ledger()
    with L.video():
        for ms in (5, 7, 9):
            L.record("siglip2", ms)
        with L.call("vlm"):
            time.sleep(0.01)
    md = L.runtime_metadata(frames_processed=3, chunks_processed=1)
    assert md.end_to_end_internal_time_ms >= 10
    by = {m.model_name: m for m in md.model_runtimes}
    assert by["siglip2"].call_count == 3 and abs(by["siglip2"].average_time_ms - 7.0) < 1e-6
    assert by["siglip2"].p50_time_ms == 7.0 and by["siglip2"].max_time_ms == 9.0
    sub = Submission(predictions=[Prediction(video_id="E001", events=[], runtime_metadata=md)])
    assert is_acceptable(validate_submission(sub, [ManifestEntry(video_id="E001", level=1)]))
