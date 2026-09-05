"""Every 'thing that catches people out' from the submission doc has a failing fixture here."""
import json

from sentinel.schema import ManifestEntry
from sentinel.validate import is_acceptable, validate_submission

MANIFEST = [ManifestEntry(video_id="E001", level=1), ManifestEntry(video_id="E009", level=1),
            ManifestEntry(video_id="E021", level=2), ManifestEntry(video_id="E030", level=3)]
RT = {"frames_processed": 10, "chunks_processed": 1, "end_to_end_internal_time_ms": 100.0, "model_runtimes": []}


def sub(*preds):
    return {"predictions": list(preds)}


def fields(problems):
    return {(p.video_id, p.field) for p in problems if p.field != "info"}


def test_valid_minimal_passes():
    s = sub({"video_id": "E001", "events": [], "runtime_metadata": RT},
            {"video_id": "E021", "events": [{"class_name": "fire", "start_time_sec": 3, "end_time_sec": 9,
                                              "explanation": "Flames and smoke rise from a vehicle on the shoulder."}],
             "runtime_metadata": RT})
    assert is_acceptable(validate_submission(s, MANIFEST))


def test_normal_class_rejected():
    s = sub({"video_id": "E001", "events": [{"class_name": "normal"}], "runtime_metadata": RT})
    assert ("E001", "events[0].class_name") in fields(validate_submission(s, MANIFEST))


def test_l1_timestamps_rejected():
    s = sub({"video_id": "E009", "events": [{"class_name": "fire", "start_time_sec": 1, "end_time_sec": 5}], "runtime_metadata": RT})
    assert ("E009", "events[0].start_time_sec") in fields(validate_submission(s, MANIFEST))


def test_l2_missing_timestamps_rejected():
    s = sub({"video_id": "E021", "events": [{"class_name": "fire"}], "runtime_metadata": RT})
    assert ("E021", "events[0].start_time_sec") in fields(validate_submission(s, MANIFEST))


def test_end_not_greater_than_start():
    s = sub({"video_id": "E021", "events": [{"class_name": "fire", "start_time_sec": 5, "end_time_sec": 5}], "runtime_metadata": RT})
    assert ("E021", "events[0].end_time_sec") in fields(validate_submission(s, MANIFEST))


def test_duplicate_and_unknown_video():
    s = sub({"video_id": "E001", "events": [], "runtime_metadata": RT},
            {"video_id": "E001", "events": [], "runtime_metadata": RT},
            {"video_id": "E999", "events": [], "runtime_metadata": RT})
    f = fields(validate_submission(s, MANIFEST))
    assert ("E001", "video_id") in f and ("E999", "video_id") in f


def test_explanation_length():
    s = sub({"video_id": "E021", "events": [{"class_name": "fire", "start_time_sec": 1, "end_time_sec": 5, "explanation": "too short"}], "runtime_metadata": RT})
    assert ("E021", "events[0].explanation") in fields(validate_submission(s, MANIFEST))


def test_average_time_mismatch_and_call_times_len():
    rt = dict(RT, model_runtimes=[{"model_name": "vlm", "call_count": 4, "total_time_ms": 1000, "average_time_ms": 300,
                                   "call_times_ms": [1, 2, 3]}])
    s = sub({"video_id": "E001", "events": [], "runtime_metadata": rt})
    f = fields(validate_submission(s, MANIFEST))
    assert ("E001", "runtime_metadata.model_runtimes[0].average_time_ms") in f
    assert ("E001", "runtime_metadata.model_runtimes[0].call_times_ms") in f


def test_average_within_two_percent_ok():
    rt = dict(RT, model_runtimes=[{"model_name": "vlm", "call_count": 4, "total_time_ms": 1000, "average_time_ms": 252}])
    s = sub({"video_id": "E001", "events": [], "runtime_metadata": rt})
    assert is_acceptable(validate_submission(s, MANIFEST))


def test_missing_runtime_metadata_rejected():
    s = sub({"video_id": "E001", "events": []})
    probs = validate_submission(s, MANIFEST)
    assert any("runtime_metadata" in p.field for p in probs)


def test_unknown_extra_fields_ignored():
    s = sub({"video_id": "E021", "events": [{"class_name": "fire", "start_time_sec": 1, "end_time_sec": 5,
                                              "confidence": 0.9, "bbox": [1, 2, 3, 4], "responder_action": "x"}],
             "runtime_metadata": RT, "debug": {"a": 1}})
    assert is_acceptable(validate_submission(json.loads(json.dumps(s)), MANIFEST))
