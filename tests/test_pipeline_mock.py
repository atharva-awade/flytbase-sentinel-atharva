"""End-to-end on synthetic videos with MockGate + MockVLM -> valid, well-shaped submission."""
import copy
import json

import pytest

from sentinel.manifest import load_manifest
from sentinel.pipeline import Pipeline, load_config
from sentinel.schema import GTEvent, ManifestEntry
from sentinel.scorer import score
from sentinel.validate import is_acceptable, validate_submission
from tests.synth import make_video


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("vids")
    files = {"E001": str(d / "E001.mp4"), "E002": str(d / "E002.mp4"), "E003": str(d / "E003.mp4"), "E004": str(d / "E004.mp4")}
    make_video(files["E001"], 20)                                    # L1 normal
    make_video(files["E002"], 20, fire_windows=[(0, 20)])            # L1 fire
    make_video(files["E003"], 60, fire_windows=[(20, 35)])           # L2 one fire 20-35
    make_video(files["E004"], 120, fire_windows=[(10, 30), (80, 100)])  # L3 two fires
    manifest = [ManifestEntry(video_id="E001", level=1), ManifestEntry(video_id="E002", level=1),
                ManifestEntry(video_id="E003", level=2), ManifestEntry(video_id="E004", level=3)]
    gt = [GTEvent(video_id="E001", level=1, is_anomaly=False, class_name="normal"),
          GTEvent(video_id="E002", level=1, is_anomaly=True, class_name="fire"),
          GTEvent(video_id="E003", level=2, is_anomaly=True, class_name="fire", start_time_sec=20, end_time_sec=35),
          GTEvent(video_id="E004", level=3, is_anomaly=True, class_name="fire", start_time_sec=10, end_time_sec=30),
          GTEvent(video_id="E004", level=3, is_anomaly=True, class_name="fire", start_time_sec=80, end_time_sec=100)]
    cfg = load_config("configs/default.yaml")
    cfg["paths"]["cache_dir"] = str(d / "cache")
    pipe = Pipeline(copy.deepcopy(cfg), mock=True)
    sub = pipe.run(manifest, files)
    return manifest, gt, sub, pipe


def test_submission_valid(world):
    manifest, gt, sub, pipe = world
    probs = validate_submission(sub, manifest)
    assert is_acceptable(probs), [str(p) for p in probs]
    assert len(json.loads(sub.to_json())["predictions"]) == 4


def test_shapes_and_scores(world):
    manifest, gt, sub, pipe = world
    by = {p.video_id: p for p in sub.predictions}
    assert by["E001"].events == []                                   # normal stays silent
    assert [e.class_name for e in by["E002"].events] == ["fire"] and by["E002"].events[0].start_time_sec is None
    assert len(by["E003"].events) == 1 and by["E003"].events[0].class_name == "fire"
    assert len(by["E004"].events) == 2                               # two events, no fragments
    for p in sub.predictions:
        for e in p.events:
            assert e.explanation and 20 <= len(e.explanation) <= 500
            assert e.responder_action
        assert p.runtime_metadata.model_runtimes and p.runtime_metadata.end_to_end_internal_time_ms > 0
    rep = score(sub.predictions, gt, {"E001": 20, "E002": 20, "E003": 60, "E004": 120})
    print(rep.summary())
    assert rep.level_points[1] == 25
    assert rep.videos and all(not v.false_alarm for v in rep.videos)
    assert all(v.matched == v.n_gt for v in rep.videos if v.level > 1), rep.summary()
    assert rep.total_points > 85


def test_wake_ratio_is_partial(world):
    _, _, sub, pipe = world
    assert 0 < sub.run_metadata.vlm_wake_ratio < 1.0                 # cascade woke the VLM on a fraction only


def test_manifest_loader_tolerates_shapes(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"videos": [{"id": "E001", "difficulty": "L2", "file": "E001.mp4"}, {"video_id": "E002", "level": 1}]}))
    m = load_manifest(p)
    assert (m[0].video_id, m[0].level, m[0].file) == ("E001", 2, "E001.mp4") and m[1].level == 1
