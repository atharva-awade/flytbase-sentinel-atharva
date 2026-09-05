import numpy as np

from sentinel import CLASSES
from sentinel.decoder import decode, decode_l1, hysteresis, merge_gaps, shape_interval
from sentinel.fusion import Verdict, fuse
from sentinel.scorer import iou


def curves(T, **kw):
    c = {k: np.zeros(T) for k in CLASSES}
    for k, v in kw.items():
        c[k] = np.asarray(v, dtype=float)
    return c


def test_hysteresis_and_merge():
    x = np.array([0, 0.7, 0.5, 0.4, 0.2, 0.7, 0.8, 0.1])
    assert hysteresis(x, 0.6, 0.35) == [(1, 4), (5, 7)]
    assert merge_gaps([(1, 4), (5, 7)], gap=1) == [(1, 7)]
    assert merge_gaps([(1, 4), (6, 7)], gap=1) == [(1, 4), (6, 7)]


def test_shape_interval_extends_symmetrically_within_clip():
    # tiny fragment (2 s) of a 12 s-median class -> grows to 70% of median = 8.4 s, symmetric, clipped to clip
    s, e = shape_interval(10, 12, 12, 60); assert abs((e - s) - 8.4) < 1e-9 and abs((s + e) / 2 - 11) < 1e-9
    s, e = shape_interval(0, 2, 12, 60);   assert s == 0.0 and abs(e - 8.4) < 1e-9
    s, e = shape_interval(58, 60, 12, 60); assert e == 60.0 and abs(s - 51.6) < 1e-9
    # already long enough -> untouched
    assert shape_interval(10, 30, 12, 60) == (10, 30)
    # spans most of the event -> at most 1.5x its own length (never lands at 2x GT)
    s, e = shape_interval(20, 35, 30, 60); assert abs((e - s) - 22.5) < 1e-9


def test_impulse_event_becomes_one_interval_with_good_iou():
    T = 60
    x = np.zeros(T); x[20:24] = 0.9; x[25:27] = 0.8       # noisy accident evidence 20-27
    iv = decode(curves(T, traffic_accident=x), level=2, duration_sec=T)
    assert len(iv) == 1 and iv[0].class_name == "traffic_accident"
    assert iou(iv[0].start, iv[0].end, 20, 30) >= 0.5      # GT accident 20-30


def test_gate_only_evidence_never_emits():
    T = 40
    s1 = {"fire": np.full(T, 1.0)}                          # gate screaming, VLM silent
    c = fuse(s1, [], T)
    assert decode(c, level=2, duration_sec=T) == []
    assert decode_l1(c) is None


def test_vlm_confirmation_emits_and_abstention_blocks_low_conf():
    T = 40
    s1 = {"fire": np.full(T, 0.6)}
    strong = fuse(s1, [Verdict(10, 22, "fire", 0.9, "Flames on a rooftop with a rising dark plume.")], T)
    weak = fuse(s1, [Verdict(10, 22, "fire", 0.5)], T)
    assert len(decode(strong, level=2, duration_sec=T)) == 1
    assert decode(weak, level=2, duration_sec=T) == []
    assert decode_l1(strong) == "fire"


def test_cross_class_nms_and_cooccurrence():
    T = 60
    fire = np.zeros(T); fire[10:40] = 0.9
    smoke = np.zeros(T); smoke[10:40] = 0.8
    acc = np.zeros(T); acc[12:20] = 0.85
    iv = decode(curves(T, fire=fire, smoke=smoke, traffic_accident=acc), level=2, duration_sec=T)
    names = sorted(i.class_name for i in iv)
    assert "fire" in names and "smoke" in names          # whitelisted pair survives
    assert "traffic_accident" not in names               # suppressed by stronger overlapping fire


def test_l3_multiple_events_kept_separate_when_far_apart():
    T = 300
    x = np.zeros(T); x[20:40] = 0.9; x[200:230] = 0.9
    iv = decode(curves(T, traffic_accident=x), level=3, duration_sec=T)
    assert len(iv) == 2
