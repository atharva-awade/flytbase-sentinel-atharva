from sentinel.schema import Event, GTEvent, Prediction, RuntimeMetadata
from sentinel.scorer import iou, score, score_video_temporal

RT = RuntimeMetadata(frames_processed=1, chunks_processed=1, end_to_end_internal_time_ms=1000)


def gt(vid, lvl, cls=None, s=None, e=None):
    return GTEvent(video_id=vid, level=lvl, is_anomaly=cls is not None, class_name=cls or "normal",
                   start_time_sec=s, end_time_sec=e)


def pred(vid, *events):
    return Prediction(video_id=vid, events=list(events), runtime_metadata=RT)


def ev(cls, s=None, e=None, expl=None):
    return Event(class_name=cls, start_time_sec=s, end_time_sec=e, explanation=expl)


def test_iou_basic():
    assert abs(iou(0, 10, 5, 15) - 5 / 15) < 1e-9
    assert iou(0, 10, 10, 20) == 0.0
    assert iou(0, 10, 0, 10) == 1.0


def test_normal_video_zero_on_any_prediction():
    g = [gt("V", 2)]
    assert score_video_temporal("V", 2, pred("V"), g).score == 1.0
    r = score_video_temporal("V", 2, pred("V", ev("fire", 1, 3)), g)
    assert r.score == 0.0 and r.false_alarm


def test_inside_interval_must_cover_half():
    g = [gt("V", 2, "fire", 0, 20)]
    assert score_video_temporal("V", 2, pred("V", ev("fire", 0, 9)), g).matched == 0   # 45% -> no
    assert score_video_temporal("V", 2, pred("V", ev("fire", 0, 11)), g).matched == 1  # 55% -> yes


def test_swallowing_interval_max_twice_as_long():
    g = [gt("V", 2, "fire", 10, 20)]
    assert score_video_temporal("V", 2, pred("V", ev("fire", 5, 25)), g).matched == 1   # 2x -> IoU 0.5 ok
    assert score_video_temporal("V", 2, pred("V", ev("fire", 4, 26)), g).matched == 0   # >2x -> no


def test_fragments_penalised():
    g = [gt("V", 2, "fire", 0, 30)]
    whole = score_video_temporal("V", 2, pred("V", ev("fire", 0, 30)), g)
    frags = score_video_temporal("V", 2, pred("V", ev("fire", 0, 16), ev("fire", 16, 30)), g)
    assert whole.score > frags.score and frags.fragments == 1 and frags.matched == 1


def test_wrong_class_no_match():
    g = [gt("V", 2, "fire", 0, 30)]
    r = score_video_temporal("V", 2, pred("V", ev("smoke", 0, 30)), g)
    assert r.matched == 0 and r.alert == 1.0


def test_l1_pooled():
    g = [gt("A", 1), gt("B", 1, "fire"), gt("C", 1, "smoke")]
    rep = score([pred("A"), pred("B", ev("fire")), pred("C", ev("fire"))], g)
    assert abs(rep.l1_anomaly_acc - 1.0) < 1e-9
    assert abs(rep.l1_class_acc - 0.5) < 1e-9
    assert abs(rep.level_points[1] - 25 * 0.75) < 1e-9


def test_report_totals_and_explanations():
    g = [gt("A", 1, "fire"), gt("B", 2, "fire", 0, 10), gt("C", 3)]
    rep = score([pred("A", ev("fire", expl="Flames visible on a rooftop seen from above.")),
                 pred("B", ev("fire", 0, 10)), pred("C")], g, video_durations={"A": 10, "B": 10, "C": 10})
    assert rep.level_points[1] == 25 and rep.level_points[3] == 40
    assert 0 < rep.level_points[2] <= 35
    assert rep.explanation_coverage == 0.5
    assert rep.rtf == 0.1
