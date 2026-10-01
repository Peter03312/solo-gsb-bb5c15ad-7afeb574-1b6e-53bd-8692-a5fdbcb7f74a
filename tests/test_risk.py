"""运输风险批内比较纯函数单元测试（不依赖 Web/DB）。"""

from __future__ import annotations

from app.risk import (
    RiskMetrics,
    RiskWeights,
    assess,
    non_dominated_flags,
    normalize_batch,
    total_score,
)


def _m(cycle, cold, heat, gap, unack) -> RiskMetrics:
    return RiskMetrics(
        cycle_count=cycle,
        cold_dose_c_min=cold,
        heat_dose_c_min=heat,
        exposure_gap_minutes=gap,
        unacknowledged_alert_count=unack,
    )


def _w(cycle=1.0, cold=1.0, heat=1.0, gap=1.0, unack=1.0) -> RiskWeights:
    return RiskWeights(
        cycle_count=cycle,
        cold_dose_c_min=cold,
        heat_dose_c_min=heat,
        exposure_gap_minutes=gap,
        unacknowledged_alert_count=unack,
    )


# ---------------- 最小最大归一化 ----------------


def test_normalize_min_max_basic() -> None:
    metrics = [_m(0, 0, 0, 0, 0), _m(2, 300, 240, 180, 2), _m(1, 150, 120, 90, 1)]
    normalized = normalize_batch(metrics)
    assert normalized[0] == {
        "cycle_count": 0.0,
        "cold_dose_c_min": 0.0,
        "heat_dose_c_min": 0.0,
        "exposure_gap_minutes": 0.0,
        "unacknowledged_alert_count": 0.0,
    }
    assert normalized[1] == {
        "cycle_count": 1.0,
        "cold_dose_c_min": 1.0,
        "heat_dose_c_min": 1.0,
        "exposure_gap_minutes": 1.0,
        "unacknowledged_alert_count": 1.0,
    }
    assert normalized[2] == {
        "cycle_count": 0.5,
        "cold_dose_c_min": 0.5,
        "heat_dose_c_min": 0.5,
        "exposure_gap_minutes": 0.5,
        "unacknowledged_alert_count": 0.5,
    }


def test_normalize_all_equal_metric_scores_zero() -> None:
    # 循环数与未确认告警数批内全相同 -> 这两项归一化值一律记 0
    metrics = [_m(2, 0, 0, 0, 1), _m(2, 100, 50, 30, 1)]
    normalized = normalize_batch(metrics)
    for norm in normalized:
        assert norm["cycle_count"] == 0.0
        assert norm["unacknowledged_alert_count"] == 0.0
    assert normalized[0]["cold_dose_c_min"] == 0.0
    assert normalized[1]["cold_dose_c_min"] == 1.0


def test_normalize_identical_samples_all_zero() -> None:
    # 全部样本所有指标完全相同 -> 所有归一化值为 0
    metrics = [_m(1, 10, 10, 10, 1), _m(1, 10, 10, 10, 1)]
    normalized = normalize_batch(metrics)
    assert all(all(v == 0.0 for v in norm.values()) for norm in normalized)


# ---------------- 总分 ----------------


def test_total_score_is_weighted_average() -> None:
    normalized = {
        "cycle_count": 1.0,
        "cold_dose_c_min": 0.5,
        "heat_dose_c_min": 0.0,
        "exposure_gap_minutes": 0.0,
        "unacknowledged_alert_count": 0.0,
    }
    # (1*1 + 1*0.5) / 2 = 0.75（其余权重为 0，不参与）
    assert total_score(normalized, _w(cycle=1.0, cold=1.0, heat=0.0, gap=0.0, unack=0.0)) == 0.75
    # 等权：1.5 / 5 = 0.3
    assert total_score(normalized, _w()) == 0.3
    # 权重同比缩放不改变总分
    assert total_score(normalized, _w(cycle=2.0, cold=2.0, heat=2.0, gap=2.0, unack=2.0)) == 0.3


def test_total_score_zero_weight_metric_ignored() -> None:
    a = {
        "cycle_count": 1.0,
        "cold_dose_c_min": 0.0,
        "heat_dose_c_min": 0.0,
        "exposure_gap_minutes": 0.0,
        "unacknowledged_alert_count": 0.0,
    }
    b = dict(a, heat_dose_c_min=1.0)
    weights = _w(cycle=1.0, cold=0.0, heat=0.0, gap=0.0, unack=0.0)
    # 高温剂量权重为 0：两项样本总分相同
    assert total_score(a, weights) == total_score(b, weights) == 1.0


# ---------------- 非支配前沿 ----------------


def test_frontier_strict_dominance() -> None:
    # a 全指标为 0，支配 b 与 c；b / c 互不相支配（b 高温剂量更高，c 其余更高）
    metrics = [
        _m(0, 0, 0, 0, 0),
        _m(1, 0, 240, 0, 1),
        _m(2, 300, 0, 180, 2),
    ]
    assert non_dominated_flags(metrics) == [True, False, False]


def test_frontier_identical_vectors_do_not_dominate_each_other() -> None:
    # 指标完全相同：互不支配（没有严格更低项），两条都在前沿上
    metrics = [_m(1, 10, 10, 10, 1), _m(1, 10, 10, 10, 1), _m(2, 20, 20, 20, 2)]
    assert non_dominated_flags(metrics) == [True, True, False]


def test_frontier_all_incomparable() -> None:
    # 三条样本各有一项指标最差：互不支配，全部在前沿上
    metrics = [
        _m(2, 0, 0, 0, 0),
        _m(0, 300, 0, 0, 0),
        _m(0, 0, 240, 180, 2),
    ]
    assert non_dominated_flags(metrics) == [True, True, True]


# ---------------- 排名与完整评估 ----------------


def test_assess_ranking_score_then_sample_id() -> None:
    # 等权下 b/c 总分相同（各一项归一化为 1）：名次按样本 ID 升序打破平局
    sample_ids = ["c", "a", "b"]
    metrics = [
        _m(0, 0, 240, 0, 0),  # c：高温剂量最差
        _m(0, 0, 0, 0, 0),    # a：全指标最优
        _m(0, 300, 0, 0, 0),  # b：低温剂量最差
    ]
    items = assess(sample_ids, metrics, _w())
    assert [item.sample_id for item in items] == ["a", "b", "c"]
    assert [item.rank for item in items] == [1, 2, 3]
    assert items[0].total_score == 0.0
    assert items[1].total_score == items[2].total_score == 0.2
    assert [item.on_frontier for item in items] == [True, False, False]


def test_assess_score_formula_and_rounding() -> None:
    # 手工核验：总分 = 归一化值加权和 / 权重和，保留 6 位小数
    sample_ids = ["s1", "s2", "s3"]
    metrics = [
        _m(0, 0, 0, 0, 0),
        _m(1, 0, 240, 0, 1),
        _m(2, 300, 0, 180, 2),
    ]
    items = assess(sample_ids, metrics, _w())
    by_id = {item.sample_id: item for item in items}
    # s2：(0.5 + 0 + 1 + 0 + 0.5) / 5 = 0.4
    assert by_id["s2"].total_score == 0.4
    # s3：(1 + 1 + 0 + 1 + 1) / 5 = 0.8
    assert by_id["s3"].total_score == 0.8
    assert by_id["s1"].total_score == 0.0
    # 归一化值与原始指标对应
    assert by_id["s2"].normalized["cycle_count"] == 0.5
    assert by_id["s3"].normalized["cold_dose_c_min"] == 1.0


def test_assess_two_sample_minimal_batch() -> None:
    items = assess(["x", "y"], [_m(0, 0, 0, 0, 0), _m(3, 9, 9, 9, 3)], _w())
    assert [item.sample_id for item in items] == ["x", "y"]
    assert items[0].total_score == 0.0
    assert items[1].total_score == 1.0
    assert [item.on_frontier for item in items] == [True, False]


def test_assess_weight_choice_changes_ranking() -> None:
    # 只看循环数时 y 更优；只看未确认告警数时 x 更优
    sample_ids = ["x", "y"]
    metrics = [_m(3, 0, 0, 0, 0), _m(0, 0, 0, 0, 2)]
    by_cycle = assess(sample_ids, metrics, _w(cycle=1.0, cold=0, heat=0, gap=0, unack=0))
    assert [item.sample_id for item in by_cycle] == ["y", "x"]
    by_unack = assess(sample_ids, metrics, _w(cycle=0, cold=0, heat=0, gap=0, unack=1.0))
    assert [item.sample_id for item in by_unack] == ["x", "y"]
    # 两种权重下两条样本互不相支配，均在前沿上
    assert all(item.on_frontier for item in by_cycle)
