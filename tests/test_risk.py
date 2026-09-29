"""批量运输风险比较纯函数单元测试（不依赖 Web/DB）。

覆盖：批内最小-最大归一化（含全相同记 0）、加权总分与权重和归一化、
非支配前沿的支配定义（全 <= 且至少一项 <）、排名（总分升序、同分 ID 升序），
并用 O(n^2) 暴力判定交叉验证前沿结果。
"""

from __future__ import annotations

import random

import pytest

from app.risk import METRIC_KEYS, compare_risk, dominates, min_max_normalize, pareto_front


def _metrics(cycle=0, cold=0.0, heat=0.0, gap=0.0, alerts=0):
    return {
        "cycle_count": cycle,
        "cold_dose_c_min": cold,
        "heat_dose_c_min": heat,
        "exposure_gap_minutes": gap,
        "unacknowledged_alert_count": alerts,
    }


EQUAL_WEIGHTS = {key: 1.0 for key in METRIC_KEYS}


# ---------------- 最小-最大归一化 ----------------


def test_min_max_normalize_basic() -> None:
    assert min_max_normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_all_equal_is_zero() -> None:
    # 全相同（含全部为 0）时整列记 0，不做除零
    assert min_max_normalize([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert min_max_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_min_max_normalize_empty() -> None:
    assert min_max_normalize([]) == []


def test_min_max_normalize_two_values() -> None:
    assert min_max_normalize([7.0, 9.0]) == [0.0, 1.0]


# ---------------- 支配关系 ----------------


def test_dominates_true_when_all_le_and_one_lt() -> None:
    a = _metrics(1, 2.0, 3.0, 4.0, 1)
    b = _metrics(1, 2.0, 5.0, 4.0, 2)
    assert dominates(a, b) is True


def test_dominates_false_when_equal_on_all() -> None:
    a = _metrics(1, 2.0, 3.0, 4.0, 1)
    b = _metrics(1, 2.0, 3.0, 4.0, 1)
    assert dominates(a, b) is False


def test_dominates_false_when_one_metric_higher() -> None:
    # a 四项更低但一项更高：互不支配
    a = _metrics(2, 1.0, 1.0, 1.0, 0)
    b = _metrics(1, 5.0, 5.0, 5.0, 5)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_pareto_front_single_sample() -> None:
    # 单条样本不被任何其他样本支配 -> 在前沿
    assert pareto_front([_metrics(1, 1.0, 1.0, 1.0, 1)]) == [True]


def test_pareto_front_strict_dominator_only() -> None:
    raw = [_metrics(0), _metrics(1, 1.0, 1.0, 1.0, 1)]
    assert pareto_front(raw) == [True, False]


def test_pareto_front_all_equal_all_on_frontier() -> None:
    raw = [_metrics(1, 1.0, 1.0, 1.0, 1) for _ in range(3)]
    assert pareto_front(raw) == [True, True, True]


def test_pareto_front_mixed() -> None:
    # a 支配 c；b 与 a 互不支配（b 循环更少但告警更多）；c 被 a、b 共同支配
    a = _metrics(1, 0.0, 0.0, 0.0, 2)
    b = _metrics(0, 5.0, 5.0, 5.0, 5)
    c = _metrics(2, 5.0, 5.0, 5.0, 5)
    assert pareto_front([a, b, c]) == [True, True, False]


# ---------------- 综合比较 ----------------


def test_compare_risk_normalization_and_ranking() -> None:
    samples = [
        ("low", _metrics(0)),
        ("high", _metrics(2, 10.0, 20.0, 30.0, 4)),
    ]
    result = compare_risk(samples, EQUAL_WEIGHTS)
    by_id = {item["sample_id"]: item for item in result}

    assert by_id["low"]["rank"] == 1
    assert by_id["high"]["rank"] == 2
    assert by_id["low"]["total_score"] == pytest.approx(0.0)
    assert by_id["high"]["total_score"] == pytest.approx(1.0)
    # 批内两项：最小项归一化为 0，最大项归一化为 1
    assert by_id["low"]["normalized"] == {key: 0.0 for key in METRIC_KEYS}
    assert by_id["high"]["normalized"] == {key: 1.0 for key in METRIC_KEYS}
    # 前沿：全零样本支配高风险样本
    assert by_id["low"]["on_frontier"] is True
    assert by_id["high"]["on_frontier"] is False


def test_compare_risk_constant_metric_contributes_zero() -> None:
    # 循环数全批相同（都为 1）：该项归一化全 0，即使权重为正也不参与总分
    samples = [
        ("a", _metrics(1, 0.0, 0.0, 0.0, 0)),
        ("b", _metrics(1, 10.0, 0.0, 0.0, 0)),
    ]
    weights = dict(EQUAL_WEIGHTS)
    result = compare_risk(samples, weights)
    by_id = {item["sample_id"]: item for item in result}
    assert by_id["a"]["normalized"]["cycle_count"] == 0.0
    assert by_id["b"]["normalized"]["cycle_count"] == 0.0
    # 只有 cold_dose 一项有差异：b 总分 1/5
    assert by_id["a"]["total_score"] == pytest.approx(0.0)
    assert by_id["b"]["total_score"] == pytest.approx(0.2)
    # 五项原始指标中循环数相等、其余 a<=b 且有严格更低 -> a 支配 b
    assert by_id["a"]["on_frontier"] is True
    assert by_id["b"]["on_frontier"] is False


def test_compare_risk_weighted_sum_divided_by_weight_sum() -> None:
    samples = [
        ("a", _metrics(0)),
        ("b", _metrics(1, 0.0, 0.0, 0.0, 0)),
    ]
    weights = {key: 0.0 for key in METRIC_KEYS}
    weights["cycle_count"] = 4.0
    result = {item["sample_id"]: item for item in compare_risk(samples, weights)}
    # 零权重指标不影响总分；b 在唯一有权重的项上归一化为 1 -> 4/4 = 1
    assert result["b"]["total_score"] == pytest.approx(1.0)

    # 混合权重：仅循环数一项有差异（其余全相同 -> 归一化为 0），
    # b 总分 = w_cycle / Σw = 1 / (1+2+3+4+5) = 1/15，精确验证除以权重和
    mixed = {key: float(i + 1) for i, key in enumerate(METRIC_KEYS)}
    result2 = {item["sample_id"]: item for item in compare_risk(samples, mixed)}
    assert result2["b"]["total_score"] == pytest.approx(1.0 / 15.0)

    # 五项全部有差异时，最大项每项归一化均为 1 -> 任意正权重下总分恒为 Σw/Σw = 1
    all_differ = [
        ("a", _metrics(0)),
        ("b", _metrics(1, 1.0, 1.0, 1.0, 1)),
    ]
    result3 = {item["sample_id"]: item for item in compare_risk(all_differ, mixed)}
    assert result3["b"]["total_score"] == pytest.approx(1.0)


def test_compare_risk_rank_order_score_then_id() -> None:
    # 三个样本在同一指标上取值 0/1/2 -> 总分 0/0.2/0.4
    samples = [
        ("z", _metrics(0)),
        ("a", _metrics(2)),
        ("m", _metrics(1)),
    ]
    result = compare_risk(samples, EQUAL_WEIGHTS)
    assert [item["sample_id"] for item in result] == ["z", "m", "a"]
    assert [item["rank"] for item in result] == [1, 2, 3]


def test_compare_risk_tie_breaks_by_sample_id() -> None:
    # 两条样本指标完全相同 -> 总分相同，按样本 ID 升序
    samples = [("bbb", _metrics(1, 1.0, 1.0, 1.0, 1)), ("aaa", _metrics(1, 1.0, 1.0, 1.0, 1))]
    result = compare_risk(samples, EQUAL_WEIGHTS)
    assert [item["sample_id"] for item in result] == ["aaa", "bbb"]
    assert [item["rank"] for item in result] == [1, 2]
    # 互不支配（全等）-> 都在前沿
    assert all(item["on_frontier"] for item in result)


def test_compare_risk_preserves_raw_metrics() -> None:
    samples = [("s", _metrics(3, 1.5, 2.5, 3.5, 7))]
    result = compare_risk(samples, EQUAL_WEIGHTS)[0]
    assert result["metrics"] == _metrics(3, 1.5, 2.5, 3.5, 7)
    assert result["rank"] == 1
    assert result["total_score"] == pytest.approx(0.0)
    assert result["on_frontier"] is True


def test_compare_risk_zero_weight_sum_rejected() -> None:
    with pytest.raises(ValueError):
        compare_risk([("a", _metrics()), ("b", _metrics(1))], {key: 0.0 for key in METRIC_KEYS})


def test_brute_force_frontier_agrees_random() -> None:
    """随机小批样本：前沿与排名结构用暴力 O(n^2) 判定交叉验证。"""

    rng = random.Random(20260929)
    for _ in range(500):
        n = rng.randint(1, 6)
        samples = []
        for i in range(n):
            samples.append(
                (
                    f"s{i:02d}",
                    _metrics(
                        cycle=rng.randint(0, 3),
                        cold=float(rng.randint(0, 5)),
                        heat=float(rng.randint(0, 5)),
                        gap=float(rng.randint(0, 5)),
                        alerts=rng.randint(0, 3),
                    ),
                )
            )
        result = compare_risk(samples, EQUAL_WEIGHTS)
        raw_by_id = {sid: m for sid, m in samples}

        # 暴力前沿：不存在另一条样本 j 支配当前样本
        for item in result:
            i = item["metrics"]
            expected_front = not any(
                other_sid != item["sample_id"] and dominates(raw_by_id[other_sid], i)
                for other_sid, _ in samples
            )
            assert item["on_frontier"] is expected_front

        # 排名严格按 (总分, ID) 升序且名次连续
        keys = [(item["total_score"], item["sample_id"]) for item in result]
        assert keys == sorted(keys)
        assert [item["rank"] for item in result] == list(range(1, n + 1))
