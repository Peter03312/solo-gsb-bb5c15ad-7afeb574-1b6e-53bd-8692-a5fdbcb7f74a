"""批量样本当前运输风险比较（纯函数，不依赖数据库与 Web 框架）。

参与比较的五项指标均取自样本**当前重算状态**，且方向统一为“数值越大、
运输风险越高”：

    cycle_count                 完整冻融循环数
    cold_dose_c_min             低温累计暴露剂量（°C·min）
    heat_dose_c_min             高温累计暴露剂量（°C·min）
    exposure_gap_minutes        暴露缺口总时长（分钟）
    unacknowledged_alert_count  未确认告警数

计算步骤：

1. **批内最小-最大归一化**：对每项指标在本批样本内做
   ``n = (x - min) / (max - min)``；该批所有样本该项数值全相同
   （含全部为 0）时，归一化值统一记 0，避免无差异指标人为放大风险；
2. **加权总分**：``score = Σ w_i·n_i / Σ w_i``，权重为有限非负数、
   权重和为正（调用方保证至少一项为正），因此总分落在 ``[0, 1]``；
3. **非支配前沿**：样本 A 的五项原始指标全部不高于样本 B，且至少一项
   严格更低时，A **支配** B（``all a_i <= b_i and any a_i < b_i``）；
   不被批内任何其他样本支配的样本构成非支配前沿（Pareto front）；
4. **排名**：按总分升序，总分相同时按样本 ID 升序，名次从 1 起连续编号。
"""

from __future__ import annotations

from typing import Mapping, Sequence

# 五项指标键（顺序即归一化列顺序，也用于权重与输出字段对齐）
METRIC_KEYS: tuple[str, ...] = (
    "cycle_count",
    "cold_dose_c_min",
    "heat_dose_c_min",
    "exposure_gap_minutes",
    "unacknowledged_alert_count",
)

# 单项指标值：循环数 / 告警数为整数，剂量与缺口为浮点数
MetricMap = dict[str, float]


def min_max_normalize(values: Sequence[float]) -> list[float]:
    """批内最小-最大归一化；数值全相同（极差为 0）时整列记 0。"""

    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        # 全相同（指标非负，span 不可能 < 0；<=0 一并防御浮点异常）
        return [0.0] * len(values)
    return [(v - low) / span for v in values]


def dominates(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    """``a`` 是否支配 ``b``：五项指标全部 <= 且至少一项严格 <。"""

    any_lower = False
    for key in METRIC_KEYS:
        x, y = a[key], b[key]
        if x > y:
            return False
        if x < y:
            any_lower = True
    return any_lower


def pareto_front(raw: Sequence[MetricMap]) -> list[bool]:
    """返回每个样本是否位于非支配前沿（不被批内任何其他样本支配）。"""

    flags = [True] * len(raw)
    for i, candidate in enumerate(raw):
        for j, other in enumerate(raw):
            if i != j and dominates(other, candidate):
                flags[i] = False
                break
    return flags


def weighted_score(normalized: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """归一化值的加权和除以权重和；调用方须保证权重和严格为正。"""

    weight_sum = sum(weights[key] for key in METRIC_KEYS)
    if weight_sum <= 0:
        raise ValueError("权重和必须为正（至少一项权重 > 0）")
    weighted_sum = sum(weights[key] * normalized[key] for key in METRIC_KEYS)
    return weighted_sum / weight_sum


def compare_risk(
    samples: Sequence[tuple[str, MetricMap]],
    weights: Mapping[str, float],
) -> list[dict]:
    """对一批样本做归一化、加权总分、非支配前沿与排名。

    :param samples: ``(sample_id, 五项原始指标)`` 列表，指标字典须包含
        :data:`METRIC_KEYS` 全部键。
    :param weights: 五项指标对应的有限非负权重，权重和须严格为正。
    :return: 按**排名**（总分升序、样本 ID 升序）排列的结果列表，每项含
        原始指标、归一化值、总分、非支配前沿标记与名次；归一化与总分在
        本模块内保持精确浮点值，展示层取舍入。
    """

    count = len(samples)
    ids = [sample_id for sample_id, _ in samples]
    raw = [dict(metrics) for _, metrics in samples]

    # 逐指标在批内做最小-最大归一化
    normalized: list[MetricMap] = [
        dict.fromkeys(METRIC_KEYS, 0.0) for _ in range(count)
    ]
    for key in METRIC_KEYS:
        column = [item[key] for item in raw]
        for i, value in enumerate(min_max_normalize(column)):
            normalized[i][key] = value

    scores = [weighted_score(normalized[i], weights) for i in range(count)]
    frontier = pareto_front(raw)

    # 排名：总分升序，总分相同按样本 ID 升序
    order = sorted(range(count), key=lambda i: (scores[i], ids[i]))
    ranks = [0] * count
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank

    return [
        {
            "rank": ranks[index],
            "sample_id": ids[index],
            "metrics": raw[index],
            "normalized": normalized[index],
            "total_score": scores[index],
            "on_frontier": frontier[index],
        }
        for index in order
    ]
