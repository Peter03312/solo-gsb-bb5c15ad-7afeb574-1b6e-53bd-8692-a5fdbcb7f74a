"""样本运输风险批内比较（纯函数，不依赖数据库与 Web 框架）。

给定一批样本的五项“越低越安全”的风险指标（冻融循环数、低温/高温暴露剂量、
暴露缺口分钟数、未确认告警数）与五项非负权重（至少一项为正，由接口层校验）：

1. **批内最小最大归一化**：逐指标 ``(x - min) / (max - min)``；
   某指标批内数值全相同（``max == min``）时，该指标归一化值一律记 0；
2. **总分** = 归一化值的加权和 / 权重和（权重和保证为正，零权重指标不影响总分）；
3. **非支配前沿**：样本 A 支配样本 B 当且仅当 A 的五项指标均不高于 B
   且至少一项严格更低；不被批内任何其他样本支配的样本位于前沿上
   （指标完全相同的样本互不支配，可同时在前沿上）；
4. **排名**按总分升序、总分相同按样本 ID 升序，名次从 1 起连续编号。

归一化值与总分输出时保留 6 位小数（与状态接口预算使用率的舍入口径一致）；
总分由舍入后的归一化值计算，排名依据舍入后的总分排序，
保证响应自洽（调用方可按返回数值复算核验）且结果可复现。
"""

from __future__ import annotations

from dataclasses import dataclass

# 五项风险指标的规范字段名（与状态接口字段一致）
METRIC_FIELDS = (
    "cycle_count",
    "cold_dose_c_min",
    "heat_dose_c_min",
    "exposure_gap_minutes",
    "unacknowledged_alert_count",
)

# 输出舍入精度（与 services 中预算使用率 round(..., 6) 的口径一致）
ROUND_DIGITS = 6


@dataclass(frozen=True)
class RiskMetrics:
    """一条样本的五项原始风险指标（均为“越低越安全”的非负数值）。"""

    cycle_count: float
    cold_dose_c_min: float
    heat_dose_c_min: float
    exposure_gap_minutes: float
    unacknowledged_alert_count: float

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.cycle_count,
            self.cold_dose_c_min,
            self.heat_dose_c_min,
            self.exposure_gap_minutes,
            self.unacknowledged_alert_count,
        )


@dataclass(frozen=True)
class RiskWeights:
    """五项指标的非负权重（至少一项为正，由接口层校验）。"""

    cycle_count: float
    cold_dose_c_min: float
    heat_dose_c_min: float
    exposure_gap_minutes: float
    unacknowledged_alert_count: float

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.cycle_count,
            self.cold_dose_c_min,
            self.heat_dose_c_min,
            self.exposure_gap_minutes,
            self.unacknowledged_alert_count,
        )

    def total(self) -> float:
        return sum(self.as_tuple())


@dataclass(frozen=True)
class AssessmentItem:
    """一条样本的评估结果：归一化值、总分、前沿标记与名次。"""

    sample_id: str
    normalized: dict[str, float]
    total_score: float
    on_frontier: bool
    rank: int


def normalize_batch(metrics: list[RiskMetrics]) -> list[dict[str, float]]:
    """批内逐指标最小最大归一化；数值全相同的指标一律记 0。"""

    columns = list(zip(*(m.as_tuple() for m in metrics)))
    spans = [(min(col), max(col)) for col in columns]
    result: list[dict[str, float]] = []
    for m in metrics:
        normalized: dict[str, float] = {}
        for name, value, (lo, hi) in zip(METRIC_FIELDS, m.as_tuple(), spans):
            if hi == lo:
                normalized[name] = 0.0
            else:
                normalized[name] = round((value - lo) / (hi - lo), ROUND_DIGITS)
        result.append(normalized)
    return result


def total_score(normalized: dict[str, float], weights: RiskWeights) -> float:
    """归一化值的加权和除以权重和（权重和为正）。"""

    numerator = sum(
        w * normalized[name]
        for name, w in zip(METRIC_FIELDS, weights.as_tuple())
    )
    return round(numerator / weights.total(), ROUND_DIGITS)


def non_dominated_flags(metrics: list[RiskMetrics]) -> list[bool]:
    """逐样本判断是否处于非支配前沿（不被批内任何其他样本支配）。"""

    vectors = [m.as_tuple() for m in metrics]
    flags: list[bool] = []
    for i, a in enumerate(vectors):
        dominated = False
        for j, b in enumerate(vectors):
            if i == j:
                continue
            # b 支配 a：b 的所有指标不高于 a，且至少一项严格更低
            if all(x <= y for x, y in zip(b, a)) and any(
                x < y for x, y in zip(b, a)
            ):
                dominated = True
                break
        flags.append(not dominated)
    return flags


def assess(
    sample_ids: list[str],
    metrics: list[RiskMetrics],
    weights: RiskWeights,
) -> list[AssessmentItem]:
    """完整评估：归一化 → 总分 → 前沿标记 → 按（总分升序、样本 ID 升序）排名。

    返回按名次排列的结果列表（第 1 名在前）。
    """

    normalized = normalize_batch(metrics)
    scores = [total_score(norm, weights) for norm in normalized]
    frontier = non_dominated_flags(metrics)
    order = sorted(
        range(len(sample_ids)), key=lambda i: (scores[i], sample_ids[i])
    )
    items: list[AssessmentItem] = []
    for rank, i in enumerate(order, start=1):
        items.append(
            AssessmentItem(
                sample_id=sample_ids[i],
                normalized=normalized[i],
                total_score=scores[i],
                on_frontier=frontier[i],
                rank=rank,
            )
        )
    return items
