"""运输风险批量比较 API 集成测试（临时 SQLite，不污染现有数据）。

覆盖：五项指标取自当前重算状态、批内最小-最大归一化（全相同记 0）、
加权总分除以权重和、非支配前沿、排名顺序、样本重复/数量越界/权重非法/
样本不存在的 422 报错，以及接口纯只读（不改变状态与告警确认）。
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_risk.db"))

    import app.database as database
    import app.services as services
    import app.main as main

    importlib.reload(database)
    importlib.reload(services)
    importlib.reload(main)

    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


def _create_sample(client, sid, **kw):
    payload = {
        "id": sid,
        "name": f"样本 {sid}",
        "freeze_threshold_c": -20,
        "thaw_threshold_c": 0,
        "min_allowed_c": -80,
        "max_allowed_c": 8,
        "max_freeze_thaw_cycles": 5,
        "cold_dose_budget_c_min": 1000,
        "heat_dose_budget_c_min": 1000,
        "max_integration_interval_minutes": 120,
    }
    payload.update(kw)
    r = client.post("/samples", json=payload)
    assert r.status_code == 201, r.text


def _submit(client, sid, readings):
    r = client.post(f"/samples/{sid}/readings", json={"readings": readings})
    assert r.status_code == 200, r.text
    return r


EQUAL_WEIGHTS = {
    "cycle_count": 1,
    "cold_dose_c_min": 1,
    "heat_dose_c_min": 1,
    "exposure_gap_minutes": 1,
    "unacknowledged_alert_count": 1,
}


def _build_three_samples(client):
    """构造低/中/高三档样本：

    low：无读数，五项全 0；
    mid：一次冻融（循环 1，在允许区间内、无剂量/缺口/告警）；
    high：两次冻融 + 低温/高温越限 + 超 120 分钟缺口 -> 五项全部更高。
    """

    _create_sample(client, "risk-low")
    _create_sample(client, "risk-mid")
    _create_sample(client, "risk-high")

    _submit(
        client,
        "risk-mid",
        [
            {"event_id": "m1", "event_time": "2026-09-20T08:00:00Z", "temperature_c": 4},
            {"event_id": "m2", "event_time": "2026-09-20T08:30:00Z", "temperature_c": -22},
            {"event_id": "m3", "event_time": "2026-09-20T10:00:00Z", "temperature_c": 2},
        ],
    )

    _submit(
        client,
        "risk-high",
        [
            {"event_id": "h1", "event_time": "2026-09-20T08:00:00Z", "temperature_c": 4},
            {"event_id": "h2", "event_time": "2026-09-20T08:30:00Z", "temperature_c": -22},
            {"event_id": "h3", "event_time": "2026-09-20T09:00:00Z", "temperature_c": 2},
            {"event_id": "h4", "event_time": "2026-09-20T09:30:00Z", "temperature_c": -82},
            {"event_id": "h5", "event_time": "2026-09-20T10:00:00Z", "temperature_c": 4},
            # 与上一条间隔 200 分钟 > 120：区间整体跳过并计入缺口
            {"event_id": "h6", "event_time": "2026-09-20T13:20:00Z", "temperature_c": 10},
            {"event_id": "h7", "event_time": "2026-09-20T14:20:00Z", "temperature_c": 4},
        ],
    )


def test_compare_uses_current_state_metrics(client) -> None:
    _build_three_samples(client)
    r = client.post(
        "/transport-risk",
        json={"sample_ids": ["risk-low", "risk-mid", "risk-high"], "weights": EQUAL_WEIGHTS},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 3
    assert body["weight_sum"] == 5

    by_id = {item["sample_id"]: item for item in body["items"]}

    low, mid, high = by_id["risk-low"], by_id["risk-mid"], by_id["risk-high"]
    # 原始指标取自当前状态接口
    for sid, item in (("risk-low", low), ("risk-mid", mid), ("risk-high", high)):
        state = client.get(f"/samples/{sid}/state").json()
        m = item["metrics"]
        assert m["cycle_count"] == state["cycle_count"]
        assert m["cold_dose_c_min"] == state["cold_dose_c_min"]
        assert m["heat_dose_c_min"] == state["heat_dose_c_min"]
        assert m["exposure_gap_minutes"] == state["exposure_gap_minutes"]
        assert m["unacknowledged_alert_count"] == state["unacknowledged_alert_count"]

    assert low["metrics"]["cycle_count"] == 0
    assert mid["metrics"]["cycle_count"] == 1
    # -22→2（第 1 轮）、2→-82→4（第 2 轮），共 2 次完整循环
    assert high["metrics"]["cycle_count"] == 2
    assert high["metrics"]["cold_dose_c_min"] > 0
    assert high["metrics"]["heat_dose_c_min"] > 0
    assert high["metrics"]["exposure_gap_minutes"] == pytest.approx(200.0)
    assert high["metrics"]["unacknowledged_alert_count"] >= 2


def test_normalization_scores_frontier_and_ranking(client) -> None:
    _build_three_samples(client)
    body = client.post(
        "/transport-risk",
        json={"sample_ids": ["risk-low", "risk-mid", "risk-high"], "weights": EQUAL_WEIGHTS},
    ).json()

    by_id = {item["sample_id"]: item for item in body["items"]}
    low, mid, high = by_id["risk-low"], by_id["risk-mid"], by_id["risk-high"]

    # 排名：low < mid < high（等权重下 low 全 0、high 在所有有差异列取最大值）
    assert [item["sample_id"] for item in body["items"]] == ["risk-low", "risk-mid", "risk-high"]
    assert [item["rank"] for item in body["items"]] == [1, 2, 3]
    assert low["total_score"] == 0.0
    assert 0.0 < mid["total_score"] < high["total_score"] <= 1.0

    # 归一化界：每列最小 0、最大 1；全零样本每项归一化 0
    for key in (
        "cycle_count",
        "cold_dose_c_min",
        "heat_dose_c_min",
        "exposure_gap_minutes",
        "unacknowledged_alert_count",
    ):
        column = [item["normalized"][key] for item in body["items"]]
        assert min(column) == 0.0
        assert max(column) == 1.0
    assert all(v == 0.0 for v in low["normalized"].values())

    # 非支配前沿：全零的 low 支配其余两条；mid 与 high 都被 low 支配
    assert low["on_frontier"] is True
    assert mid["on_frontier"] is False
    assert high["on_frontier"] is False


def test_constant_metric_normalized_zero(client) -> None:
    # 两条样本五项原始指标完全相同（均为空样本）：
    # 归一化全 0、总分全 0、都在前沿，按 ID 排序
    _create_sample(client, "tie-b")
    _create_sample(client, "tie-a")
    body = client.post(
        "/transport-risk",
        json={"sample_ids": ["tie-b", "tie-a"], "weights": EQUAL_WEIGHTS},
    ).json()
    assert [item["sample_id"] for item in body["items"]] == ["tie-a", "tie-b"]
    for item in body["items"]:
        assert item["total_score"] == 0.0
        assert all(v == 0.0 for v in item["normalized"].values())
        assert item["on_frontier"] is True


def test_partial_weights_zero(client) -> None:
    _build_three_samples(client)
    weights = {
        "cycle_count": 0,
        "cold_dose_c_min": 0,
        "heat_dose_c_min": 0,
        "exposure_gap_minutes": 0,
        "unacknowledged_alert_count": 3,  # 只有告警权重为正
    }
    body = client.post(
        "/transport-risk",
        json={"sample_ids": ["risk-low", "risk-mid", "risk-high"], "weights": weights},
    ).json()
    assert body["weight_sum"] == 3
    by_id = {item["sample_id"]: item for item in body["items"]}
    # 总分只来自告警列：low/mid 告警 0 -> 0，high 有未确认告警 -> 1
    assert by_id["risk-low"]["total_score"] == 0.0
    assert by_id["risk-mid"]["total_score"] == 0.0
    assert by_id["risk-high"]["total_score"] == 1.0
    # low 与 mid 总分相同，按 ID 升序
    assert [item["sample_id"] for item in body["items"]][:2] == ["risk-low", "risk-mid"]


def test_duplicate_sample_ids_rejected(client) -> None:
    _create_sample(client, "risk-low")
    _create_sample(client, "risk-mid")
    r = client.post(
        "/transport-risk",
        json={"sample_ids": ["risk-low", "risk-low"], "weights": EQUAL_WEIGHTS},
    )
    assert r.status_code == 422


def test_sample_count_out_of_range_rejected(client) -> None:
    _create_sample(client, "only-one")
    # 少于 2 个
    r = client.post(
        "/transport-risk",
        json={"sample_ids": ["only-one"], "weights": EQUAL_WEIGHTS},
    )
    assert r.status_code == 422

    # 超过 50 个
    r2 = client.post(
        "/transport-risk",
        json={"sample_ids": [f"s{i}" for i in range(51)], "weights": EQUAL_WEIGHTS},
    )
    assert r2.status_code == 422


def test_missing_sample_rejected_422(client) -> None:
    _create_sample(client, "exists-1")
    _create_sample(client, "exists-2")
    r = client.post(
        "/transport-risk",
        json={"sample_ids": ["exists-1", "ghost"], "weights": EQUAL_WEIGHTS},
    )
    assert r.status_code == 422
    assert "ghost" in r.json()["detail"]


def test_invalid_weights_rejected(client) -> None:
    _create_sample(client, "w1")
    _create_sample(client, "w2")

    def _post(weights):
        return client.post(
            "/transport-risk",
            json={"sample_ids": ["w1", "w2"], "weights": weights},
        )

    # 全零权重
    r = _post({k: 0 for k in EQUAL_WEIGHTS})
    assert r.status_code == 422
    # 负权重
    r = _post({**EQUAL_WEIGHTS, "cycle_count": -1})
    assert r.status_code == 422
    # 非有限权重：NaN/Infinity 不是 JSON 合法值，但 Python 可用 allow_nan
    # 手工编码发送，服务端 Pydantic 须拒绝（422）
    import json

    for bad in ("Infinity", "NaN"):
        raw_weights = {
            "cycle_count": 1,
            "cold_dose_c_min": 1,
            "heat_dose_c_min": float(bad),
            "exposure_gap_minutes": 1,
            "unacknowledged_alert_count": 1,
        }
        body = json.dumps({"sample_ids": ["w1", "w2"], "weights": raw_weights}, allow_nan=True)
        r = client.post(
            "/transport-risk",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422, bad
    # 缺失权重字段
    r = client.post(
        "/transport-risk",
        json={"sample_ids": ["w1", "w2"], "weights": {"cycle_count": 1}},
    )
    assert r.status_code == 422


def test_missing_weights_field_rejected(client) -> None:
    _create_sample(client, "z1")
    _create_sample(client, "z2")
    r = client.post("/transport-risk", json={"sample_ids": ["z1", "z2"]})
    assert r.status_code == 422


def test_read_only_does_not_change_state_or_alerts(client) -> None:
    _build_three_samples(client)

    before = {
        sid: client.get(f"/samples/{sid}/state").json()
        for sid in ("risk-low", "risk-mid", "risk-high")
    }
    alerts_before = {
        sid: client.get(f"/alerts?sample_id={sid}").json()
        for sid in ("risk-low", "risk-mid", "risk-high")
    }

    r = client.post(
        "/transport-risk",
        json={
            "sample_ids": ["risk-high", "risk-low", "risk-mid"],
            "weights": EQUAL_WEIGHTS,
        },
    )
    assert r.status_code == 200

    for sid in ("risk-low", "risk-mid", "risk-high"):
        after = client.get(f"/samples/{sid}/state").json()
        assert after == before[sid]
        alerts_after = client.get(f"/alerts?sample_id={sid}").json()
        assert alerts_after == alerts_before[sid]


def test_boundary_50_samples_accepted(client) -> None:
    for i in range(50):
        _create_sample(client, f"bulk-{i:02d}")
    r = client.post(
        "/transport-risk",
        json={"sample_ids": [f"bulk-{i:02d}" for i in range(50)], "weights": EQUAL_WEIGHTS},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 50
    assert len(body["items"]) == 50
    # 空样本指标全等：归一化与总分全 0，排名按 ID 升序
    assert [item["rank"] for item in body["items"]] == list(range(1, 51))
    assert [item["sample_id"] for item in body["items"]] == [f"bulk-{i:02d}" for i in range(50)]
    assert all(item["on_frontier"] for item in body["items"])
