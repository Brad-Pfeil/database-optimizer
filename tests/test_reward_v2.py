from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock

from database_optimiser.config import Config
from database_optimiser.evaluator.metrics import MetricsCalculator
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.storage.metadata import MetadataStore


def _insert_query_log(
    metadata_store: MetadataStore,
    *,
    ts: datetime,
    table_name: str,
    layout_id: str | None,
    runtime_ms: float,
    rows_scanned: int | None = None,
) -> None:
    with metadata_store._connection() as conn:  # noqa: SLF001 - test helper
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_log (
              ts, user_id, table_name, layout_id, columns_used, predicates, joins,
              group_by_cols, order_by_cols, runtime_ms, rows_scanned, rows_returned, query_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                None,
                table_name,
                layout_id,
                "[]",
                "[]",
                "[]",
                "[]",
                "[]",
                runtime_ms,
                rows_scanned,
                0,
                "SELECT 1",
            ),
        )
        conn.commit()


def _set_layout_created_at(
    metadata_store: MetadataStore, layout_id: str, created_at: datetime
) -> None:
    with metadata_store._connection() as conn:  # noqa: SLF001 - test helper
        cur = conn.cursor()
        cur.execute(
            "UPDATE table_layout SET created_at = ? WHERE layout_id = ?",
            (created_at, layout_id),
        )
        conn.commit()


def test_reward_breakdown_includes_p95_p99_rows_scanned():
    # Purely functional test: no DB access required for breakdown.
    cfg = Config()
    metrics = Mock(spec=MetricsCalculator)
    store = Mock(spec=MetadataStore)
    r = RewardCalculator(
        metrics_calculator=metrics, config=cfg, metadata_store=store
    )

    baseline = {
        "avg_latency_ms": 100.0,
        "p95_latency_ms": 200.0,
        "p99_latency_ms": 400.0,
        "avg_rows_scanned": 1000.0,
        "latency_cv": 0.5,
    }
    new = {
        "avg_latency_ms": 50.0,
        "p95_latency_ms": 100.0,
        "p99_latency_ms": 200.0,
        "avg_rows_scanned": 500.0,
        "latency_cv": 0.25,
    }

    bd = r.calculate_reward_breakdown(
        baseline_metrics=baseline, new_metrics=new, rewrite_cost_sec=0.0
    )
    assert bd["mean_improvement"] > 0
    assert bd["p95_improvement"] > 0
    assert bd["p99_improvement"] > 0
    assert bd["rows_scanned_improvement"] > 0


def test_reward_guardrail_min_samples_blocks_scoring(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        min_queries_for_eval=10,
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    metrics = MetricsCalculator(store)
    rewards = RewardCalculator(metrics, config, store)

    table = "nyc_taxi"
    layout_id = "layout_test"
    store.create_layout(
        layout_id=layout_id,
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "dummy"),
        file_size_mb=1.0,
    )
    now = datetime.utcnow()
    _set_layout_created_at(store, layout_id, now - timedelta(hours=1))

    # Not enough queries: only 1 baseline + 1 new
    _insert_query_log(
        store,
        ts=now - timedelta(hours=2),
        table_name=table,
        layout_id=None,
        runtime_ms=100.0,
    )
    _insert_query_log(
        store,
        ts=now - timedelta(minutes=30),
        table_name=table,
        layout_id=layout_id,
        runtime_ms=50.0,
    )

    out = rewards.evaluate_layout(
        layout_id=layout_id, eval_window_hours=24, rewrite_cost_sec=0.0
    )
    assert out["reward_score"] is None
    assert out["eval_status"] == "insufficient_data"


def test_reward_guardrail_confidence_blocks_noise_win(tmp_path):
    # Small difference with overlapping distributions should be guarded to 0.
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        min_queries_for_eval=10,
        reward_bootstrap_iters=200,
        reward_confidence_alpha=0.05,
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    metrics = MetricsCalculator(store)
    rewards = RewardCalculator(metrics, config, store)

    table = "nyc_taxi"
    layout_id = "layout_test"
    store.create_layout(
        layout_id=layout_id,
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "dummy"),
        file_size_mb=1.0,
    )
    now = datetime.utcnow()
    _set_layout_created_at(store, layout_id, now - timedelta(hours=1))

    # Baseline: around 100ms, New: around 99ms (tiny delta)
    for i in range(50):
        _insert_query_log(
            store,
            ts=now - timedelta(hours=2, seconds=i),
            table_name=table,
            layout_id=None,
            runtime_ms=100.0 + (i % 5),
        )
    for i in range(50):
        _insert_query_log(
            store,
            ts=now - timedelta(minutes=30, seconds=i),
            table_name=table,
            layout_id=layout_id,
            runtime_ms=99.5 + (i % 5),
        )

    out = rewards.evaluate_layout(
        layout_id=layout_id, eval_window_hours=24, rewrite_cost_sec=0.0
    )
    # Either scored negative/small or guarded to 0; key is it shouldn't confidently reward.
    assert out["reward_score"] in (None, 0.0) or out["reward_score"] <= 0.05
