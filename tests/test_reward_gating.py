from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
                int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000),
                None,
                table_name,
                layout_id,
                "[]",
                "[]",
                "[]",
                "[]",
                "[]",
                runtime_ms,
                None,
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
            (
                int(
                    created_at.replace(tzinfo=timezone.utc).timestamp() * 1000
                ),
                layout_id,
            ),
        )
        conn.commit()


def test_reward_is_none_when_insufficient_new_queries(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        eval_window_hours=24,
        min_queries_for_eval=5,
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
        file_size_mb=128.0,
    )

    now = datetime.now(timezone.utc)
    created_at = now - timedelta(hours=1)
    _set_layout_created_at(store, layout_id, created_at)

    # Baseline: 5 queries before layout creation
    for i in range(5):
        _insert_query_log(
            store,
            ts=now - timedelta(hours=2, minutes=i),
            table_name=table,
            layout_id=None,
            runtime_ms=100.0,
        )

    # New layout: only 2 queries after layout creation => insufficient
    for i in range(2):
        _insert_query_log(
            store,
            ts=now - timedelta(minutes=30, seconds=i),
            table_name=table,
            layout_id=layout_id,
            runtime_ms=80.0,
        )

    out = rewards.evaluate_layout(
        layout_id=layout_id, eval_window_hours=24, rewrite_cost_sec=0.0
    )
    assert out["eval_status"] == "insufficient_data"
    assert out["reward_score"] is None
    assert out["reward_breakdown"] is None
    assert out["metrics"]["queries_evaluated"] == 2


def test_reward_is_scored_when_enough_queries(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        eval_window_hours=24,
        min_queries_for_eval=1,
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
        file_size_mb=128.0,
    )

    now = datetime.now(timezone.utc)
    created_at = now - timedelta(hours=1)
    _set_layout_created_at(store, layout_id, created_at)

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
    assert out["eval_status"] == "scored"
    assert out["reward_score"] is not None
    assert out["reward_breakdown"] is not None
    assert out["metrics"]["queries_evaluated"] == 1
