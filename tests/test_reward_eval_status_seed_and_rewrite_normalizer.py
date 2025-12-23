from __future__ import annotations

from datetime import datetime, timedelta, timezone

from database_optimiser.config import Config
from database_optimiser.evaluator.metrics import MetricsCalculator
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.storage.metadata import MetadataStore


def _insert_query_log(
    store: MetadataStore,
    *,
    ts: datetime,
    table_name: str,
    layout_id: str | None,
    runtime_ms: float,
) -> None:
    store.log_query(
        table_name=table_name,
        columns_used=[],
        predicates=[],
        joins=[],
        group_by_cols=[],
        order_by_cols=[],
        runtime_ms=runtime_ms,
        rows_scanned=None,
        rows_returned=0,
        query_text="select 1",
        layout_id=layout_id,
        context_key="k",
        cluster_id=None,
    )
    # Manually set timestamp for deterministic windows
    with store._connection() as conn:  # type: ignore[attr-defined]
        cur = conn.cursor()
        ts_ms = int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)
        cur.execute(
            "UPDATE query_log SET ts=? WHERE query_id=(SELECT MAX(query_id) FROM query_log)",
            (ts_ms,),
        )
        conn.commit()


def test_eval_status_persisted_in_layout_eval(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        min_queries_for_eval=10,
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    metrics = MetricsCalculator(store)
    rewards = RewardCalculator(metrics, config, store)

    table = "t"
    store.create_layout(
        layout_id="l1",
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "dummy"),
        file_size_mb=1.0,
    )

    now = datetime.now(timezone.utc)
    # Not enough samples -> insufficient_data
    _insert_query_log(
        store,
        ts=now - timedelta(minutes=10),
        table_name=table,
        layout_id="l1",
        runtime_ms=10.0,
    )
    _insert_query_log(
        store,
        ts=now - timedelta(minutes=10),
        table_name=table,
        layout_id=None,
        runtime_ms=10.0,
    )
    out = rewards.evaluate_layout(
        "l1", eval_window_hours=24, rewrite_cost_sec=0.0
    )
    assert out["eval_status"] == "insufficient_data"

    rows = store.get_layout_evaluations("l1", limit=1)
    assert rows and rows[0].get("eval_status") == "insufficient_data"


def test_bootstrap_seed_deterministic_per_window(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    metrics = MetricsCalculator(store)
    rewards = RewardCalculator(metrics, config, store)

    start = datetime(2025, 1, 1, 0, 0, 0)
    end1 = datetime(2025, 1, 2, 0, 0, 0)
    end2 = datetime(2025, 1, 2, 0, 0, 1)

    s1 = rewards._bootstrap_seed(  # type: ignore[attr-defined]
        layout_id="l1",
        baseline_layout_id="initial",
        start_time=start,
        end_time=end1,
    )
    s1b = rewards._bootstrap_seed(  # type: ignore[attr-defined]
        layout_id="l1",
        baseline_layout_id="initial",
        start_time=start,
        end_time=end1,
    )
    s2 = rewards._bootstrap_seed(  # type: ignore[attr-defined]
        layout_id="l1",
        baseline_layout_id="initial",
        start_time=start,
        end_time=end2,
    )
    assert s1 == s1b
    assert s1 != s2


def test_rewrite_cost_normalizer_historical_median(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
    )
    config.rewrite_cost_normalizer_mode = "historical_median"
    config.rewrite_cost_normalizer_sec_default = 3600.0
    config.rewrite_cost_normalizer_history_n = 10
    config.ensure_dirs()

    store = MetadataStore(config)
    metrics = MetricsCalculator(store)
    rewards = RewardCalculator(metrics, config, store)

    table = "t"
    store.create_layout(
        layout_id="l1",
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "dummy"),
        file_size_mb=1.0,
    )
    now = datetime.now(timezone.utc)
    # Insert some historical rewrite costs (median=150)
    store.record_evaluation(
        layout_id="l1",
        table_name=table,
        cluster_id=None,
        baseline_layout_id="initial",
        eval_mode="natural_window",
        eval_status="scored",
        eval_window_start=now - timedelta(hours=3),
        eval_window_end=now - timedelta(hours=2),
        avg_latency_ms=0.0,
        queries_evaluated=1,
        rewrite_cost_sec=100.0,
        reward_score=0.0,
    )
    store.record_evaluation(
        layout_id="l1",
        table_name=table,
        cluster_id=None,
        baseline_layout_id="initial",
        eval_mode="natural_window",
        eval_status="scored",
        eval_window_start=now - timedelta(hours=2),
        eval_window_end=now - timedelta(hours=1),
        avg_latency_ms=0.0,
        queries_evaluated=1,
        rewrite_cost_sec=200.0,
        reward_score=0.0,
    )

    baseline = {"avg_latency_ms": 10.0, "queries_evaluated": 10}
    new = {"avg_latency_ms": 9.0, "queries_evaluated": 10}
    breakdown = rewards.calculate_reward_breakdown(
        baseline_metrics=baseline,
        new_metrics=new,
        rewrite_cost_sec=75.0,
        table_name=table,
        cluster_id=None,
    )
    # 75/150 = 0.5
    assert abs(breakdown["normalized_rewrite_cost"] - 0.5) < 1e-6
