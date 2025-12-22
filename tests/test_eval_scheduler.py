from __future__ import annotations

from database_optimiser.config import Config
from database_optimiser.evaluator.metrics import MetricsCalculator
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.evaluator.scheduler import EvaluationScheduler
from database_optimiser.storage.metadata import MetadataStore


def test_scheduler_records_eval_when_enough_new_queries(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        min_queries_for_eval=5,
        eval_scheduler_min_new_queries=1,
        eval_window_hours=24,
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    # Create baseline and candidate layouts
    store.create_layout(
        layout_id="baseline",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "b"),
        file_size_mb=1.0,
    )
    store.activate_layout("baseline", "t")
    store.create_layout(
        layout_id="cand",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "c"),
        file_size_mb=1.0,
    )

    # Insert query logs for baseline and candidate in window
    for i in range(6):
        store.log_query(
            table_name="t",
            columns_used=[],
            predicates=[],
            joins=[],
            group_by_cols=[],
            order_by_cols=[],
            runtime_ms=100.0,
            rows_scanned=1000,
            rows_returned=1,
            query_text=f"SELECT {i}",
            layout_id="baseline",
        )
    for i in range(6):
        store.log_query(
            table_name="t",
            columns_used=[],
            predicates=[],
            joins=[],
            group_by_cols=[],
            order_by_cols=[],
            runtime_ms=50.0,
            rows_scanned=800,
            rows_returned=1,
            query_text=f"SELECT {i}",
            layout_id="cand",
        )

    reward = RewardCalculator(MetricsCalculator(store), config, store)
    sched = EvaluationScheduler(config, store, reward)

    n = sched.evaluate_table(table_name="t")
    assert n >= 1
    latest = store.get_latest_evaluation("cand")
    assert latest is not None
