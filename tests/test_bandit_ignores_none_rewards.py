from __future__ import annotations

from datetime import datetime, timedelta

from database_optimiser.adaptive.bandit import MultiArmedBandit
from database_optimiser.config import Config
from database_optimiser.storage.metadata import MetadataStore


def test_bandit_loads_only_non_null_rewards(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
    )
    config.ensure_dirs()

    store = MetadataStore(config)
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

    now = datetime.utcnow()
    store.record_evaluation(
        layout_id=layout_id,
        eval_window_start=now - timedelta(hours=2),
        eval_window_end=now - timedelta(hours=1),
        avg_latency_ms=0.0,
        queries_evaluated=0,
        rewrite_cost_sec=0.0,
        reward_score=None,
    )
    store.record_evaluation(
        layout_id=layout_id,
        eval_window_start=now - timedelta(hours=1),
        eval_window_end=now,
        avg_latency_ms=0.0,
        queries_evaluated=1,
        rewrite_cost_sec=0.0,
        reward_score=0.25,
    )

    bandit = MultiArmedBandit(metadata_store=store)
    bandit.load_evaluations(table)

    stats = bandit.get_arm_stats()
    assert stats[layout_id]["pulls"] == 1
    assert abs(stats[layout_id]["mean_reward"] - 0.25) < 1e-9
