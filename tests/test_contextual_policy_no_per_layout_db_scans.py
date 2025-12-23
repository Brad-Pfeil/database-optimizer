from __future__ import annotations

from datetime import datetime, timedelta, timezone

from database_optimiser.adaptive.contextual_bandit import (
    ContextualBanditPolicy,
)
from database_optimiser.config import Config
from database_optimiser.query.context import extract_query_context
from database_optimiser.storage.metadata import MetadataStore


def test_contextual_policy_does_not_call_get_layout_evaluations_per_layout(
    tmp_path,
):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    store.create_layout(
        layout_id="l1",
        table_name="t",
        partition_cols=None,
        sort_cols=["a"],
        layout_path=str(tmp_path / "l1"),
        file_size_mb=1.0,
    )
    store.create_layout(
        layout_id="l2",
        table_name="t",
        partition_cols=None,
        sort_cols=["b"],
        layout_path=str(tmp_path / "l2"),
        file_size_mb=1.0,
    )

    now = datetime.now(timezone.utc)
    store.record_evaluation(
        layout_id="l2",
        eval_window_start=now - timedelta(hours=2),
        eval_window_end=now - timedelta(hours=1),
        avg_latency_ms=0.0,
        queries_evaluated=10,
        rewrite_cost_sec=0.0,
        reward_score=0.5,
    )

    # If the policy still did per-layout DB scans, this would be called.
    def _boom(*_args, **_kwargs):
        raise AssertionError("policy should not call get_layout_evaluations()")

    store.get_layout_evaluations = _boom  # type: ignore[assignment]

    policy = ContextualBanditPolicy(metadata_store=store, exploration_rate=0.0)
    ctx = extract_query_context("SELECT * FROM t WHERE a = 1")
    chosen = policy.select_layout(table_name="t", context=ctx)
    assert chosen in {"l1", "l2"}
