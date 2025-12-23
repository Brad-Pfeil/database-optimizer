from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from database_optimiser.adaptive.contextual_bandit import (
    ContextualBanditPolicy,
)
from database_optimiser.config import Config
from database_optimiser.query.context import extract_query_context
from database_optimiser.storage.metadata import MetadataStore


def test_derived_partition_mapping_counts_as_partition_match(tmp_path):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    # Layout A is physically partitioned on date_year_month but logically matches date filters.
    store.create_layout(
        layout_id="la",
        table_name="t",
        partition_cols=["date_year_month"],
        sort_cols=None,
        layout_path=str(tmp_path / "la"),
        file_size_mb=1.0,
        notes=json.dumps(
            {"derived_partition_cols": {"date_year_month": "date"}}
        ),
    )
    # Layout B does not match date filters.
    store.create_layout(
        layout_id="lb",
        table_name="t",
        partition_cols=["other"],
        sort_cols=None,
        layout_path=str(tmp_path / "lb"),
        file_size_mb=1.0,
    )

    policy = ContextualBanditPolicy(metadata_store=store, exploration_rate=0.0)
    ctx = extract_query_context("SELECT * FROM t WHERE date = '2020-01-01'")
    chosen = policy.select_layout(table_name="t", context=ctx)
    assert chosen == "la"


def test_cluster_global_union_considers_global_even_when_cluster_layouts_exist(
    tmp_path,
):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    # Global layout with good reward
    store.create_layout(
        layout_id="g",
        table_name="t",
        partition_cols=None,
        sort_cols=["a"],
        layout_path=str(tmp_path / "g"),
        file_size_mb=1.0,
        cluster_id=None,
    )
    # Cluster-scoped layout with no reward history
    store.create_layout(
        layout_id="c",
        table_name="t",
        partition_cols=None,
        sort_cols=["b"],
        layout_path=str(tmp_path / "c"),
        file_size_mb=1.0,
        cluster_id="c0",
    )

    now = datetime.now(timezone.utc)
    store.record_evaluation(
        layout_id="g",
        table_name="t",
        cluster_id=None,
        baseline_layout_id="initial",
        eval_mode="natural_window",
        eval_status="scored",
        eval_window_start=now - timedelta(hours=2),
        eval_window_end=now - timedelta(hours=1),
        avg_latency_ms=0.0,
        queries_evaluated=10,
        rewrite_cost_sec=0.0,
        reward_score=0.5,
    )

    policy = ContextualBanditPolicy(metadata_store=store, exploration_rate=0.0)
    ctx = extract_query_context("SELECT * FROM t WHERE a = 1")
    chosen = policy.select_layout(table_name="t", context=ctx, cluster_id="c0")
    assert chosen == "g"


def test_reward_stats_cached_within_ttl(tmp_path):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    store.create_layout(
        layout_id="l1",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "l1"),
        file_size_mb=1.0,
    )

    calls = {"n": 0}
    orig = store.get_reward_stats_for_table

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    store.get_reward_stats_for_table = wrapped  # type: ignore[assignment]

    policy = ContextualBanditPolicy(
        metadata_store=store, exploration_rate=0.0, stats_cache_ttl_sec=999.0
    )
    ctx = extract_query_context("SELECT * FROM t WHERE a = 1")
    _ = policy.select_layout(table_name="t", context=ctx)
    _ = policy.select_layout(table_name="t", context=ctx)
    assert calls["n"] == 1
