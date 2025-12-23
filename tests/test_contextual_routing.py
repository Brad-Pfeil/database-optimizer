from __future__ import annotations

from datetime import datetime, timedelta, timezone

from database_optimiser.adaptive.contextual_bandit import (
    ContextualBanditPolicy,
)
from database_optimiser.config import Config
from database_optimiser.query.context import extract_query_context
from database_optimiser.storage.metadata import MetadataStore


def test_context_extractor_stable_features():
    sql = "SELECT SUM(fare_amount) FROM nyc_taxi WHERE PULocationID = 3 GROUP BY DOLocationID ORDER BY DOLocationID"
    ctx = extract_query_context(sql)
    assert ctx.table_name == "nyc_taxi"
    assert "PULocationID" in ctx.filter_cols
    assert "DOLocationID" in ctx.group_by_cols
    assert "DOLocationID" in ctx.order_by_cols
    assert ctx.has_agg is True


def test_contextual_policy_prefers_layout_matching_filter(tmp_path):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    # Two layouts: one sorted on PULocationID, one on fare_amount
    store.create_layout(
        layout_id="layout_a",
        table_name="nyc_taxi",
        partition_cols=None,
        sort_cols=["PULocationID"],
        layout_path=str(tmp_path / "a"),
        file_size_mb=1.0,
    )
    store.create_layout(
        layout_id="layout_b",
        table_name="nyc_taxi",
        partition_cols=None,
        sort_cols=["fare_amount"],
        layout_path=str(tmp_path / "b"),
        file_size_mb=1.0,
    )

    # Add a mild global reward advantage to layout_b, but context should override via bonus.
    now = datetime.now(timezone.utc)
    store.record_evaluation(
        layout_id="layout_b",
        eval_window_start=now - timedelta(hours=2),
        eval_window_end=now - timedelta(hours=1),
        avg_latency_ms=0.0,
        queries_evaluated=10,
        rewrite_cost_sec=0.0,
        reward_score=0.01,
    )

    policy = ContextualBanditPolicy(metadata_store=store, exploration_rate=0.0)
    sql = "SELECT * FROM nyc_taxi WHERE PULocationID = 3"
    ctx = extract_query_context(sql)

    chosen = policy.select_layout(table_name="nyc_taxi", context=ctx)
    assert chosen == "layout_a"


def test_low_confidence_parse_does_not_use_contextual_policy(tmp_path):
    # Force regex fallback by using a bad dialect in config so confidence is low for CTE/subquery.
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        exploration_rate=0.0,
    )
    config.sql_parser_dialect = "__nope__"
    config.parse_confidence_threshold = 0.7
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

    from database_optimiser.adaptive.explorer import LayoutExplorer

    class _Dummy:
        pass

    explorer = LayoutExplorer(
        config,
        store,
        _Dummy(),  # type: ignore[arg-type]
        _Dummy(),  # type: ignore[arg-type]
        _Dummy(),  # type: ignore[arg-type]
        _Dummy(),  # type: ignore[arg-type]
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("contextual policy should not be called")

    explorer.contextual_policy.select_layout = _boom  # type: ignore[assignment]

    sql = "WITH x AS (SELECT * FROM t) SELECT * FROM x"
    chosen = explorer.select_layout_for_query("t", sql=sql)
    assert chosen == "l1"
