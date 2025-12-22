from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from click.testing import CliRunner

from database_optimiser.cli.main import cli
from database_optimiser.config import Config
from database_optimiser.storage.metadata import MetadataStore


def _mk_store(tmp_path: Path) -> tuple[Config, MetadataStore]:
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    return config, MetadataStore(config)


def test_metadata_eval_history_filters_and_order(tmp_path: Path) -> None:
    _config, store = _mk_store(tmp_path)

    store.create_layout(
        layout_id="l1",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "l1"),
        file_size_mb=1.0,
        cluster_id="c0",
    )
    store.create_layout(
        layout_id="l2",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "l2"),
        file_size_mb=1.0,
        cluster_id=None,
    )

    now = datetime.utcnow()
    store.record_evaluation(
        layout_id="l1",
        eval_window_start=now - timedelta(hours=2),
        eval_window_end=now - timedelta(hours=1),
        avg_latency_ms=10.0,
        queries_evaluated=10,
        rewrite_cost_sec=0.0,
        reward_score=None,
    )
    store.record_evaluation(
        layout_id="l1",
        eval_window_start=now - timedelta(hours=1),
        eval_window_end=now,
        avg_latency_ms=9.0,
        queries_evaluated=20,
        rewrite_cost_sec=0.0,
        reward_score=0.1,
    )
    store.record_evaluation(
        layout_id="l2",
        eval_window_start=now - timedelta(hours=1),
        eval_window_end=now,
        avg_latency_ms=11.0,
        queries_evaluated=20,
        rewrite_cost_sec=0.0,
        reward_score=0.0,
    )

    rows = store.get_layout_eval_history(table_name="t", cluster_id="c0")
    assert [r["layout_id"] for r in rows] == ["l1", "l1"]
    assert rows[0]["eval_window_end"] >= rows[1]["eval_window_end"]

    scored = store.get_layout_eval_history(
        table_name="t", cluster_id="c0", only_scored=True
    )
    assert len(scored) == 1
    assert scored[0]["reward_score"] == 0.1


def test_metadata_layout_query_counts(tmp_path: Path) -> None:
    _config, store = _mk_store(tmp_path)
    store.create_layout(
        layout_id="l1",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "l1"),
        file_size_mb=1.0,
        cluster_id="c0",
    )

    store.log_query(
        table_name="t",
        columns_used=[],
        predicates=[],
        joins=[],
        group_by_cols=[],
        order_by_cols=[],
        runtime_ms=1.0,
        rows_scanned=None,
        rows_returned=0,
        query_text="select 1",
        layout_id="l1",
        cluster_id="c0",
        context_key="k",
    )
    store.log_query(
        table_name="t",
        columns_used=[],
        predicates=[],
        joins=[],
        group_by_cols=[],
        order_by_cols=[],
        runtime_ms=1.0,
        rows_scanned=None,
        rows_returned=0,
        query_text="select 1",
        layout_id=None,
        cluster_id="c0",
        context_key="k",
    )

    rows = store.get_layout_query_counts(table_name="t", cluster_id="c0")
    counts = {
        (r["layout_id"], r["cluster_id"]): r["query_count"] for r in rows
    }
    assert counts[("l1", "c0")] == 1
    assert counts[(None, "c0")] == 1


def test_cli_history_json_output(tmp_path: Path) -> None:
    config, store = _mk_store(tmp_path)
    store.create_layout(
        layout_id="l1",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "l1"),
        file_size_mb=1.0,
    )
    now = datetime.utcnow()
    store.record_evaluation(
        layout_id="l1",
        eval_window_start=now - timedelta(hours=1),
        eval_window_end=now,
        avg_latency_ms=10.0,
        queries_evaluated=10,
        rewrite_cost_sec=0.0,
        reward_score=0.2,
    )

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--data-dir",
            str(config.data_dir),
            "--metadata-db",
            str(config.metadata_db_path),
            "history",
            "--table-name",
            "t",
            "--format",
            "json",
        ],
    )
    assert res.exit_code == 0, res.output
    rows = json.loads(res.output)
    assert rows and rows[0]["layout_id"] == "l1"


def test_cli_report_writes_artifacts(tmp_path: Path) -> None:
    config, store = _mk_store(tmp_path)
    store.create_layout(
        layout_id="l1",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "l1"),
        file_size_mb=1.0,
    )
    now = datetime.utcnow()
    store.record_evaluation(
        layout_id="l1",
        eval_window_start=now - timedelta(hours=1),
        eval_window_end=now,
        avg_latency_ms=10.0,
        queries_evaluated=10,
        rewrite_cost_sec=0.0,
        reward_score=0.2,
    )
    store.log_query(
        table_name="t",
        columns_used=[],
        predicates=[],
        joins=[],
        group_by_cols=[],
        order_by_cols=[],
        runtime_ms=1.0,
        rows_scanned=None,
        rows_returned=0,
        query_text="select 1",
        layout_id="l1",
        cluster_id=None,
        context_key="k",
    )

    out_dir = tmp_path / "reports" / "run1"
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "--data-dir",
            str(config.data_dir),
            "--metadata-db",
            str(config.metadata_db_path),
            "report",
            "--table-name",
            "t",
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "layout_eval.csv").exists()
    assert (out_dir / "layout_eval.json").exists()
    assert (out_dir / "query_routing_summary.csv").exists()
