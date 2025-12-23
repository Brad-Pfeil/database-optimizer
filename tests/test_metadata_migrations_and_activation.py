from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database_optimiser.config import Config
from database_optimiser.storage.metadata import MetadataStore


def test_schema_version_created_and_migrations_run(tmp_path: Path) -> None:
    db = tmp_path / "meta.db"
    # Create a minimal pre-migration DB without schema_version.
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE query_log (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            table_name TEXT
        )
        """
    )
    cur.execute(
        "INSERT INTO query_log(ts, table_name) VALUES(?, ?)",
        ("2025-01-01 00:00:00", "t"),
    )
    conn.commit()
    conn.close()

    config = Config(data_dir=tmp_path / "data", metadata_db_path=db)
    config.ensure_dirs()
    store = MetadataStore(config)

    # schema_version should exist and be at latest.
    with store._connection() as conn:  # noqa: SLF001
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        assert cur.fetchone() is not None
        cur.execute("SELECT version FROM schema_version")
        v = int(cur.fetchone()[0])
        assert v == MetadataStore.LATEST_SCHEMA_VERSION


def test_activate_layout_scoped_by_cluster(tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    store.create_layout(
        layout_id="g1",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "g1"),
        file_size_mb=1.0,
        cluster_id=None,
    )
    store.create_layout(
        layout_id="c1",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "c1"),
        file_size_mb=1.0,
        cluster_id="c0",
    )
    store.create_layout(
        layout_id="c2",
        table_name="t",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "c2"),
        file_size_mb=1.0,
        cluster_id="c0",
    )

    store.activate_layout("g1", "t")
    assert store.get_active_layout("t", cluster_id=None)["layout_id"] == "g1"  # type: ignore[index]

    store.activate_layout("c1", "t")
    assert store.get_active_layout("t", cluster_id="c0")["layout_id"] == "c1"  # type: ignore[index]
    # Global activation should remain intact
    assert store.get_active_layout("t", cluster_id=None)["layout_id"] == "g1"  # type: ignore[index]

    store.activate_layout("c2", "t")
    assert store.get_active_layout("t", cluster_id="c0")["layout_id"] == "c2"  # type: ignore[index]


def test_query_log_ts_filters_use_epoch_ms(tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    now = datetime.now(timezone.utc)
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
        context_key="k",
        cluster_id=None,
    )
    # backdate the ts by direct update for deterministic filtering
    with store._connection() as conn:  # noqa: SLF001
        cur = conn.cursor()
        cur.execute(
            "UPDATE query_log SET ts=? WHERE query_id=(SELECT MAX(query_id) FROM query_log)",
            (
                int(
                    (now - timedelta(hours=2))
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                ),
            ),
        )
        conn.commit()

    rows = store.get_query_logs(
        table_name="t",
        start_time=now - timedelta(hours=3),
        end_time=now - timedelta(hours=1),
    )
    assert rows
