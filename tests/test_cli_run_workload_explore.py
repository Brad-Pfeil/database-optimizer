from __future__ import annotations

import re

from click.testing import CliRunner

from database_optimiser.cli.main import cli
from database_optimiser.config import Config
from database_optimiser.storage.metadata import MetadataStore


class _StubExecutor:
    def __init__(
        self, config: Config, metadata_store: MetadataStore, query_logger
    ):
        self.config = config
        self.metadata_store = metadata_store
        self.query_logger = query_logger
        self.current_layouts: dict[str, str | None] = {}

    def register_table(self, table_name: str, parquet_path: str) -> None:
        self.current_layouts.setdefault(table_name, None)

    def register_layout(
        self, table_name: str, layout_path: str, layout_id: str | None = None
    ) -> None:
        self.current_layouts[table_name] = layout_id

    def run_query(self, sql: str):
        m = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
        table = m.group(1) if m else None
        layout_id = self.current_layouts.get(table) if table else None
        self.query_logger.log_query_execution(
            sql=sql,
            runtime_ms=1.0,
            rows_returned=0,
            rows_scanned=None,
            user_id=None,
            layout_id=layout_id,
        )
        return object()

    def close(self) -> None:
        return None


class _StubExplorer:
    def __init__(self, *args, **kwargs):
        pass

    def select_layout_for_query(self, table_name: str, sql: str | None = None):
        return "layout_test"


def test_run_workload_explore_routes_and_logs_layout_id(tmp_path, monkeypatch):
    # Patch CLI internals to avoid DuckDB + heavy wiring
    import database_optimiser.cli.main as cli_main

    monkeypatch.setattr(cli_main, "QueryExecutor", _StubExecutor)
    monkeypatch.setattr(cli_main, "LayoutExplorer", _StubExplorer)

    # Stub query generation
    def _gen_queries(self, table_name: str, num_queries: int):
        return [
            f"SELECT 1 FROM {table_name} WHERE region = 'x'" for _ in range(5)
        ]

    monkeypatch.setattr(
        cli_main.DatasetGenerator,
        "generate_workload_queries",
        _gen_queries,
        raising=False,
    )

    # Prepare paths and metadata
    data_dir = tmp_path / "data"
    (data_dir / "events" / "layout_initial").mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "meta.db"

    # Create a layout that routing will pick
    cfg = Config(data_dir=data_dir, metadata_db_path=db_path)
    cfg.ensure_dirs()
    store = MetadataStore(cfg)
    store.create_layout(
        layout_id="layout_test",
        table_name="events",
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "layout_test_path"),
        file_size_mb=1.0,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--data-dir",
            str(data_dir),
            "--metadata-db",
            str(db_path),
            "run-workload",
            "--table-name",
            "events",
            "--num-queries",
            "5",
            "--explore",
        ],
    )
    assert result.exit_code == 0, result.output

    rows = store.get_query_logs(table_name="events")
    assert any(r.get("layout_id") == "layout_test" for r in rows)
