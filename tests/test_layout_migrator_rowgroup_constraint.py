from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from database_optimiser.config import Config
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.storage.metadata import MetadataStore


def test_migrate_table_sets_rowgroup_leq_file_rows(tmp_path: Path):
    # Arrange: create a tiny source parquet dataset
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    migrator = LayoutMigrator(config, store)

    table_name = "nyc_taxi"
    source_path = config.data_dir / table_name / "layout_initial"
    source_path.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.table({"a": list(range(10)), "b": ["x"] * 10}),
        source_path / "part_00000.parquet",
    )

    # This is the failing shape: non-partitioned write + file_size_mb set.
    layout_spec = LayoutSpec(
        partition_cols=None, sort_cols=None, file_size_mb=1.0
    )

    # Act: should not raise
    out_path = migrator.migrate_table(
        table_name=table_name,
        source_path=source_path,
        layout_spec=layout_spec,
        layout_id="layout_test_rowgroup",
    )

    # Assert: output exists and contains parquet files
    assert out_path.exists()
    assert any(out_path.rglob("*.parquet"))
