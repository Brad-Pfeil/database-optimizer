from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from database_optimiser.config import Config
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.storage.metadata import MetadataStore


def _write_parquet_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"a": [3, 2, 1], "ts": [None, None, None]}),
        path / "part-0.parquet",
    )


def test_migrator_does_not_mutate_layout_spec_and_writes_structured_notes(
    tmp_path: Path,
) -> None:
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    migrator = LayoutMigrator(config=config, metadata_store=store)

    source = tmp_path / "source"
    _write_parquet_dir(source)

    spec = LayoutSpec(partition_cols=["ts"], sort_cols=["a"], file_size_mb=1.0)
    # Copy for immutability check
    requested_partition_cols = list(spec.partition_cols or [])
    requested_sort_cols = list(spec.sort_cols or [])

    layout_id = "immut"
    _ = migrator.migrate_table(
        table_name="t",
        source_path=source,
        layout_spec=spec,
        layout_id=layout_id,
    )

    # Spec should be unchanged
    assert list(spec.partition_cols or []) == requested_partition_cols
    assert list(spec.sort_cols or []) == requested_sort_cols

    layout = store.get_layout(layout_id)
    assert layout is not None
    notes = json.loads(layout.get("notes") or "{}")
    assert "derived_partition_cols" in notes
    assert notes["requested_partition_cols"] == requested_partition_cols
    assert "effective_partition_cols" in notes
    assert notes["requested_sort_cols"] == requested_sort_cols
    assert "effective_sort_cols" in notes
    assert "validation" in notes
    assert "schema_hash" in notes["validation"]


def test_estimate_rewrite_cost_counts_nested_parquet_files(
    tmp_path: Path,
) -> None:
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    migrator = LayoutMigrator(config=config, metadata_store=store)

    root = tmp_path / "hive"
    (root / "p=1").mkdir(parents=True, exist_ok=True)
    (root / "p=2").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"a": [1]}), root / "p=1" / "part-0.parquet")
    pq.write_table(pa.table({"a": [2]}), root / "p=2" / "part-0.parquet")

    cost = migrator.estimate_rewrite_cost("t", root)
    assert cost > 0.0
