from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from database_optimiser.config import Config
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.storage.metadata import MetadataStore


def _write_parquet_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({"a": [1, 2, 3], "ts": [None, None, None]})
    pq.write_table(tbl, path / "part-0.parquet")


def test_migrator_writes_layout_id_with_layout_prefix_and_no_tmp_leftover(
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

    layout_id = "abc123"
    spec = LayoutSpec(partition_cols=None, sort_cols=None, file_size_mb=None)
    out = migrator.migrate_table(
        table_name="t",
        source_path=source,
        layout_spec=spec,
        layout_id=layout_id,
    )

    assert out == tmp_path / "data" / "t" / "layout_abc123"
    assert out.exists()
    assert not (tmp_path / "data" / "t" / "layout_abc123__tmp").exists()
