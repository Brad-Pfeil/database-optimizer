from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from database_optimiser.config import Config
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.storage.metadata import MetadataStore


def test_sort_mode_none_preserves_input_order_for_single_file(
    tmp_path: Path,
) -> None:
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.sort_mode = "none"
    config.ensure_dirs()
    store = MetadataStore(config)
    migrator = LayoutMigrator(config=config, metadata_store=store)

    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"a": [3, 2, 1], "b": [0, 0, 0]}),
        source / "part-0.parquet",
    )

    spec = LayoutSpec(partition_cols=None, sort_cols=["a"], file_size_mb=None)
    out = migrator.migrate_table(
        table_name="t",
        source_path=source,
        layout_spec=spec,
        layout_id="nosort",
    )

    # Since we used a single input file and sort_mode=none, the output should preserve row order.
    out_ds = ds.dataset(out, format="parquet")
    rows = out_ds.to_table(columns=["a"]).column(0).to_pylist()
    assert rows == [3, 2, 1]
