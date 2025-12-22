from __future__ import annotations

from pathlib import Path

import pandas as pd

from database_optimiser.config import Config
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.storage.metadata import MetadataStore


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_incremental_partition_safety_drops_high_cardinality_key(tmp_path):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    migrator = LayoutMigrator(config, store)

    # customer_id has >1024 unique values -> must be dropped to avoid partition explosion
    n = 2000
    df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2025-01-01"] * n),
            "customer_id": list(range(n)),
            "region": ["x"] * n,
        }
    )
    src = tmp_path / "src" / "part_0.parquet"
    _write_parquet(src, df)

    out = tmp_path / "out"
    spec = LayoutSpec(
        partition_cols=["event_date", "customer_id"],
        sort_cols=None,
        file_size_mb=1.0,
    )

    eff_cols, derived = migrator.migrate_files_into_dataset(
        table_name="t",
        source_files=[src],
        layout_spec=spec,
        output_path=out,
    )

    assert eff_cols is not None
    assert "customer_id" not in eff_cols
    assert out.exists()


def test_incremental_partition_safety_coarsens_datetime(tmp_path):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    migrator = LayoutMigrator(config, store)

    # 2000 unique timestamps across 2 months -> coarsen to month bucket (<=1024)
    n = 2000
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2025-01-01", periods=n, freq="h"),
            "x": list(range(n)),
        }
    )
    src = tmp_path / "src" / "part_0.parquet"
    _write_parquet(src, df)

    out = tmp_path / "out"
    spec = LayoutSpec(
        partition_cols=["event_date"], sort_cols=None, file_size_mb=1.0
    )

    eff_cols, derived = migrator.migrate_files_into_dataset(
        table_name="t",
        source_files=[src],
        layout_spec=spec,
        output_path=out,
    )

    assert eff_cols == ["event_date_year_month"]
    assert derived == {"event_date_year_month": "event_date"}
    assert out.exists()
