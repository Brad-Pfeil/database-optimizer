from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from database_optimiser.config import Config
from database_optimiser.layout.migration_worker import MigrationWorker
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.storage.metadata import MetadataStore


def _write_parquet_files(dir_path: Path, nfiles: int = 2) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(nfiles):
        df = pd.DataFrame({"a": [i, i + 1], "b": [10, 11]})
        df.to_parquet(dir_path / f"part_{i}.parquet", index=False)


def test_incremental_migration_job_completes(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        migration_batch_files=1,
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    migrator = LayoutMigrator(config, store)

    # Prepare an initial layout on disk (no active layout in metadata).
    initial_path = config.data_dir / "t" / "layout_initial"
    _write_parquet_files(initial_path, nfiles=2)

    job_id = "job_test"
    layout_id = "layout_new"
    spec = LayoutSpec(partition_cols=None, sort_cols=None, file_size_mb=1.0)
    total_files = len(migrator.list_parquet_files(initial_path))

    store.enqueue_migration_job(
        job_id=job_id,
        table_name="t",
        layout_id=layout_id,
        mode="incremental",
        requested_spec_json=json.dumps(spec.to_dict()),
        total_files=total_files,
    )

    worker = MigrationWorker(
        config=config, metadata_store=store, migrator=migrator, batch_files=1
    )
    did = worker.run_once()
    assert did

    job = store.get_migration_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert int(job["processed_files"] or 0) == total_files

    layout = store.get_layout(layout_id)
    assert layout is not None
    assert Path(layout["layout_path"]).exists()
