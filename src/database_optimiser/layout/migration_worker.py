"""Async/incremental migration worker (Phase 6)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from ..config import Config
from ..storage.metadata import MetadataStore
from .migrator import LayoutMigrator
from .spec import LayoutSpec


@dataclass
class MigrationWorker:
    config: Config
    metadata_store: MetadataStore
    migrator: LayoutMigrator
    batch_files: int = 10  # for incremental mode

    def run_once(self) -> bool:
        job = self.metadata_store.claim_next_migration_job()
        if not job:
            return False

        job_id = job["job_id"]
        table_name = job["table_name"]
        layout_id = job["layout_id"]
        mode = job["mode"]
        spec = LayoutSpec.from_dict(json.loads(job["requested_spec"] or "{}"))

        try:
            source = self.migrator.get_source_path(table_name)
            if not source:
                raise ValueError(f"No source path found for {table_name}")

            if mode == "full":
                self.migrator.migrate_table(
                    table_name, source, spec, layout_id
                )
                self.metadata_store.complete_migration_job(job_id)
                return True

            if mode == "incremental":
                out = self.config.data_dir / table_name / layout_id
                out.mkdir(parents=True, exist_ok=True)
                files = self.migrator.list_parquet_files(source)
                total = len(files)
                processed = 0
                derived_partition_map: dict[str, str] = {}
                effective_partition_cols: Optional[list[str]] = None
                for i in range(0, total, max(1, self.batch_files)):
                    batch = files[i : i + self.batch_files]
                    eff_cols, derived_map = (
                        self.migrator.migrate_files_into_dataset(
                            table_name=table_name,
                            source_files=batch,
                            layout_spec=spec,
                            output_path=out,
                        )
                    )
                    if eff_cols is not None:
                        effective_partition_cols = eff_cols
                    derived_partition_map.update(derived_map)
                    processed = min(total, i + len(batch))
                    self.metadata_store.update_migration_job_progress(
                        job_id, processed
                    )

                # After incremental batches, create layout record if not already created by migrator.
                # `migrate_files_into_dataset` does not touch metadata; we create the final layout record here.
                notes = (
                    json.dumps(
                        {"derived_partition_cols": derived_partition_map}
                    )
                    if derived_partition_map
                    else None
                )
                self.metadata_store.create_layout(
                    layout_id=layout_id,
                    table_name=table_name,
                    partition_cols=effective_partition_cols
                    or spec.partition_cols,
                    sort_cols=spec.sort_cols,
                    layout_path=str(out),
                    file_size_mb=spec.file_size_mb,
                    notes=notes,
                )
                self.metadata_store.complete_migration_job(job_id)
                return True

            raise ValueError(f"Unknown migration mode: {mode}")
        except Exception as e:
            self.metadata_store.fail_migration_job(job_id, str(e))
            return True

    def run_loop(self, sleep_sec: int = 2) -> None:
        while True:
            did = self.run_once()
            if not did:
                time.sleep(sleep_sec)


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:10]}"
