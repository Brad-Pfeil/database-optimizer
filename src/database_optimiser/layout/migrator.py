"""Layout migration: rewrite Parquet datasets with new layouts."""

import glob
import shutil
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ..config import Config
from ..storage.metadata import MetadataStore
from .spec import LayoutSpec


class LayoutMigrator:
    """Migrates data to new layouts by rewriting Parquet files."""

    def __init__(
        self,
        config: Config,
        metadata_store: MetadataStore,
    ):
        """Initialize layout migrator."""
        self.config = config
        self.metadata_store = metadata_store

    def migrate_table(
        self,
        table_name: str,
        source_path: Path,
        layout_spec: LayoutSpec,
        layout_id: str,
    ) -> Path:
        """
        Migrate a table to a new layout.

        Args:
            table_name: Name of the table
            source_path: Path to source Parquet files
            layout_spec: New layout specification
            layout_id: Unique layout identifier

        Returns:
            Path to the new layout
        """
        start_time = time.time()
        requested_partition_cols = (
            list(layout_spec.partition_cols)
            if layout_spec.partition_cols
            else None
        )
        requested_sort_cols = (
            list(layout_spec.sort_cols) if layout_spec.sort_cols else None
        )

        # Create output directory (README expects layout_<id>)
        output_path = self.config.data_dir / table_name / f"layout_{layout_id}"
        tmp_path = (
            self.config.data_dir / table_name / f"layout_{layout_id}__tmp"
        )
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing layout path: {output_path}"
            )
        # The tmp path is only an intermediate artifact; if a previous run crashed,
        # it may be left behind. Clean it up so we can retry safely.
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=False)

        print(f"Migrating {table_name} to layout {layout_id}...")
        print(f"  Partition cols: {layout_spec.partition_cols}")
        print(f"  Sort cols: {layout_spec.sort_cols}")

        # Read source dataset
        source_dataset = ds.dataset(source_path, format="parquet")

        # Calculate max_rows_per_file more reasonably
        max_rows_per_file: Optional[int] = None
        if layout_spec.file_size_mb and bool(
            getattr(self.config, "migration_enable_row_chunking", False)
        ):
            # Estimate avg row size from one scanned batch to pick a reasonable rows/file.
            try:
                sample = source_dataset.scanner(batch_size=10_000).to_table()
                if sample.num_rows > 0:
                    avg_row_bytes = max(
                        1.0, float(sample.nbytes) / sample.num_rows
                    )
                    target_bytes = (
                        float(layout_spec.file_size_mb) * 1024 * 1024
                    )
                    max_rows_per_file = int(target_bytes / avg_row_bytes)
                    if max_rows_per_file <= 0:
                        max_rows_per_file = None
            except Exception:
                max_rows_per_file = None

        # PyArrow constraint: max_rows_per_group <= max_rows_per_file (if both are set).
        # If we set max_rows_per_file, always set max_rows_per_group to the same value to avoid
        # `max_rows_per_group must be less than or equal to max_rows_per_file`.
        max_rows_per_group = max_rows_per_file if max_rows_per_file else None

        # Track effective partitioning and any derived column mapping for equivalence-aware de-dupe.
        effective_partition_cols: Optional[list[str]] = None
        derived_partition_map: dict[str, str] = {}

        # Partition planning (shared with incremental mode): coarsen datetime or drop high-card cols.
        if layout_spec.partition_cols:
            effective_partition_cols, derived_partition_map = (
                self._plan_partition_cols_dataset(
                    source_dataset,
                    requested_cols=list(layout_spec.partition_cols),
                    max_partitions=1024,
                )
            )
        else:
            effective_partition_cols = None
            derived_partition_map = {}

        effective_sort_cols = [
            c
            for c in (requested_sort_cols or [])
            if c in source_dataset.schema.names
        ] or None

        self._rewrite_dataset_streaming(
            source_dataset=source_dataset,
            output_path=tmp_path,
            sort_cols=list(effective_sort_cols or []),
            effective_partition_cols=effective_partition_cols,
            derived_partition_map=derived_partition_map,
            max_rows_per_file=max_rows_per_file,
            max_rows_per_group=max_rows_per_group,
            existing_data_behavior="overwrite_or_ignore",
        )

        validation = self._validate_dataset_dir(tmp_path)
        tmp_path.rename(output_path)

        rewrite_time = time.time() - start_time

        # Record layout in metadata
        notes_obj: dict[str, object] = {
            "derived_partition_cols": derived_partition_map,
            "requested_partition_cols": requested_partition_cols,
            "effective_partition_cols": effective_partition_cols,
            "requested_sort_cols": requested_sort_cols,
            "effective_sort_cols": effective_sort_cols,
            "validation": validation,
        }
        self.metadata_store.create_layout(
            layout_id=layout_id,
            table_name=table_name,
            partition_cols=effective_partition_cols
            or requested_partition_cols,
            sort_cols=effective_sort_cols or requested_sort_cols,
            layout_path=str(output_path),
            file_size_mb=layout_spec.file_size_mb,
            notes=json.dumps(notes_obj),
        )

        print(f"Migration complete in {rewrite_time:.2f} seconds")
        print(f"  Output: {output_path}")

        return output_path

    def get_source_path(
        self,
        table_name: str,
    ) -> Optional[Path]:
        """Get the source path for a table (from active layout or initial)."""
        # Try to get active layout
        active_layout = self.metadata_store.get_active_layout(table_name)
        if active_layout:
            return Path(active_layout["layout_path"])

        # Fall back to initial layout
        initial_path = self.config.data_dir / table_name / "layout_initial"
        if initial_path.exists():
            return initial_path

        return None

    def list_parquet_files(self, source_path: Path) -> list[Path]:
        """List parquet files under a path (file or directory, recursively)."""
        if source_path.is_file():
            return [source_path]
        pattern = str(source_path / "**" / "*.parquet")
        return [Path(p) for p in glob.glob(pattern, recursive=True)]

    def migrate_files_into_dataset(
        self,
        *,
        table_name: str,
        source_files: Sequence[Path],
        layout_spec: LayoutSpec,
        output_path: Path,
    ) -> tuple[Optional[list[str]], dict[str, str]]:
        """Incrementally migrate a subset of parquet files into an output dataset.

        Returns (effective_partition_cols, derived_partition_map) for this batch.
        """
        if not source_files:
            return (None, {})

        requested_sort_cols = (
            list(layout_spec.sort_cols) if layout_spec.sort_cols else None
        )

        effective_partition_cols: Optional[list[str]] = None
        derived_partition_map: dict[str, str] = {}

        # Plan partition cols using only this batch of files (keeps memory bounded).
        batch_ds = ds.dataset([str(p) for p in source_files], format="parquet")
        if layout_spec.partition_cols:
            effective_partition_cols, derived_partition_map = (
                self._plan_partition_cols_dataset(
                    batch_ds,
                    requested_cols=list(layout_spec.partition_cols),
                    max_partitions=1024,
                )
            )

        effective_sort_cols = [
            c
            for c in (requested_sort_cols or [])
            if c in batch_ds.schema.names
        ] or None
        self._rewrite_dataset_streaming(
            source_dataset=batch_ds,
            output_path=output_path,
            sort_cols=list(effective_sort_cols or []),
            effective_partition_cols=effective_partition_cols,
            derived_partition_map=derived_partition_map,
            max_rows_per_file=None,
            max_rows_per_group=None,
            existing_data_behavior="overwrite_or_ignore",
        )
        # Caller controls metadata recording for incremental mode; return derived map as before.
        return (effective_partition_cols, derived_partition_map)

    def _plan_partition_cols_dataset(
        self,
        dataset: ds.Dataset,
        *,
        requested_cols: list[str],
        max_partitions: int,
        sample_rows: int = 250_000,
    ) -> tuple[Optional[list[str]], dict[str, str]]:
        """Choose safe partition columns using bounded sampling over a dataset.

        - Keep cols with cardinality <= max_partitions (estimated from sample)
        - For timestamp/date cols that exceed max_partitions, derive `<col>_year_month` (YYYY-MM)
        - Drop anything else; if nothing remains, return None
        """
        derived_map: dict[str, str] = {}
        chosen: list[str] = []

        schema = dataset.schema

        def _estimate_cardinality(col: str) -> int:
            # Estimate unique count up to max_partitions+1 using a bounded sample.
            seen: set[object] = set()
            rows_seen = 0
            scanner = dataset.scanner(columns=[col], batch_size=64 * 1024)
            for batch in scanner.to_batches():
                arr = batch.column(0)
                for v in arr.to_pylist():
                    if v is None:
                        continue
                    seen.add(v)
                    if len(seen) > max_partitions:
                        return max_partitions + 1
                rows_seen += len(arr)
                if rows_seen >= sample_rows:
                    break
            return len(seen)

        def _estimate_year_month_cardinality(col: str) -> int:
            seen: set[str] = set()
            rows_seen = 0
            scanner = dataset.scanner(columns=[col], batch_size=64 * 1024)
            for batch in scanner.to_batches():
                arr = batch.column(0)
                for v in arr.to_pylist():
                    if v is None:
                        continue
                    # v may be a python datetime/date; normalize to YYYY-MM
                    s = str(v)
                    # Common formats: 'YYYY-MM-DD ...' or 'YYYY-MM-DD'
                    ym = s[:7] if len(s) >= 7 else s
                    seen.add(ym)
                    if len(seen) > max_partitions:
                        return max_partitions + 1
                rows_seen += len(arr)
                if rows_seen >= sample_rows:
                    break
            return len(seen)

        for col in requested_cols:
            if col not in schema.names:
                continue

            card = _estimate_cardinality(col)
            if card <= max_partitions:
                chosen.append(col)
                continue

            field = schema.field(col)
            if pa.types.is_timestamp(field.type) or pa.types.is_date(
                field.type
            ):
                derived_col = f"{col}_year_month"
                derived_card = _estimate_year_month_cardinality(col)
                if derived_card <= max_partitions:
                    chosen.append(derived_col)
                    derived_map[derived_col] = col

        return (chosen or None, derived_map)

    def _rewrite_dataset_streaming(
        self,
        *,
        source_dataset: ds.Dataset,
        output_path: Path,
        sort_cols: list[str],
        effective_partition_cols: Optional[list[str]],
        derived_partition_map: dict[str, str],
        max_rows_per_file: Optional[int],
        max_rows_per_group: Optional[int],
        existing_data_behavior: str = "error",
    ) -> None:
        """Rewrite a dataset in bounded memory by scanning record batches."""
        # Partitioning schema (derived cols become strings)
        partitioning = None
        if effective_partition_cols:
            fields = []
            for col in effective_partition_cols:
                if col in derived_partition_map:
                    fields.append(pa.field(col, pa.string()))
                else:
                    fields.append(source_dataset.schema.field(col))
            partitioning = ds.partitioning(pa.schema(fields), flavor="hive")

        # Batch scan (batch_size is in rows)
        batch_rows = int(
            getattr(self.config, "max_rows_in_memory_for_full_sort", 131072)
            or 131072
        )
        scanner = source_dataset.scanner(batch_size=batch_rows)
        batch_idx = 0
        wrote_any = False
        for batch in scanner.to_batches():
            table = pa.Table.from_batches([batch])

            # Add derived partition columns (currently only year_month for datetime-like cols)
            for derived_col, src_col in derived_partition_map.items():
                if derived_col in table.column_names:
                    continue
                if src_col not in table.column_names:
                    continue
                src = table[src_col]
                try:
                    if pa.types.is_date(src.type):
                        src = pc.cast(src, pa.timestamp("ms"))
                    # pyarrow.compute stubs may not expose strftime; treat as dynamic.
                    pc_any: object = pc
                    ym = pc_any.strftime(src, format="%Y-%m")  # type: ignore[attr-defined]
                except Exception:
                    # If strftime fails (unexpected type), skip derived col.
                    continue
                table = table.append_column(derived_col, ym)

            sort_mode = str(
                getattr(self.config, "sort_mode", "fragment") or "fragment"
            )
            sort_mode = sort_mode.lower()

            # Bounded sorting (not global)
            if sort_cols and sort_mode != "none":
                valid = [c for c in sort_cols if c in table.column_names]
                if valid:
                    if sort_mode in ("fragment", "partition_chunk"):
                        table = table.sort_by(
                            [(c, "ascending") for c in valid]
                        )

            write_kwargs: dict[str, object] = {
                "format": "parquet",
                "existing_data_behavior": existing_data_behavior,
                "basename_template": f"part-{batch_idx}-{{i}}.parquet",
            }
            if partitioning is not None:
                write_kwargs["partitioning"] = partitioning
            if max_rows_per_file:
                write_kwargs["max_rows_per_file"] = max_rows_per_file
            if max_rows_per_group:
                write_kwargs["max_rows_per_group"] = max_rows_per_group

            ds.write_dataset(table, output_path, **write_kwargs)
            wrote_any = True
            batch_idx += 1

        if not wrote_any:
            raise ValueError("No data found to migrate (empty dataset)")

    def _validate_dataset_dir(self, path: Path) -> dict[str, object]:
        """Validate that a dataset dir is readable and non-empty; return summary info."""
        files = self.list_parquet_files(path)
        if not files:
            raise ValueError(
                f"Migration produced no parquet files under {path}"
            )
        # Ensure PyArrow can open the dataset and has a schema.
        d = ds.dataset(path, format="parquet")
        schema = d.schema
        schema_hash = hashlib.sha256(str(schema).encode("utf-8")).hexdigest()
        out: dict[str, object] = {
            "schema_hash": schema_hash,
        }
        if bool(getattr(self.config, "migration_validate_row_count", True)):
            out["row_count"] = int(d.count_rows())
        return out

    def estimate_rewrite_cost(
        self,
        table_name: str,
        source_path: Path,
    ) -> float:
        """
        Estimate the cost (time in seconds) to rewrite a table.

        This is a heuristic based on file size.
        """
        files = self.list_parquet_files(source_path)
        if not files:
            return 0.0

        total_bytes = 0
        for f in files:
            try:
                total_bytes += int(f.stat().st_size)
            except FileNotFoundError:
                continue

        total_mb = total_bytes / (1024 * 1024)
        # Heuristic model (tunable): per-file overhead + per-MB throughput cost.
        # - 0.01s per file ~= 1s per 100 files
        # - 0.002s per MB ~= 500MB/s effective rewrite speed
        return (0.01 * len(files)) + (0.002 * total_mb)
