"""Layout migration: rewrite Parquet datasets with new layouts."""

import glob
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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

        # Create output directory
        output_path = self.config.data_dir / table_name / layout_id
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Migrating {table_name} to layout {layout_id}...")
        print(f"  Partition cols: {layout_spec.partition_cols}")
        print(f"  Sort cols: {layout_spec.sort_cols}")

        # Read source dataset
        source_dataset = ds.dataset(source_path, format="parquet")

        # Convert to pandas for easier manipulation
        # Read in chunks to manage memory
        chunks = []
        for fragment in source_dataset.get_fragments():
            table = fragment.to_table()
            df = table.to_pandas()
            chunks.append(df)

        if not chunks:
            raise ValueError(f"No data found in {source_path}")

        # Combine chunks
        combined_df = pd.concat(chunks, ignore_index=True)

        # Apply sorting if specified
        if layout_spec.sort_cols:
            # Ensure all sort columns exist
            valid_sort_cols = [
                col
                for col in layout_spec.sort_cols
                if col in combined_df.columns
            ]
            if valid_sort_cols:
                combined_df = combined_df.sort_values(by=valid_sort_cols)

        # Convert back to PyArrow table
        table = pa.Table.from_pandas(combined_df)

        # Calculate max_rows_per_file more reasonably
        # Estimate: assume ~1KB per row on average, so 128MB ≈ 128,000 rows
        if layout_spec.file_size_mb:
            max_rows_per_file = int(
                layout_spec.file_size_mb * 1000
            )  # Rough estimate: 1KB per row
        else:
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
                self._plan_partition_cols(
                    combined_df,
                    requested_cols=list(layout_spec.partition_cols),
                    max_partitions=1024,
                )
            )
            # Ensure layout_spec reflects effective physical layout.
            layout_spec.partition_cols = effective_partition_cols
            # Rebuild table if we added derived columns
            table = pa.Table.from_pandas(combined_df)
        else:
            effective_partition_cols = None
            derived_partition_map = {}

        if effective_partition_cols:
            partitioning = ds.partitioning(
                pa.schema(
                    [
                        table.schema.field(col)
                        for col in effective_partition_cols
                    ]
                ),
                flavor="hive",
            )
            write_kwargs = {"format": "parquet", "partitioning": partitioning}
        else:
            write_kwargs = {"format": "parquet"}

        if max_rows_per_file:
            write_kwargs["max_rows_per_file"] = max_rows_per_file
            write_kwargs["max_rows_per_group"] = max_rows_per_group

        ds.write_dataset(table, output_path, **write_kwargs)

        rewrite_time = time.time() - start_time

        # Record layout in metadata
        self.metadata_store.create_layout(
            layout_id=layout_id,
            table_name=table_name,
            partition_cols=effective_partition_cols
            or layout_spec.partition_cols,
            sort_cols=layout_spec.sort_cols,
            layout_path=str(output_path),
            file_size_mb=layout_spec.file_size_mb,
            notes=(
                json.dumps({"derived_partition_cols": derived_partition_map})
                if derived_partition_map
                else None
            ),
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

        dfs = []
        for f in source_files:
            t = pq.read_table(str(f))
            dfs.append(t.to_pandas())
        combined_df = (
            pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        )
        table = pa.Table.from_pandas(combined_df)

        # Sorting
        if layout_spec.sort_cols:
            valid_sort_cols = [
                c for c in layout_spec.sort_cols if c in combined_df.columns
            ]
            if valid_sort_cols:
                combined_df = combined_df.sort_values(by=valid_sort_cols)
                table = pa.Table.from_pandas(combined_df)

        write_kwargs = {
            "format": "parquet",
            "existing_data_behavior": "overwrite_or_ignore",
        }
        effective_partition_cols: Optional[list[str]] = None
        derived_partition_map: dict[str, str] = {}
        if layout_spec.partition_cols:
            effective_partition_cols, derived_partition_map = (
                self._plan_partition_cols(
                    combined_df,
                    requested_cols=list(layout_spec.partition_cols),
                    max_partitions=1024,
                )
            )
            # Rebuild table if we added derived columns
            table = pa.Table.from_pandas(combined_df)

        if effective_partition_cols:
            partitioning = ds.partitioning(
                pa.schema(
                    [
                        table.schema.field(col)
                        for col in effective_partition_cols
                    ]
                ),
                flavor="hive",
            )
            write_kwargs["partitioning"] = partitioning

        ds.write_dataset(table, output_path, **write_kwargs)
        return (effective_partition_cols, derived_partition_map)

    def _plan_partition_cols(
        self,
        df: pd.DataFrame,
        *,
        requested_cols: list[str],
        max_partitions: int,
    ) -> tuple[Optional[list[str]], dict[str, str]]:
        """Choose safe partition columns for a dataframe, optionally deriving coarsened datetime cols.

        - Keep cols with cardinality <= max_partitions
        - For datetime-like cols that exceed max_partitions, derive `<col>_year_month`
        - Drop anything else; if nothing remains, return None
        """
        derived_map: dict[str, str] = {}
        chosen: list[str] = []

        for col in requested_cols:
            if col not in df.columns:
                continue
            cardinality = int(df[col].nunique(dropna=True))
            if cardinality <= max_partitions:
                chosen.append(col)
                continue

            # Try to coarsen datetime-like columns
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                derived_col = f"{col}_year_month"
                if derived_col not in df.columns:
                    df[derived_col] = df[col].dt.to_period("M").astype(str)
                derived_cardinality = int(df[derived_col].nunique(dropna=True))
                if derived_cardinality <= max_partitions:
                    chosen.append(derived_col)
                    derived_map[derived_col] = col
                    continue

            # Otherwise drop unsafe col.

        return (chosen or None, derived_map)

    def estimate_rewrite_cost(
        self,
        table_name: str,
        source_path: Path,
    ) -> float:
        """
        Estimate the cost (time in seconds) to rewrite a table.

        This is a heuristic based on file size.
        """
        # Count Parquet files
        parquet_files = list(source_path.glob("*.parquet"))
        if not parquet_files:
            return 0.0

        # Estimate based on number of files (rough heuristic)
        # Assume ~1 second per 100 files
        return len(parquet_files) / 100.0
