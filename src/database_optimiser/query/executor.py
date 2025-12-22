"""Query executor with DuckDB and telemetry logging."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb

from ..config import Config
from ..storage.metadata import MetadataStore
from ..storage.query_logger import QueryLogger


class QueryExecutor:
    """Executes queries via DuckDB and logs telemetry."""

    def __init__(
        self,
        config: Config,
        metadata_store: MetadataStore,
        query_logger: QueryLogger,
    ):
        """Initialize query executor."""
        self.config = config
        self.metadata_store = metadata_store
        self.query_logger = query_logger

        # Create DuckDB connection
        self.conn = duckdb.connect()

        # Register Parquet extension
        self.conn.execute("INSTALL parquet; LOAD parquet;")

        # Track current layout per table
        self.current_layouts: Dict[str, Optional[str]] = {}

    def register_table(self, table_name: str, parquet_path: str) -> None:
        """Register a Parquet dataset as a table in DuckDB."""
        import glob

        path_obj = Path(parquet_path)

        # For directories, we need to handle Hive-style partitioned datasets
        # DuckDB's read_parquet can accept a list of file paths
        if path_obj.is_dir():
            # Find all parquet files recursively (handles Hive-style partitioning)
            pattern = str(path_obj / "**" / "*.parquet")
            parquet_files = glob.glob(pattern, recursive=True)

            if parquet_files:
                # DuckDB can read from a list of file paths
                # Format as a list of strings for DuckDB
                file_list = "', '".join(parquet_files)
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name} AS
                    SELECT * FROM read_parquet(['{file_list}'])
                """)
            else:
                # Fallback: try direct directory (might work for non-partitioned)
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name} AS
                    SELECT * FROM read_parquet('{str(path_obj)}/*.parquet')
                """)
        else:
            # Single file
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT * FROM read_parquet('{parquet_path}')
            """)

    def register_layout(
        self,
        table_name: str,
        layout_path: str,
        layout_id: Optional[str] = None,
    ) -> None:
        """Register a specific layout version for a table."""
        self.register_table(table_name, layout_path)
        if layout_id:
            self.current_layouts[table_name] = layout_id

    def run_query(
        self,
        sql: str,
        user_id: Optional[str] = None,
        return_df: bool = True,
    ) -> Any:
        """
        Execute a query and log telemetry.

        Args:
            sql: SQL query string
            user_id: Optional user identifier
            return_df: If True, return pandas DataFrame; else return DuckDB result

        Returns:
            Query results (DataFrame or DuckDB result)
        """
        start_time = time.time()

        try:
            # Execute query
            result = self.conn.execute(sql)

            # Fetch results
            if return_df:
                df = result.fetchdf()
                rows_returned = len(df)
            else:
                df = None
                rows_returned = (
                    result.rowcount if hasattr(result, "rowcount") else 0
                )

            end_time = time.time()
            runtime_ms = (end_time - start_time) * 1000

            # Try to get rows scanned (approximate from EXPLAIN)
            rows_scanned = self._estimate_rows_scanned(sql)

            # Get current layout for the table (if any)
            # Try to determine table name from SQL
            table_name = None
            import re

            from_match = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
            if from_match:
                table_name = from_match.group(1)

            layout_id = None
            if table_name and table_name in self.current_layouts:
                layout_id = self.current_layouts[table_name]
            elif table_name:
                # Check if there's an active layout for this table
                active_layout = self.metadata_store.get_active_layout(
                    table_name
                )
                if active_layout:
                    layout_id = active_layout["layout_id"]
                    self.current_layouts[table_name] = layout_id

            # Log query execution
            self.query_logger.log_query_execution(
                sql=sql,
                runtime_ms=runtime_ms,
                rows_returned=rows_returned,
                rows_scanned=rows_scanned,
                user_id=user_id,
                layout_id=layout_id,
            )

            return df if return_df else result

        except Exception:
            end_time = time.time()
            runtime_ms = (end_time - start_time) * 1000

            # Log failed query (with error info)
            try:
                self.query_logger.log_query_execution(
                    sql=sql,
                    runtime_ms=runtime_ms,
                    rows_returned=0,
                    rows_scanned=None,
                    user_id=user_id,
                )
            except Exception:
                pass  # Don't fail on logging errors

            raise

    def _estimate_rows_scanned(self, sql: str) -> Optional[int]:
        """Estimate rows scanned using EXPLAIN."""
        try:
            explain_sql = f"EXPLAIN {sql}"
            explain_result = self.conn.execute(explain_sql).fetchall()

            # Try to extract row count from explain output
            # This is a heuristic - DuckDB's EXPLAIN format may vary
            for row in explain_result:
                row_str = str(row).lower()
                # Look for patterns like "rows: 123456"
                import re

                match = re.search(r"rows?[:\s]+(\d+)", row_str)
                if match:
                    return int(match.group(1))

            return None
        except Exception:
            return None

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a registered table."""
        try:
            # Get row count
            count_result = self.conn.execute(
                f"SELECT COUNT(*) as cnt FROM {table_name}"
            ).fetchone()
            row_count = count_result[0] if count_result else 0

            # Get column info
            desc_result = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()

            return {
                "row_count": row_count,
                "columns": desc_result.to_dict("records"),
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self) -> None:
        """Close DuckDB connection."""
        self.conn.close()
