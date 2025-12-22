"""SQLite metadata storage for query logs, layouts, and evaluations."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config


class MetadataStore:
    """Manages SQLite database for storing query logs, layouts, and evaluations."""

    def __init__(self, config: Config):
        """Initialize metadata store with config."""
        self.config = config
        self.db_path = config.metadata_db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema if it doesn't exist."""
        with self._connection() as conn:
            cursor = conn.cursor()

            # Query log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TIMESTAMP NOT NULL,
                    user_id TEXT,
                    table_name TEXT,
                    layout_id TEXT,     -- Track which layout was used
                    context_key TEXT,   -- Stable query-shape signature
                    cluster_id TEXT,    -- Deterministic workload cluster id
                    columns_used TEXT,  -- JSON array
                    predicates TEXT,    -- JSON array
                    joins TEXT,         -- JSON array
                    group_by_cols TEXT, -- JSON array
                    order_by_cols TEXT, -- JSON array
                    runtime_ms REAL NOT NULL,
                    rows_scanned INTEGER,
                    rows_returned INTEGER,
                    query_text TEXT
                )
            """)

            # Add layout_id column if it doesn't exist (for existing databases)
            # This must be done BEFORE creating indexes
            # Check if table exists and column doesn't exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='query_log'
            """)
            table_exists = cursor.fetchone()

            if table_exists:
                # Check if layout_id column exists
                cursor.execute("PRAGMA table_info(query_log)")
                columns = [row[1] for row in cursor.fetchall()]
                if "layout_id" not in columns:
                    try:
                        cursor.execute(
                            "ALTER TABLE query_log ADD COLUMN layout_id TEXT"
                        )
                    except sqlite3.OperationalError:
                        pass  # Column might have been added concurrently
                if "context_key" not in columns:
                    try:
                        cursor.execute(
                            "ALTER TABLE query_log ADD COLUMN context_key TEXT"
                        )
                    except sqlite3.OperationalError:
                        pass
                if "cluster_id" not in columns:
                    try:
                        cursor.execute(
                            "ALTER TABLE query_log ADD COLUMN cluster_id TEXT"
                        )
                    except sqlite3.OperationalError:
                        pass

            # Create index on timestamp for efficient time-window queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_log_ts 
                ON query_log(ts)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_log_table 
                ON query_log(table_name)
            """)

            # Create index on layout_id
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_log_layout 
                    ON query_log(layout_id)
                """)
            except sqlite3.OperationalError:
                pass  # Column might not exist yet

            # Create index on cluster_id (if column exists)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_log_cluster
                    ON query_log(cluster_id)
                """)
            except sqlite3.OperationalError:
                pass

            # Table layout table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_layout (
                    layout_id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 0,
                    cluster_id TEXT,        -- Optional cluster scope (NULL = global)
                    partition_cols TEXT,  -- JSON array
                    sort_cols TEXT,        -- JSON array
                    index_strategy TEXT,   -- JSON object
                    compression TEXT,     -- JSON object
                    file_size_mb REAL,
                    notes TEXT,
                    layout_path TEXT      -- Path to Parquet files
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_table_layout_table_active 
                ON table_layout(table_name, is_active)
            """)

            # Add cluster_id column for existing DBs if needed
            cursor.execute("PRAGMA table_info(table_layout)")
            tl_cols = [row[1] for row in cursor.fetchall()]
            if "cluster_id" not in tl_cols:
                try:
                    cursor.execute(
                        "ALTER TABLE table_layout ADD COLUMN cluster_id TEXT"
                    )
                except sqlite3.OperationalError:
                    pass

            # Layout evaluation table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS layout_eval (
                    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layout_id TEXT NOT NULL,
                    eval_window_start TIMESTAMP NOT NULL,
                    eval_window_end TIMESTAMP NOT NULL,
                    avg_latency_ms REAL,
                    p95_latency_ms REAL,
                    p99_latency_ms REAL,
                    avg_rows_scanned REAL,
                    queries_evaluated INTEGER,
                    rewrite_cost_sec REAL,
                    reward_score REAL,
                    FOREIGN KEY (layout_id) REFERENCES table_layout(layout_id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_layout_eval_layout 
                ON layout_eval(layout_id)
            """)

            # Migration job queue (Phase 6)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS migration_job (
                    job_id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    layout_id TEXT NOT NULL,
                    status TEXT NOT NULL,           -- queued|running|completed|failed
                    mode TEXT NOT NULL,             -- full|incremental
                    requested_spec TEXT,            -- JSON LayoutSpec
                    cluster_id TEXT,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    error TEXT,
                    total_files INTEGER,
                    processed_files INTEGER
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_migration_job_status
                ON migration_job(status, created_at)
            """)

            conn.commit()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log_query(
        self,
        table_name: Optional[str],
        columns_used: List[str],
        predicates: List[Dict[str, Any]],
        joins: List[Dict[str, Any]],
        group_by_cols: List[str],
        order_by_cols: List[str],
        runtime_ms: float,
        rows_scanned: Optional[int],
        rows_returned: int,
        query_text: str,
        user_id: Optional[str] = None,
        layout_id: Optional[str] = None,
        context_key: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> int:
        """Log a query execution."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_log (
                    ts, user_id, table_name, layout_id, context_key, cluster_id, columns_used, predicates, joins,
                    group_by_cols, order_by_cols, runtime_ms, rows_scanned,
                    rows_returned, query_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.utcnow(),
                    user_id,
                    table_name,
                    layout_id,
                    context_key,
                    cluster_id,
                    json.dumps(columns_used),
                    json.dumps(predicates),
                    json.dumps(joins),
                    json.dumps(group_by_cols),
                    json.dumps(order_by_cols),
                    runtime_ms,
                    rows_scanned,
                    rows_returned,
                    query_text,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_query_logs(
        self,
        table_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        layout_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve query logs with optional filters."""
        with self._connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM query_log WHERE 1=1"
            params = []

            if table_name:
                query += " AND table_name = ?"
                params.append(table_name)

            if start_time:
                query += " AND ts >= ?"
                params.append(start_time)

            if end_time:
                query += " AND ts <= ?"
                params.append(end_time)

            if layout_id:
                query += " AND layout_id = ?"
                params.append(layout_id)

            if cluster_id:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            query += " ORDER BY ts DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def count_query_logs(
        self,
        table_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        layout_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> int:
        """Count query logs with optional filters (more efficient than fetching rows)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(1) AS c FROM query_log WHERE 1=1"
            params = []

            if table_name:
                query += " AND table_name = ?"
                params.append(table_name)
            if start_time:
                query += " AND ts >= ?"
                params.append(start_time)
            if end_time:
                query += " AND ts <= ?"
                params.append(end_time)
            if layout_id:
                query += " AND layout_id = ?"
                params.append(layout_id)
            if cluster_id:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            cursor.execute(query, params)
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def create_layout(
        self,
        layout_id: str,
        table_name: str,
        partition_cols: Optional[List[str]],
        sort_cols: Optional[List[str]],
        layout_path: str,
        cluster_id: Optional[str] = None,
        index_strategy: Optional[Dict[str, Any]] = None,
        compression: Optional[Dict[str, Any]] = None,
        file_size_mb: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Create a new layout record."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO table_layout (
                    layout_id, table_name, created_at, is_active,
                    cluster_id, partition_cols, sort_cols, index_strategy,
                    compression, file_size_mb, notes, layout_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    layout_id,
                    table_name,
                    datetime.utcnow(),
                    False,  # New layouts start inactive
                    cluster_id,
                    json.dumps(partition_cols) if partition_cols else None,
                    json.dumps(sort_cols) if sort_cols else None,
                    json.dumps(index_strategy) if index_strategy else None,
                    json.dumps(compression) if compression else None,
                    file_size_mb,
                    notes,
                    layout_path,
                ),
            )
            conn.commit()

    def get_active_layout(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get the currently active layout for a table."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM table_layout
                WHERE table_name = ? AND is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (table_name,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_layouts(
        self, table_name: str, cluster_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all layouts for a table, optionally filtered by cluster_id (NULL cluster_id = global)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if cluster_id is None:
                cursor.execute(
                    """
                SELECT * FROM table_layout
                WHERE table_name = ?
                ORDER BY created_at DESC
                """,
                    (table_name,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM table_layout
                    WHERE table_name = ? AND cluster_id = ?
                    ORDER BY created_at DESC
                """,
                    (table_name, cluster_id),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_layout(self, layout_id: str) -> Optional[Dict[str, Any]]:
        """Get layout information by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM table_layout
                WHERE layout_id = ?
            """,
                (layout_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def activate_layout(self, layout_id: str, table_name: str) -> None:
        """Activate a layout and deactivate others for the same table."""
        with self._connection() as conn:
            cursor = conn.cursor()
            # Deactivate all layouts for this table
            cursor.execute(
                """
                UPDATE table_layout
                SET is_active = 0
                WHERE table_name = ?
            """,
                (table_name,),
            )
            # Activate the specified layout
            cursor.execute(
                """
                UPDATE table_layout
                SET is_active = 1
                WHERE layout_id = ?
            """,
                (layout_id,),
            )
            conn.commit()

    def record_evaluation(
        self,
        layout_id: str,
        eval_window_start: datetime,
        eval_window_end: datetime,
        avg_latency_ms: float,
        p95_latency_ms: Optional[float] = None,
        p99_latency_ms: Optional[float] = None,
        avg_rows_scanned: Optional[float] = None,
        queries_evaluated: int = 0,
        rewrite_cost_sec: float = 0.0,
        reward_score: Optional[float] = None,
    ) -> int:
        """Record layout evaluation results."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO layout_eval (
                    layout_id, eval_window_start, eval_window_end,
                    avg_latency_ms, p95_latency_ms, p99_latency_ms,
                    avg_rows_scanned, queries_evaluated, rewrite_cost_sec,
                    reward_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    layout_id,
                    eval_window_start,
                    eval_window_end,
                    avg_latency_ms,
                    p95_latency_ms,
                    p99_latency_ms,
                    avg_rows_scanned,
                    queries_evaluated,
                    rewrite_cost_sec,
                    reward_score,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_layout_evaluations(
        self,
        layout_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get evaluation history for a layout."""
        with self._connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM layout_eval
                WHERE layout_id = ?
                ORDER BY eval_window_end DESC
            """
            params = [layout_id]
            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_latest_evaluation(
        self, layout_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest evaluation row for a layout (by eval_window_end)."""
        rows = self.get_layout_evaluations(layout_id, limit=1)
        return rows[0] if rows else None

    def get_latest_scored_eval(
        self, layout_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest *scored* evaluation (reward_score is not NULL)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM layout_eval
                WHERE layout_id = ? AND reward_score IS NOT NULL
                ORDER BY eval_window_end DESC
                LIMIT 1
                """,
                (layout_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_layout_eval_history(
        self,
        *,
        table_name: str,
        layout_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        only_scored: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get evaluation history scoped to a table (optionally a layout/cluster)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT
                    le.*,
                    tl.table_name AS table_name,
                    tl.cluster_id AS cluster_id
                FROM layout_eval le
                JOIN table_layout tl ON tl.layout_id = le.layout_id
                WHERE tl.table_name = ?
            """
            params: list[Any] = [table_name]

            if layout_id:
                query += " AND le.layout_id = ?"
                params.append(layout_id)

            if cluster_id is not None:
                query += " AND tl.cluster_id = ?"
                params.append(cluster_id)

            if since is not None:
                query += " AND le.eval_window_end >= ?"
                params.append(since)

            if only_scored:
                query += " AND le.reward_score IS NOT NULL"

            query += " ORDER BY le.eval_window_end DESC"

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_layout_query_counts(
        self,
        *,
        table_name: str,
        cluster_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return routing distribution: count of queries by layout_id (and cluster_id)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT
                    cluster_id,
                    layout_id,
                    COUNT(1) AS query_count
                FROM query_log
                WHERE table_name = ?
            """
            params: list[Any] = [table_name]

            if cluster_id is not None:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            if since is not None:
                query += " AND ts >= ?"
                params.append(since)

            if until is not None:
                query += " AND ts <= ?"
                params.append(until)

            query += """
                GROUP BY cluster_id, layout_id
                ORDER BY query_count DESC
            """
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------
    # Migration job queue (Phase 6)
    # -------------------------

    def enqueue_migration_job(
        self,
        *,
        job_id: str,
        table_name: str,
        layout_id: str,
        mode: str,
        requested_spec_json: str,
        cluster_id: Optional[str] = None,
        total_files: Optional[int] = None,
    ) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO migration_job (
                    job_id, table_name, layout_id, status, mode, requested_spec, cluster_id,
                    created_at, total_files, processed_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    table_name,
                    layout_id,
                    "queued",
                    mode,
                    requested_spec_json,
                    cluster_id,
                    datetime.utcnow(),
                    total_files,
                    0,
                ),
            )
            conn.commit()

    def claim_next_migration_job(self) -> Optional[Dict[str, Any]]:
        """Atomically claim the next queued job and mark it running."""
        with self._connection() as conn:
            cursor = conn.cursor()
            conn.isolation_level = None
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                SELECT * FROM migration_job
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if not row:
                cursor.execute("COMMIT")
                return None
            job_id = row["job_id"]
            cursor.execute(
                """
                UPDATE migration_job
                SET status = 'running', started_at = ?
                WHERE job_id = ?
                """,
                (datetime.utcnow(), job_id),
            )
            cursor.execute("COMMIT")
            return dict(row)

    def update_migration_job_progress(
        self, job_id: str, processed_files: int
    ) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE migration_job SET processed_files = ? WHERE job_id = ?",
                (processed_files, job_id),
            )
            conn.commit()

    def complete_migration_job(self, job_id: str) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE migration_job SET status='completed', finished_at=? WHERE job_id=?",
                (datetime.utcnow(), job_id),
            )
            conn.commit()

    def fail_migration_job(self, job_id: str, error: str) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE migration_job SET status='failed', finished_at=?, error=? WHERE job_id=?",
                (datetime.utcnow(), error, job_id),
            )
            conn.commit()

    def get_migration_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM migration_job WHERE job_id = ?", (job_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
