"""SQLite metadata storage for query logs, layouts, and evaluations.

This module uses a minimal, explicit schema migration mechanism via `schema_version`.
Timestamps are stored as epoch milliseconds (INTEGER) in SQLite columns (SQLite is
dynamically typed; we use integer values consistently).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional

from ..config import Config


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ts_param(value: Any) -> Any:
    """Convert datetime inputs to epoch ms; pass through ints/floats/None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _dt_to_ms(value)
    return value


class MetadataStore:
    """Manages SQLite database for storing query logs, layouts, and evaluations."""

    LATEST_SCHEMA_VERSION = 1

    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.metadata_db_path
        self._init_schema()

    # -------------------------
    # Schema + migrations
    # -------------------------
    def _init_schema(self) -> None:
        with self._connection() as conn:
            cur = conn.cursor()
            self._ensure_schema_version(cur)
            self._run_migrations(cur)
            conn.commit()

    def _ensure_schema_version(self, cur: sqlite3.Cursor) -> None:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)"
        )
        cur.execute("SELECT COUNT(1) FROM schema_version")
        if int(cur.fetchone()[0]) == 0:
            cur.execute("INSERT INTO schema_version(version) VALUES (0)")

    def _get_schema_version(self, cur: sqlite3.Cursor) -> int:
        cur.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def _set_schema_version(self, cur: sqlite3.Cursor, v: int) -> None:
        cur.execute("UPDATE schema_version SET version = ?", (int(v),))

    def _run_migrations(self, cur: sqlite3.Cursor) -> None:
        v = self._get_schema_version(cur)
        while v < self.LATEST_SCHEMA_VERSION:
            if v == 0:
                self._migration_1(cur)
                v = 1
                self._set_schema_version(cur, v)
            else:
                break

    def _migration_1(self, cur: sqlite3.Cursor) -> None:
        """Initial schema + indices + epoch-ms backfill (best-effort)."""

        # query_log
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL,          -- stored as epoch ms integer
                user_id TEXT,
                table_name TEXT,
                layout_id TEXT,
                context_key TEXT,
                cluster_id TEXT,
                parse_success BOOLEAN,
                parse_confidence REAL,
                parser_version TEXT,
                columns_used TEXT,
                predicates TEXT,
                joins TEXT,
                group_by_cols TEXT,
                order_by_cols TEXT,
                runtime_ms REAL NOT NULL,
                rows_scanned INTEGER,
                rows_returned INTEGER,
                query_text TEXT
            )
            """
        )

        # Ensure columns exist for older databases (SQLite can't ALTER types, but can add columns).
        cur.execute("PRAGMA table_info(query_log)")
        ql_cols = {row[1] for row in cur.fetchall()}
        for name, typ in [
            ("user_id", "TEXT"),
            ("layout_id", "TEXT"),
            ("context_key", "TEXT"),
            ("cluster_id", "TEXT"),
            ("parse_success", "BOOLEAN"),
            ("parse_confidence", "REAL"),
            ("parser_version", "TEXT"),
            ("columns_used", "TEXT"),
            ("predicates", "TEXT"),
            ("joins", "TEXT"),
            ("group_by_cols", "TEXT"),
            ("order_by_cols", "TEXT"),
            ("runtime_ms", "REAL"),
            ("rows_scanned", "INTEGER"),
            ("rows_returned", "INTEGER"),
            ("query_text", "TEXT"),
        ]:
            if name not in ql_cols:
                try:
                    cur.execute(
                        f"ALTER TABLE query_log ADD COLUMN {name} {typ}"
                    )
                except sqlite3.OperationalError:
                    pass

        # table_layout
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS table_layout (
                layout_id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,  -- epoch ms integer
                is_active BOOLEAN NOT NULL DEFAULT 0,
                cluster_id TEXT,
                partition_cols TEXT,
                sort_cols TEXT,
                index_strategy TEXT,
                compression TEXT,
                file_size_mb REAL,
                notes TEXT,
                layout_path TEXT
            )
            """
        )
        cur.execute("PRAGMA table_info(table_layout)")
        tl_cols = {row[1] for row in cur.fetchall()}
        for name, typ in [
            ("cluster_id", "TEXT"),
            ("partition_cols", "TEXT"),
            ("sort_cols", "TEXT"),
            ("index_strategy", "TEXT"),
            ("compression", "TEXT"),
            ("file_size_mb", "REAL"),
            ("notes", "TEXT"),
            ("layout_path", "TEXT"),
        ]:
            if name not in tl_cols:
                try:
                    cur.execute(
                        f"ALTER TABLE table_layout ADD COLUMN {name} {typ}"
                    )
                except sqlite3.OperationalError:
                    pass

        # layout_eval
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS layout_eval (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                layout_id TEXT NOT NULL,
                table_name TEXT,
                cluster_id TEXT,
                baseline_layout_id TEXT,
                eval_mode TEXT,
                eval_status TEXT,
                eval_window_start TIMESTAMP NOT NULL,  -- epoch ms integer
                eval_window_end TIMESTAMP NOT NULL,    -- epoch ms integer
                avg_latency_ms REAL,
                p95_latency_ms REAL,
                p99_latency_ms REAL,
                avg_rows_scanned REAL,
                queries_evaluated INTEGER,
                rewrite_cost_sec REAL,
                reward_score REAL,
                FOREIGN KEY (layout_id) REFERENCES table_layout(layout_id)
            )
            """
        )
        cur.execute("PRAGMA table_info(layout_eval)")
        le_cols = {row[1] for row in cur.fetchall()}
        for name, typ in [
            ("table_name", "TEXT"),
            ("cluster_id", "TEXT"),
            ("baseline_layout_id", "TEXT"),
            ("eval_mode", "TEXT"),
            ("eval_status", "TEXT"),
            ("p95_latency_ms", "REAL"),
            ("p99_latency_ms", "REAL"),
            ("avg_rows_scanned", "REAL"),
            ("queries_evaluated", "INTEGER"),
            ("rewrite_cost_sec", "REAL"),
            ("reward_score", "REAL"),
        ]:
            if name not in le_cols:
                try:
                    cur.execute(
                        f"ALTER TABLE layout_eval ADD COLUMN {name} {typ}"
                    )
                except sqlite3.OperationalError:
                    pass

        # migration_job
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_job (
                job_id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                layout_id TEXT NOT NULL,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                requested_spec TEXT,
                cluster_id TEXT,
                created_at TIMESTAMP NOT NULL,  -- epoch ms integer
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                error TEXT,
                total_files INTEGER,
                processed_files INTEGER
            )
            """
        )
        cur.execute("PRAGMA table_info(migration_job)")
        mj_cols = {row[1] for row in cur.fetchall()}
        for name, typ in [
            ("requested_spec", "TEXT"),
            ("cluster_id", "TEXT"),
            ("started_at", "TIMESTAMP"),
            ("finished_at", "TIMESTAMP"),
            ("error", "TEXT"),
            ("total_files", "INTEGER"),
            ("processed_files", "INTEGER"),
        ]:
            if name not in mj_cols:
                try:
                    cur.execute(
                        f"ALTER TABLE migration_job ADD COLUMN {name} {typ}"
                    )
                except sqlite3.OperationalError:
                    pass

        # query_log indices (single + composites)
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log(ts)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_table ON query_log(table_name)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_layout ON query_log(layout_id)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_cluster ON query_log(cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_table_ts ON query_log(table_name, ts)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_table_cluster_ts ON query_log(table_name, cluster_id, ts)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_table_layout_ts ON query_log(table_name, layout_id, ts)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_table_context_ts ON query_log(table_name, context_key, ts)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_table_cluster_layout_ts ON query_log(table_name, cluster_id, layout_id, ts)",
        ]:
            try:
                cur.execute(stmt)
            except sqlite3.OperationalError:
                pass

        # table_layout indices
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_table_layout_table_cluster_active ON table_layout(table_name, cluster_id, is_active)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_table_layout_table_cluster ON table_layout(table_name, cluster_id)"
        )

        # layout_eval indices
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_layout_eval_layout ON layout_eval(layout_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_layout_eval_layout_end ON layout_eval(layout_id, eval_window_end)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_layout_eval_table_cluster_end ON layout_eval(table_name, cluster_id, eval_window_end)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_layout_eval_table_cluster_status_end ON layout_eval(table_name, cluster_id, eval_status, eval_window_end)"
        )

        # migration_job indices
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_migration_job_status ON migration_job(status, created_at)"
        )

        # Best-effort backfill: convert TEXT timestamps to epoch ms ints in-place
        for table, col in [
            ("query_log", "ts"),
            ("table_layout", "created_at"),
            ("layout_eval", "eval_window_start"),
            ("layout_eval", "eval_window_end"),
            ("migration_job", "created_at"),
            ("migration_job", "started_at"),
            ("migration_job", "finished_at"),
        ]:
            try:
                cur.execute(
                    f"UPDATE {table} "
                    f"SET {col} = CAST(strftime('%s', {col}) AS INTEGER) * 1000 "
                    f"WHERE {col} IS NOT NULL AND typeof({col}) = 'text'"
                )
            except sqlite3.OperationalError:
                pass

    # -------------------------
    # Connections
    # -------------------------
    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # -------------------------
    # Query logging
    # -------------------------
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
        parse_success: Optional[bool] = None,
        parse_confidence: Optional[float] = None,
        parser_version: Optional[str] = None,
    ) -> int:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO query_log (
                    ts, user_id, table_name, layout_id, context_key, cluster_id,
                    parse_success, parse_confidence, parser_version,
                    columns_used, predicates, joins,
                    group_by_cols, order_by_cols, runtime_ms, rows_scanned,
                    rows_returned, query_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_ms(),
                    user_id,
                    table_name,
                    layout_id,
                    context_key,
                    cluster_id,
                    parse_success,
                    parse_confidence,
                    parser_version,
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
            return int(cur.lastrowid)

    def get_query_logs(
        self,
        table_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        layout_id: Optional[str] = None,
        layout_id_is_null: bool = False,
        cluster_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            query = "SELECT * FROM query_log WHERE 1=1"
            params: list[Any] = []

            if table_name:
                query += " AND table_name = ?"
                params.append(table_name)
            if start_time:
                query += " AND ts >= ?"
                params.append(_ts_param(start_time))
            if end_time:
                query += " AND ts <= ?"
                params.append(_ts_param(end_time))
            if layout_id_is_null:
                query += " AND layout_id IS NULL"
            elif layout_id:
                query += " AND layout_id = ?"
                params.append(layout_id)
            if cluster_id is not None:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            query += " ORDER BY ts DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(int(limit))

            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def count_query_logs(
        self,
        table_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        layout_id: Optional[str] = None,
        layout_id_is_null: bool = False,
        cluster_id: Optional[str] = None,
    ) -> int:
        with self._connection() as conn:
            cur = conn.cursor()
            query = "SELECT COUNT(1) AS c FROM query_log WHERE 1=1"
            params: list[Any] = []

            if table_name:
                query += " AND table_name = ?"
                params.append(table_name)
            if start_time:
                query += " AND ts >= ?"
                params.append(_ts_param(start_time))
            if end_time:
                query += " AND ts <= ?"
                params.append(_ts_param(end_time))
            if layout_id_is_null:
                query += " AND layout_id IS NULL"
            elif layout_id:
                query += " AND layout_id = ?"
                params.append(layout_id)
            if cluster_id is not None:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            cur.execute(query, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0

    # -------------------------
    # Layouts
    # -------------------------
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
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
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
                    _now_ms(),
                    False,
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

    def get_active_layout(
        self, table_name: str, *, cluster_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            if cluster_id is None:
                cur.execute(
                    """
                    SELECT * FROM table_layout
                    WHERE table_name = ? AND cluster_id IS NULL AND is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (table_name,),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM table_layout
                    WHERE table_name = ? AND cluster_id = ? AND is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (table_name, cluster_id),
                )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_all_layouts(
        self, table_name: str, cluster_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            if cluster_id is None:
                cur.execute(
                    """
                    SELECT * FROM table_layout
                    WHERE table_name = ?
                    ORDER BY created_at DESC
                    """,
                    (table_name,),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM table_layout
                    WHERE table_name = ? AND cluster_id = ?
                    ORDER BY created_at DESC
                    """,
                    (table_name, cluster_id),
                )
            return [dict(r) for r in cur.fetchall()]

    def get_layout(self, layout_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM table_layout WHERE layout_id = ?",
                (layout_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def activate_layout(self, layout_id: str, table_name: str) -> None:
        """Activate a layout within its (table_name, cluster_id) scope."""
        info = self.get_layout(layout_id)
        if not info:
            raise ValueError(f"Layout {layout_id} not found")
        cluster_id = info.get("cluster_id")
        with self._connection() as conn:
            cur = conn.cursor()
            if cluster_id is None:
                cur.execute(
                    """
                    UPDATE table_layout
                    SET is_active = 0
                    WHERE table_name = ? AND cluster_id IS NULL
                    """,
                    (table_name,),
                )
                cur.execute(
                    """
                    UPDATE table_layout
                    SET is_active = 1
                    WHERE layout_id = ?
                    """,
                    (layout_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE table_layout
                    SET is_active = 0
                    WHERE table_name = ? AND cluster_id = ?
                    """,
                    (table_name, cluster_id),
                )
                cur.execute(
                    """
                    UPDATE table_layout
                    SET is_active = 1
                    WHERE layout_id = ?
                    """,
                    (layout_id,),
                )
            conn.commit()

    # -------------------------
    # Evaluations
    # -------------------------
    def record_evaluation(
        self,
        layout_id: str,
        eval_window_start: datetime,
        eval_window_end: datetime,
        avg_latency_ms: float,
        table_name: Optional[str] = None,
        cluster_id: Optional[str] = None,
        baseline_layout_id: Optional[str] = None,
        eval_mode: Optional[str] = None,
        eval_status: Optional[str] = None,
        p95_latency_ms: Optional[float] = None,
        p99_latency_ms: Optional[float] = None,
        avg_rows_scanned: Optional[float] = None,
        queries_evaluated: int = 0,
        rewrite_cost_sec: float = 0.0,
        reward_score: Optional[float] = None,
    ) -> int:
        # Fill denormalized fields from layout metadata if not provided.
        if table_name is None or cluster_id is None:
            info = self.get_layout(layout_id)
            if info:
                if table_name is None:
                    table_name = info.get("table_name")
                # If caller didn't specify cluster_id, inherit layout scope.
                if cluster_id is None:
                    cluster_id = info.get("cluster_id")
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO layout_eval (
                    layout_id, table_name, cluster_id, baseline_layout_id, eval_mode, eval_status,
                    eval_window_start, eval_window_end,
                    avg_latency_ms, p95_latency_ms, p99_latency_ms,
                    avg_rows_scanned, queries_evaluated, rewrite_cost_sec, reward_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    layout_id,
                    table_name,
                    cluster_id,
                    baseline_layout_id,
                    eval_mode,
                    eval_status,
                    _dt_to_ms(eval_window_start),
                    _dt_to_ms(eval_window_end),
                    avg_latency_ms,
                    p95_latency_ms,
                    p99_latency_ms,
                    avg_rows_scanned,
                    int(queries_evaluated),
                    float(rewrite_cost_sec or 0.0),
                    reward_score,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_layout_evaluations(
        self, layout_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            q = """
                SELECT * FROM layout_eval
                WHERE layout_id = ?
                ORDER BY eval_window_end DESC
            """
            params: list[Any] = [layout_id]
            if limit is not None:
                q += " LIMIT ?"
                params.append(int(limit))
            cur.execute(q, params)
            return [dict(r) for r in cur.fetchall()]

    def get_latest_evaluation(
        self, layout_id: str
    ) -> Optional[Dict[str, Any]]:
        rows = self.get_layout_evaluations(layout_id, limit=1)
        return rows[0] if rows else None

    def get_latest_scored_eval(
        self, layout_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM layout_eval
                WHERE layout_id = ? AND reward_score IS NOT NULL
                ORDER BY eval_window_end DESC
                LIMIT 1
                """,
                (layout_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_layout_reward_summary(self, layout_id: str) -> Dict[str, Any]:
        """Return mean_reward, n_scored, and last_eval_end for a layout."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COALESCE(AVG(reward_score), 0.0) AS mean_reward,
                    COALESCE(COUNT(reward_score), 0) AS n_scored,
                    MAX(eval_window_end) AS last_eval_end
                FROM layout_eval
                WHERE layout_id = ? AND reward_score IS NOT NULL
                """,
                (layout_id,),
            )
            row = cur.fetchone()
            return (
                dict(row)
                if row
                else {"mean_reward": 0.0, "n_scored": 0, "last_eval_end": None}
            )

    def get_reward_stats_for_table(
        self, *, table_name: str, cluster_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            if cluster_id is None:
                cur.execute(
                    """
                    SELECT
                        tl.layout_id AS layout_id,
                        COALESCE(COUNT(le.reward_score), 0) AS n,
                        COALESCE(AVG(le.reward_score), 0.0) AS mean_reward,
                        MAX(le.eval_window_end) AS last_eval_end
                    FROM table_layout tl
                    LEFT JOIN layout_eval le
                        ON le.layout_id = tl.layout_id AND le.reward_score IS NOT NULL
                    WHERE tl.table_name = ?
                    GROUP BY tl.layout_id
                    """,
                    (table_name,),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        tl.layout_id AS layout_id,
                        COALESCE(COUNT(le.reward_score), 0) AS n,
                        COALESCE(AVG(le.reward_score), 0.0) AS mean_reward,
                        MAX(le.eval_window_end) AS last_eval_end
                    FROM table_layout tl
                    LEFT JOIN layout_eval le
                        ON le.layout_id = tl.layout_id AND le.reward_score IS NOT NULL
                    WHERE tl.table_name = ? AND tl.cluster_id = ?
                    GROUP BY tl.layout_id
                    """,
                    (table_name, cluster_id),
                )
            rows = cur.fetchall()
            out: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                d = dict(r)
                out[str(d["layout_id"])] = d
            return out

    def get_recent_rewrite_costs_sec(
        self,
        *,
        table_name: str,
        cluster_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[float]:
        with self._connection() as conn:
            cur = conn.cursor()
            q = """
                SELECT rewrite_cost_sec
                FROM layout_eval
                WHERE table_name = ? AND rewrite_cost_sec IS NOT NULL AND rewrite_cost_sec > 0
            """
            params: list[Any] = [table_name]
            if cluster_id is not None:
                q += " AND cluster_id = ?"
                params.append(cluster_id)
            q += " ORDER BY eval_window_end DESC LIMIT ?"
            params.append(int(limit))
            cur.execute(q, params)
            rows = cur.fetchall()
            out: list[float] = []
            for r in rows:
                v = r[0]
                if v is None:
                    continue
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out

    def get_rewrite_cost_normalizer_sec(
        self,
        *,
        table_name: str,
        cluster_id: Optional[str] = None,
        fallback_sec: float = 3600.0,
        limit: int = 50,
    ) -> float:
        xs = self.get_recent_rewrite_costs_sec(
            table_name=table_name, cluster_id=cluster_id, limit=limit
        )
        if not xs:
            return float(fallback_sec)
        try:
            return float(median(xs))
        except Exception:
            return float(fallback_sec)

    # -------------------------
    # Reporting helpers
    # -------------------------
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
        with self._connection() as conn:
            cur = conn.cursor()
            q = """
                SELECT le.*
                FROM layout_eval le
                WHERE le.table_name = ?
            """
            params: list[Any] = [table_name]
            if layout_id:
                q += " AND le.layout_id = ?"
                params.append(layout_id)
            if cluster_id is not None:
                q += " AND le.cluster_id = ?"
                params.append(cluster_id)
            if since is not None:
                q += " AND le.eval_window_end >= ?"
                params.append(_ts_param(since))
            if only_scored:
                q += " AND le.reward_score IS NOT NULL"
            q += " ORDER BY le.eval_window_end DESC"
            if limit is not None:
                q += " LIMIT ?"
                params.append(int(limit))
            cur.execute(q, params)
            return [dict(r) for r in cur.fetchall()]

    def get_layout_query_counts(
        self,
        *,
        table_name: str,
        cluster_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            q = """
                SELECT
                    cluster_id,
                    layout_id,
                    COUNT(1) AS query_count
                FROM query_log
                WHERE table_name = ?
            """
            params: list[Any] = [table_name]
            if cluster_id is not None:
                q += " AND cluster_id = ?"
                params.append(cluster_id)
            if since is not None:
                q += " AND ts >= ?"
                params.append(_ts_param(since))
            if until is not None:
                q += " AND ts <= ?"
                params.append(_ts_param(until))
            q += " GROUP BY cluster_id, layout_id ORDER BY query_count DESC"
            cur.execute(q, params)
            return [dict(r) for r in cur.fetchall()]

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
            cur = conn.cursor()
            cur.execute(
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
                    _now_ms(),
                    total_files,
                    0,
                ),
            )
            conn.commit()

    def claim_next_migration_job(self) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            conn.isolation_level = None
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(
                """
                SELECT * FROM migration_job
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                cur.execute("COMMIT")
                return None
            job_id = row["job_id"]
            cur.execute(
                """
                UPDATE migration_job
                SET status = 'running', started_at = ?
                WHERE job_id = ?
                """,
                (_now_ms(), job_id),
            )
            cur.execute("COMMIT")
            return dict(row)

    def update_migration_job_progress(
        self, job_id: str, processed_files: int
    ) -> None:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE migration_job SET processed_files = ? WHERE job_id = ?",
                (int(processed_files), job_id),
            )
            conn.commit()

    def complete_migration_job(self, job_id: str) -> None:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE migration_job SET status='completed', finished_at=? WHERE job_id=?",
                (_now_ms(), job_id),
            )
            conn.commit()

    def fail_migration_job(self, job_id: str, error: str) -> None:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE migration_job SET status='failed', finished_at=?, error=? WHERE job_id=?",
                (_now_ms(), error, job_id),
            )
            conn.commit()

    def get_migration_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM migration_job WHERE job_id = ?", (job_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None
