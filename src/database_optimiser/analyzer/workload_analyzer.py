"""Workload analyzer that aggregates query patterns and computes column statistics."""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..storage.metadata import MetadataStore
from .column_stats import ColumnStats


class WorkloadAnalyzer:
    """Analyzes query workload to extract patterns and column statistics."""

    def __init__(self, metadata_store: MetadataStore):
        """Initialize workload analyzer."""
        self.metadata_store = metadata_store

    def analyze_table(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, ColumnStats]:
        """
        Analyze workload for a specific table.

        Returns:
            Dictionary mapping column_name -> ColumnStats
        """
        # Get query logs for the table
        end_time = datetime.utcnow()
        start_time = (
            end_time - timedelta(hours=window_hours) if window_hours else None
        )

        query_logs = self.metadata_store.get_query_logs(
            table_name=table_name,
            start_time=start_time,
            end_time=end_time,
            cluster_id=cluster_id,
        )

        if not query_logs:
            return {}

        # Aggregate statistics per column
        column_stats: Dict[str, ColumnStats] = {}
        total_queries = len(query_logs)

        # Counters for each column
        filter_counts: defaultdict[str, int] = defaultdict(int)
        join_counts: defaultdict[str, int] = defaultdict(int)
        groupby_counts: defaultdict[str, int] = defaultdict(int)
        orderby_counts: defaultdict[str, int] = defaultdict(int)
        selectivity_sums: defaultdict[str, float] = defaultdict(float)
        selectivity_counts: defaultdict[str, int] = defaultdict(int)

        for log in query_logs:
            # Parse JSON fields
            columns_used = json.loads(log["columns_used"] or "[]")
            predicates = json.loads(log["predicates"] or "[]")
            joins = json.loads(log["joins"] or "[]")
            group_by_cols = json.loads(log["group_by_cols"] or "[]")
            order_by_cols = json.loads(log["order_by_cols"] or "[]")

            # Track which columns are used
            columns_in_query = set(columns_used)

            # Count filters
            for pred in predicates:
                col = pred.get("col")
                if col:
                    filter_counts[col] += 1
                    columns_in_query.add(col)

                    # Estimate selectivity (heuristic: equality = high selectivity, range = medium)
                    op = pred.get("op", "").upper()
                    if op in ["=", "IN"]:
                        selectivity = 0.1  # High selectivity (equality)
                    elif op in [">", "<", ">=", "<=", "BETWEEN"]:
                        selectivity = 0.3  # Medium selectivity (range)
                    else:
                        selectivity = 0.5  # Low selectivity (LIKE, etc.)

                    selectivity_sums[col] += selectivity
                    selectivity_counts[col] += 1

            # Count joins
            for join in joins:
                left_col = join.get("left")
                right_col = (
                    join.get("right", "").split(".")[-1]
                    if "." in join.get("right", "")
                    else join.get("right")
                )

                if left_col:
                    join_counts[left_col] += 1
                    columns_in_query.add(left_col)
                if right_col:
                    join_counts[right_col] += 1

            # Count group by
            for col in group_by_cols:
                groupby_counts[col] += 1
                columns_in_query.add(col)

            # Count order by
            for col in order_by_cols:
                orderby_counts[col] += 1
                columns_in_query.add(col)

        # Create ColumnStats objects
        all_columns = (
            set(filter_counts.keys())
            | set(join_counts.keys())
            | set(groupby_counts.keys())
            | set(orderby_counts.keys())
        )

        for col in all_columns:
            avg_selectivity = (
                selectivity_sums[col] / selectivity_counts[col]
                if selectivity_counts[col] > 0
                else 1.0
            )

            column_stats[col] = ColumnStats(
                table_name=table_name,
                column_name=col,
                filter_freq=filter_counts[col],
                join_freq=join_counts[col],
                groupby_freq=groupby_counts[col],
                orderby_freq=orderby_counts[col],
                avg_selectivity=avg_selectivity,
                total_queries=total_queries,
            )

        return column_stats

    def get_partition_candidates(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
        cluster_id: Optional[str] = None,
    ) -> List[tuple[str, float]]:
        """
        Get candidate columns for partitioning, ranked by score.

        Returns:
            List of (column_name, score) tuples, sorted by score descending
        """
        column_stats = self.analyze_table(
            table_name, window_hours, cluster_id=cluster_id
        )

        candidates = []
        for col, stats in column_stats.items():
            # Score based on filter frequency and selectivity
            # Higher filter_freq + lower selectivity = better partition candidate
            score = stats.filter_freq * (1.0 - stats.avg_selectivity)

            # Bonus for time-like columns (heuristic: contains "date" or "time")
            if "date" in col.lower() or "time" in col.lower():
                score *= 1.5

            candidates.append((col, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def get_sort_candidates(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
        max_cols: int = 3,
        cluster_id: Optional[str] = None,
    ) -> List[str]:
        """
        Get candidate columns for sorting/clustering, ranked by importance.

        Returns:
            List of column names in order of importance
        """
        column_stats = self.analyze_table(
            table_name, window_hours, cluster_id=cluster_id
        )

        candidates = []
        for col, stats in column_stats.items():
            # Score based on filter + groupby + orderby frequency
            score = (
                stats.filter_freq * 2.0  # Filters are most important
                + stats.groupby_freq * 1.5  # Group by is important
                + stats.orderby_freq * 1.0  # Order by is less important
            )

            candidates.append((col, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top N columns
        return [col for col, _ in candidates[:max_cols]]

    def get_workload_summary(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get a summary of the workload for a table."""
        query_logs = self.metadata_store.get_query_logs(
            table_name=table_name,
            start_time=datetime.utcnow() - timedelta(hours=window_hours)
            if window_hours
            else None,
        )

        if not query_logs:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            }

        latencies = [log["runtime_ms"] for log in query_logs]
        latencies.sort()

        return {
            "total_queries": len(query_logs),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)]
            if latencies
            else 0.0,
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)]
            if latencies
            else 0.0,
        }
