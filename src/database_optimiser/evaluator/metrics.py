"""Performance metrics calculation for layout evaluation."""

import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..storage.metadata import MetadataStore


class MetricsCalculator:
    """Calculates performance metrics from query logs."""

    def __init__(self, metadata_store: MetadataStore):
        """Initialize metrics calculator."""
        self.metadata_store = metadata_store

    def calculate_layout_metrics(
        self,
        layout_id: str,
        start_time: datetime,
        end_time: datetime,
        table_name: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for a layout over a time window.

        Args:
            layout_id: Layout ID to filter queries (or "initial" for queries before any layout)
            start_time: Start of time window
            end_time: End of time window
            table_name: Optional table name to filter queries

        Returns:
            Dictionary with metrics: avg_latency_ms, p95_latency_ms, p99_latency_ms,
            avg_rows_scanned, queries_evaluated
        """
        query_logs = self.metadata_store.get_query_logs(
            table_name=table_name,
            start_time=start_time,
            end_time=end_time,
            layout_id=layout_id if layout_id != "initial" else None,
            layout_id_is_null=(layout_id == "initial"),
            cluster_id=cluster_id,
        )

        if not query_logs:
            return {
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "latency_stddev_ms": 0.0,
                "latency_cv": 0.0,
                "avg_rows_scanned": 0.0,
                "queries_evaluated": 0,
            }

        latencies = [
            log["runtime_ms"] for log in query_logs if log["runtime_ms"]
        ]
        rows_scanned = [
            log["rows_scanned"]
            for log in query_logs
            if log["rows_scanned"] is not None
        ]

        if not latencies:
            return {
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "latency_stddev_ms": 0.0,
                "latency_cv": 0.0,
                "avg_rows_scanned": statistics.mean(rows_scanned)
                if rows_scanned
                else 0.0,
                "queries_evaluated": len(query_logs),
            }

        latencies.sort()
        avg = statistics.mean(latencies)
        stddev = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        cv = (stddev / avg) if avg > 0 else 0.0

        return {
            "avg_latency_ms": avg,
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)]
            if latencies
            else 0.0,
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)]
            if latencies
            else 0.0,
            "latency_stddev_ms": stddev,
            "latency_cv": cv,
            "avg_rows_scanned": statistics.mean(rows_scanned)
            if rows_scanned
            else 0.0,
            "queries_evaluated": len(query_logs),
        }

    def get_latency_samples(
        self,
        layout_id: str,
        start_time: datetime,
        end_time: datetime,
        table_name: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> List[float]:
        """Return raw latency samples for guardrails (bootstrap, outlier handling)."""
        query_logs = self.metadata_store.get_query_logs(
            table_name=table_name,
            start_time=start_time,
            end_time=end_time,
            layout_id=layout_id if layout_id != "initial" else None,
            layout_id_is_null=(layout_id == "initial"),
            cluster_id=cluster_id,
        )

        return [
            float(log["runtime_ms"])
            for log in query_logs
            if log.get("runtime_ms")
        ]

    def compare_layouts(
        self,
        baseline_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compare two layouts' metrics.

        Returns:
            Dictionary with improvement percentages
        """
        baseline_latency = baseline_metrics.get("avg_latency_ms", 0.0)
        new_latency = new_metrics.get("avg_latency_ms", 0.0)

        if baseline_latency == 0:
            return {
                "latency_improvement": 0.0,
                "p95_improvement": 0.0,
                "rows_scanned_improvement": 0.0,
            }

        latency_improvement = (
            baseline_latency - new_latency
        ) / baseline_latency

        baseline_p95 = baseline_metrics.get("p95_latency_ms", 0.0)
        new_p95 = new_metrics.get("p95_latency_ms", 0.0)
        p95_improvement = (
            (baseline_p95 - new_p95) / baseline_p95
            if baseline_p95 > 0
            else 0.0
        )

        baseline_rows = baseline_metrics.get("avg_rows_scanned", 0.0)
        new_rows = new_metrics.get("avg_rows_scanned", 0.0)
        rows_improvement = (
            (baseline_rows - new_rows) / baseline_rows
            if baseline_rows > 0
            else 0.0
        )

        return {
            "latency_improvement": latency_improvement,
            "p95_improvement": p95_improvement,
            "rows_scanned_improvement": rows_improvement,
        }
