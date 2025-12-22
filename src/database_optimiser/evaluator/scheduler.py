"""Scheduled evaluation / backfill loop (Phase 5).

This module evaluates candidate layouts on a rolling time window, rather than only when `optimize`
is invoked. It is dataset-agnostic: it relies only on query telemetry stored in metadata.db.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..config import Config
from ..storage.metadata import MetadataStore
from .reward import RewardCalculator


@dataclass
class EvaluationScheduler:
    config: Config
    metadata_store: MetadataStore
    reward_calculator: RewardCalculator

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _count_queries_in_window(
        self,
        *,
        table_name: str,
        layout_id: Optional[str],
        cluster_id: Optional[str],
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        return self.metadata_store.count_query_logs(
            table_name=table_name,
            layout_id=layout_id,
            cluster_id=cluster_id,
            start_time=start_time,
            end_time=end_time,
        )

    def evaluate_table(
        self,
        *,
        table_name: str,
        cluster_id: Optional[str] = None,
        eval_window_hours: Optional[int] = None,
        min_new_queries: Optional[int] = None,
    ) -> int:
        """Evaluate all known layouts for a table against the active baseline on a rolling window.

        Returns number of evaluations recorded.
        """
        end_time = self._now()
        window_hours = eval_window_hours or self.config.eval_window_hours
        start_time = end_time - timedelta(hours=window_hours)
        min_new = (
            min_new_queries
            if min_new_queries is not None
            else self.config.eval_scheduler_min_new_queries
        )

        active = self.metadata_store.get_active_layout(table_name)
        baseline_layout_id = active["layout_id"] if active else None

        layouts = (
            self.metadata_store.get_all_layouts(
                table_name, cluster_id=cluster_id
            )
            if cluster_id
            else self.metadata_store.get_all_layouts(table_name)
        )
        n = 0
        for layout in layouts:
            lid = layout["layout_id"]
            if baseline_layout_id and lid == baseline_layout_id:
                continue

            latest = self.metadata_store.get_latest_evaluation(lid)
            last_end = latest["eval_window_end"] if latest else None
            if last_end is not None:
                # Ensure we make progress: only evaluate if enough new queries since last eval end.
                since = last_end
                new_q = self._count_queries_in_window(
                    table_name=table_name,
                    layout_id=lid,
                    cluster_id=cluster_id or layout.get("cluster_id"),
                    start_time=since,
                    end_time=end_time,
                )
                if new_q < min_new:
                    continue

            self.reward_calculator.evaluate_layout_window(
                layout_id=lid,
                baseline_layout_id=baseline_layout_id,
                start_time=start_time,
                end_time=end_time,
                rewrite_cost_sec=0.0,
                cluster_id=cluster_id,
            )
            n += 1
        return n
