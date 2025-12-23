"""Contextual bandit policy for routing queries to layouts.

This is a lightweight contextual policy that combines:
- global mean reward per layout (from stored evaluations)
- a context/layout compatibility bonus based on query features vs layout keys

It enables per-query routing decisions without requiring per-query reward persistence.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

from ..layout.spec import LayoutSpec
from ..query.context import QueryContext
from ..storage.metadata import MetadataStore


@dataclass(frozen=True)
class _LayoutInfo:
    layout_id: str
    spec: LayoutSpec
    cluster_id: Optional[str]
    derived_partition_cols: dict[str, str]

    @property
    def logical_partition_cols(self) -> set[str]:
        # Treat the original columns referenced by derived partition keys as logically partitioned.
        return set(self.derived_partition_cols.values())


@dataclass
class ContextualBanditPolicy:
    metadata_store: MetadataStore
    exploration_rate: float = 0.2
    rng: random.Random = random.Random(0)
    stats_cache_ttl_sec: float = 5.0
    spec_cache_ttl_sec: float = 5.0
    cluster_layout_bonus: float = 0.0

    # Cache: (table_name, cluster_id) -> (monotonic_ts, stats)
    _reward_stats_cache: dict[
        tuple[str, Optional[str]], tuple[float, dict]
    ] = None  # type: ignore[assignment]
    # Cache: (table_name, cluster_id) -> (monotonic_ts, layout_infos)
    _spec_cache: dict[
        tuple[str, Optional[str]], tuple[float, dict[str, _LayoutInfo]]
    ] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._reward_stats_cache is None:
            self._reward_stats_cache = {}
        if self._spec_cache is None:
            self._spec_cache = {}

    def _layout_infos(
        self, *, table_name: str, cluster_id: Optional[str]
    ) -> Dict[str, _LayoutInfo]:
        key = (table_name, cluster_id)
        now = time.monotonic()
        cached = self._spec_cache.get(key)
        if cached is not None:
            ts, infos = cached
            if (now - ts) <= self.spec_cache_ttl_sec:
                return infos

        # Candidate set = (cluster layouts ∪ global layouts) if cluster_id is provided.
        all_layouts = self.metadata_store.get_all_layouts(table_name)
        if cluster_id is None:
            layouts = all_layouts
        else:
            layouts = [
                layout
                for layout in all_layouts
                if layout.get("cluster_id") in (None, cluster_id)
            ]

        infos: dict[str, _LayoutInfo] = {}
        for layout in layouts:
            partition_cols = json.loads(layout.get("partition_cols") or "null")
            sort_cols = json.loads(layout.get("sort_cols") or "null")
            derived_map: dict[str, str] = {}
            notes = layout.get("notes")
            if notes:
                try:
                    notes_obj = json.loads(notes)
                    if isinstance(notes_obj, dict) and isinstance(
                        notes_obj.get("derived_partition_cols"), dict
                    ):
                        derived_map = {
                            str(k): str(v)
                            for k, v in notes_obj[
                                "derived_partition_cols"
                            ].items()
                        }
                except Exception:
                    derived_map = {}
            lid = str(layout["layout_id"])
            infos[lid] = _LayoutInfo(
                layout_id=lid,
                spec=LayoutSpec(
                    partition_cols=partition_cols, sort_cols=sort_cols
                ),
                cluster_id=layout.get("cluster_id"),
                derived_partition_cols=derived_map,
            )

        self._spec_cache[key] = (now, infos)
        return infos

    def _reward_stats(
        self, *, table_name: str, cluster_id: Optional[str]
    ) -> Dict[str, Dict[str, object]]:
        key = (table_name, cluster_id)
        now = time.monotonic()
        cached = self._reward_stats_cache.get(key)
        if cached is not None:
            ts, stats = cached
            if (now - ts) <= self.stats_cache_ttl_sec:
                return stats

        stats = self.metadata_store.get_reward_stats_for_table(
            table_name=table_name, cluster_id=cluster_id
        )
        self._reward_stats_cache[key] = (now, stats)
        return stats

    def _compatibility_bonus(
        self, ctx: QueryContext, info: _LayoutInfo
    ) -> float:
        bonus = 0.0
        # Filters benefit from partition/sort matches
        for c in ctx.filter_cols:
            if info.spec.partition_cols and c in info.spec.partition_cols:
                bonus += 0.05
            # Derived partition keys: treat logical original column as compatible.
            if c in info.logical_partition_cols:
                bonus += 0.05
            if info.spec.sort_cols and c in info.spec.sort_cols:
                bonus += 0.02
        # Group/order benefit from sort matches
        for c in ctx.group_by_cols | ctx.order_by_cols:
            if info.spec.sort_cols and c in info.spec.sort_cols:
                bonus += 0.01
        return bonus

    def select_layout(
        self,
        *,
        table_name: str,
        context: QueryContext,
        cluster_id: Optional[str] = None,
    ) -> Optional[str]:
        infos = self._layout_infos(
            table_name=table_name, cluster_id=cluster_id
        )
        if not infos:
            return None

        layout_ids = list(infos.keys())

        # Exploration: random layout
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(layout_ids)

        # Fetch reward stats once (cached) rather than per-layout DB queries.
        stats = self._reward_stats(
            table_name=table_name, cluster_id=cluster_id
        )

        # Exploitation: max(global_reward + compatibility bonus)
        best_id = None
        best_score = float("-inf")

        prior_n = 5.0  # shrinkage strength (prevents 1-sample domination)
        uncertainty_c = 0.02  # small optimism bonus for low-n layouts

        def _as_float(v: object, *, default: float = 0.0) -> float:
            if v is None:
                return default
            if isinstance(v, bool):
                return default
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    return default
            return default

        for lid, info in infos.items():
            s = stats.get(lid, {})
            n = _as_float(s.get("n"), default=0.0)
            mean = _as_float(s.get("mean_reward"), default=0.0)

            shrunk_mean = mean * (n / (n + prior_n)) if n >= 0 else 0.0
            uncertainty_bonus = uncertainty_c / ((n + 1.0) ** 0.5)

            score = (
                shrunk_mean
                + uncertainty_bonus
                + self._compatibility_bonus(context, info)
            )
            # Slight bias toward cluster-scoped layouts (optional).
            if cluster_id is not None and info.cluster_id == cluster_id:
                score += float(self.cluster_layout_bonus or 0.0)
            if score > best_score:
                best_score = score
                best_id = lid
        return best_id
