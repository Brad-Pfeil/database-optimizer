"""Contextual bandit policy for routing queries to layouts.

This is a lightweight contextual policy that combines:
- global mean reward per layout (from stored evaluations)
- a context/layout compatibility bonus based on query features vs layout keys

It enables per-query routing decisions without requiring per-query reward persistence.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Optional

from ..layout.spec import LayoutSpec
from ..query.context import QueryContext
from ..storage.metadata import MetadataStore


@dataclass
class ContextualBanditPolicy:
    metadata_store: MetadataStore
    exploration_rate: float = 0.2
    rng: random.Random = random.Random(0)

    def _layout_specs(
        self, table_name: str, cluster_id: Optional[str] = None
    ) -> Dict[str, LayoutSpec]:
        if cluster_id is None:
            layouts = self.metadata_store.get_all_layouts(table_name)
        else:
            # Prefer cluster-scoped layouts; fall back to global if none.
            layouts = self.metadata_store.get_all_layouts(
                table_name, cluster_id=cluster_id
            )
            if not layouts:
                layouts = self.metadata_store.get_all_layouts(table_name)
        out: Dict[str, LayoutSpec] = {}
        for layout in layouts:
            partition_cols = json.loads(layout.get("partition_cols") or "null")
            sort_cols = json.loads(layout.get("sort_cols") or "null")
            out[layout["layout_id"]] = LayoutSpec(
                partition_cols=partition_cols, sort_cols=sort_cols
            )
        return out

    def _mean_reward(self, layout_id: str) -> float:
        evals = self.metadata_store.get_layout_evaluations(layout_id)
        raw_rewards = [e.get("reward_score") for e in evals]
        rewards: list[float] = [float(r) for r in raw_rewards if r is not None]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)

    def _compatibility_bonus(
        self, ctx: QueryContext, spec: LayoutSpec
    ) -> float:
        bonus = 0.0
        # Filters benefit from partition/sort matches
        for c in ctx.filter_cols:
            if spec.partition_cols and c in spec.partition_cols:
                bonus += 0.05
            if spec.sort_cols and c in spec.sort_cols:
                bonus += 0.02
        # Group/order benefit from sort matches
        for c in ctx.group_by_cols | ctx.order_by_cols:
            if spec.sort_cols and c in spec.sort_cols:
                bonus += 0.01
        return bonus

    def select_layout(
        self,
        *,
        table_name: str,
        context: QueryContext,
        cluster_id: Optional[str] = None,
    ) -> Optional[str]:
        specs = self._layout_specs(table_name, cluster_id=cluster_id)
        if not specs:
            return None

        layout_ids = list(specs.keys())

        # Exploration: random layout
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(layout_ids)

        # Exploitation: max(global_reward + compatibility bonus)
        best_id = None
        best_score = float("-inf")
        for lid, spec in specs.items():
            score = self._mean_reward(lid) + self._compatibility_bonus(
                context, spec
            )
            if score > best_score:
                best_score = score
                best_id = lid
        return best_id
