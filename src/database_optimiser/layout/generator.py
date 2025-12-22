"""Heuristic layout generator based on workload analysis."""

import itertools
import random
import uuid
from typing import List, Optional

from ..analyzer.workload_analyzer import WorkloadAnalyzer
from ..config import Config
from .spec import LayoutSpec


class LayoutGenerator:
    """Generates layout specifications based on workload analysis."""

    def __init__(
        self,
        workload_analyzer: WorkloadAnalyzer,
        config: Config,
    ):
        """Initialize layout generator."""
        self.workload_analyzer = workload_analyzer
        self.config = config

    def generate_layout(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
    ) -> LayoutSpec:
        """
        Generate a layout specification for a table based on workload.

        Heuristic algorithm:
        1. Choose partition column: time column or highest filter_freq with good selectivity
        2. Choose sort columns: top filter + groupby columns
        """
        # Keep the old behavior (best single layout) by selecting the first candidate.
        candidates = self.generate_candidate_layouts(
            table_name,
            window_hours=window_hours,
        )
        if candidates:
            return candidates[0]
        return LayoutSpec(
            partition_cols=None,
            sort_cols=None,
            file_size_mb=self.config.default_file_size_mb,
        )

    def generate_random_layout(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
    ) -> LayoutSpec:
        """Generate a random (but workload-informed) layout for forced exploration."""
        candidates = self.generate_candidate_layouts(
            table_name,
            window_hours=window_hours,
        )
        if candidates:
            return random.choice(candidates)
        return LayoutSpec(
            partition_cols=None,
            sort_cols=None,
            file_size_mb=self.config.default_file_size_mb,
        )

    def generate_candidate_layouts(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
        cluster_id: Optional[str] = None,
        max_partition_candidates: int = 3,
        max_sort_candidates: int = 3,
    ) -> List[LayoutSpec]:
        """
        Generate a small set of candidate layouts.

        This helps avoid the system repeatedly proposing the same deterministic layout and enables
        the explorer/bandit to evaluate multiple plausible physical designs.
        """
        partition_candidates = self.workload_analyzer.get_partition_candidates(
            table_name,
            window_hours=window_hours,
            cluster_id=cluster_id,
        )
        sort_candidates = self.workload_analyzer.get_sort_candidates(
            table_name,
            window_hours=window_hours,
            max_cols=max_sort_candidates,
            cluster_id=cluster_id,
        )

        # Phase 4 (beam-ish search): build multi-column partition/sort combinations and rank them.
        p_scored = [
            (c, float(s))
            for (c, s) in (partition_candidates or [])
            if float(s) > 0
        ]
        p_cols = [
            c for (c, _s) in p_scored[: self.config.candidate_top_k_partition]
        ]
        p_score_map = {c: s for (c, s) in p_scored}

        s_cols = (sort_candidates or [])[: self.config.candidate_top_k_sort]
        s_rank_map = {c: i for i, c in enumerate(s_cols)}

        # Partition combos: () plus 1..max_partition_cols combinations (order doesn't matter)
        p_combos: list[tuple[str, ...]] = [()]
        for k in range(
            1, max(1, self.config.candidate_max_partition_cols) + 1
        ):
            p_combos.extend(list(itertools.combinations(p_cols, k)))

        # Sort combos: () plus 1..max_sort_cols permutations that preserve rank order
        # (we generate combinations and then sort by rank to keep it deterministic).
        s_combos: list[tuple[str, ...]] = [()]
        for k in range(1, max(1, self.config.candidate_max_sort_cols) + 1):
            for comb in itertools.combinations(s_cols, k):
                s_combos.append(
                    tuple(
                        sorted(comb, key=lambda c: s_rank_map.get(c, 10_000))
                    )
                )

        def score(p: tuple[str, ...], s: tuple[str, ...]) -> float:
            # Prefer higher partition selectivity score; prefer earlier-ranked sort columns.
            part = sum(p_score_map.get(c, 0.0) for c in p)
            sort = sum(1.0 / (1.0 + s_rank_map.get(c, 10_000)) for c in s)
            return 2.0 * part + 0.25 * sort

        # Build, normalize, dedupe, then take top beam_width.
        candidates: list[tuple[float, LayoutSpec]] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

        for p in p_combos:
            for s in s_combos:
                # Exclude partition cols from sort cols
                if p:
                    s2 = tuple(c for c in s if c not in p)
                else:
                    s2 = s
                p_norm = list(p) if p else None
                s_norm = list(s2) if s2 else None
                key = (tuple(p_norm or ()), tuple(s_norm or ()))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    (
                        score(p, s2),
                        LayoutSpec(
                            partition_cols=p_norm,
                            sort_cols=s_norm,
                            file_size_mb=self.config.default_file_size_mb,
                        ),
                    )
                )

        candidates.sort(key=lambda t: t[0], reverse=True)
        return [
            spec
            for (_sc, spec) in candidates[: self.config.candidate_beam_width]
        ]

    def generate_layout_id(self) -> str:
        """Generate a unique layout ID."""
        return f"layout_{uuid.uuid4().hex[:8]}"

    def compare_layouts(
        self,
        layout1: LayoutSpec,
        layout2: LayoutSpec,
    ) -> bool:
        """
        Compare two layouts to see if they're different.

        Returns:
            True if layouts are different, False if same
        """
        return (
            layout1.partition_cols != layout2.partition_cols
            or layout1.sort_cols != layout2.sort_cols
        )
