"""Layout explorer that routes traffic and manages exploration."""

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ..analyzer.workload_clusterer import WorkloadClusterer
from ..config import Config
from ..evaluator.reward import RewardCalculator
from ..layout.generator import LayoutGenerator
from ..layout.migrator import LayoutMigrator
from ..layout.spec import LayoutSpec
from ..query.context import extract_query_context_result
from ..query.executor import QueryExecutor
from ..storage.metadata import MetadataStore
from .bandit import MultiArmedBandit
from .contextual_bandit import ContextualBanditPolicy


class LayoutExplorer:
    """Manages layout exploration using multi-armed bandit."""

    def __init__(
        self,
        config: Config,
        metadata_store: MetadataStore,
        layout_generator: LayoutGenerator,
        layout_migrator: LayoutMigrator,
        reward_calculator: RewardCalculator,
        query_executor: QueryExecutor,
    ):
        """Initialize layout explorer."""
        self.config = config
        self.metadata_store = metadata_store
        self.layout_generator = layout_generator
        self.layout_migrator = layout_migrator
        self.reward_calculator = reward_calculator
        self.query_executor = query_executor

        # Contextual policy (uses global rewards + compatibility bonus)
        self.contextual_policy = ContextualBanditPolicy(
            metadata_store=self.metadata_store,
            exploration_rate=self.config.exploration_rate,
        )
        self.clusterer = WorkloadClusterer(
            num_clusters=self.config.num_clusters_per_table
        )

        # Bandit per table
        self.bandits: Dict[str, MultiArmedBandit] = {}

    def get_bandit(self, table_name: str) -> MultiArmedBandit:
        """Get or create bandit for a table."""
        if table_name not in self.bandits:
            self.bandits[table_name] = MultiArmedBandit(
                metadata_store=self.metadata_store,
                exploration_constant=2.0,
            )
        # Always reload evaluations to get latest data
        self.bandits[table_name].load_evaluations(table_name)
        return self.bandits[table_name]

    def select_layout_for_query(
        self,
        table_name: str,
        sql: Optional[str] = None,
    ) -> Optional[str]:
        """
        Select which layout to use for a query (routing decision).

        Uses exploration rate: 80% best, 20% exploration.

        Returns:
            Layout ID to use, or None if no layouts available
        """
        # Contextual routing if query is available
        if sql:
            ctx_res = extract_query_context_result(
                sql, dialect=self.config.sql_parser_dialect
            )
            if (
                ctx_res.parse_success
                and ctx_res.parse_confidence
                >= self.config.parse_confidence_threshold
            ):
                ctx = ctx_res.context
                cluster_id = self.clusterer.cluster_id_for_context_key(
                    ctx.key()
                )
                lid = self.contextual_policy.select_layout(
                    table_name=table_name,
                    context=ctx,
                    cluster_id=cluster_id,
                )
                if lid is not None:
                    return lid

        # Fallback: global bandit routing
        bandit = self.get_bandit(table_name)
        if not bandit.arms:
            return None
        if random.random() < self.config.exploration_rate:
            return bandit.select_arm(method="ucb1")
        best_arm = bandit.get_best_arm()
        return best_arm or bandit.select_arm(method="ucb1")

    def propose_new_layout(
        self,
        table_name: str,
        window_hours: Optional[int] = None,
    ) -> Optional[tuple[str, LayoutSpec]]:
        """
        Propose a new layout candidate based on workload analysis.

        Returns:
            Tuple of (layout_id, layout_spec) or None
        """
        # Generate a small set of workload-informed candidates
        candidates = self.layout_generator.generate_candidate_layouts(
            table_name,
            window_hours=window_hours or self.config.analysis_window_hours,
        )

        # Check candidates against existing layouts
        bandit = self.get_bandit(table_name)
        existing_layouts = self.metadata_store.get_all_layouts(table_name)

        import json

        def _normalize_partition_cols(
            spec_cols, derived_map: dict[str, str]
        ) -> Optional[list[str]]:
            if not spec_cols:
                return None
            out = []
            for c in spec_cols:
                out.append(derived_map.get(c, c))
            return out or None

        existing_specs: list[LayoutSpec] = []
        for existing in existing_layouts:
            partition_cols = json.loads(
                existing.get("partition_cols") or "null"
            )
            sort_cols = json.loads(existing.get("sort_cols") or "null")
            derived_map: dict[str, str] = {}
            notes = existing.get("notes")
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
            existing_specs.append(
                LayoutSpec(
                    partition_cols=_normalize_partition_cols(
                        partition_cols, derived_map
                    ),
                    sort_cols=sort_cols,
                )
            )

        layout_spec: Optional[LayoutSpec] = None
        for candidate in candidates:
            if all(
                self.layout_generator.compare_layouts(candidate, es)
                for es in existing_specs
            ):
                layout_spec = candidate
                break

        if layout_spec is None:
            return None

        # Generate new layout ID
        layout_id = self.layout_generator.generate_layout_id()

        # Add to bandit
        bandit.add_arm(layout_id, layout_spec)

        return (layout_id, layout_spec)

    def create_and_evaluate_layout(
        self,
        table_name: str,
        layout_id: str,
        layout_spec: LayoutSpec,
    ) -> Dict:
        """
        Create a new layout and evaluate it.

        Returns:
            Evaluation results dictionary
        """
        # Get source path
        source_path = self.layout_migrator.get_source_path(table_name)
        if not source_path:
            raise ValueError(f"No source path found for {table_name}")

        # Estimate rewrite cost
        rewrite_cost = self.layout_migrator.estimate_rewrite_cost(
            table_name,
            source_path,
        )

        # Migrate to new layout
        new_path = self.layout_migrator.migrate_table(
            table_name,
            source_path,
            layout_spec,
            layout_id,
        )

        # Register new layout in query executor
        self.query_executor.register_layout(
            table_name, str(new_path), layout_id
        )

        # Note: Evaluation compares queries BEFORE layout creation (baseline)
        # with queries AFTER layout creation (new layout).
        # Since the layout was just created, there may be no queries against it yet.
        # The evaluation will use queries that run after this point.

        # Get baseline layout
        baseline_layout = self.metadata_store.get_active_layout(table_name)
        baseline_layout_id = (
            baseline_layout["layout_id"] if baseline_layout else None
        )

        # Evaluate new layout
        # This will compare baseline (queries before layout creation)
        # with new layout (queries after creation, which may be empty initially)
        eval_result = self.reward_calculator.evaluate_layout(
            layout_id=layout_id,
            eval_window_hours=self.config.eval_window_hours,
            baseline_layout_id=baseline_layout_id,
            rewrite_cost_sec=rewrite_cost,
        )

        # Update bandit with reward
        bandit = self.get_bandit(table_name)
        bandit.add_arm(layout_id, layout_spec)
        bandit.update_arm(layout_id, eval_result["reward_score"])

        return eval_result

    def backfill_evaluations(
        self,
        table_name: str,
    ) -> Dict[str, object]:
        """
        Backfill evaluations for existing layouts that now have enough post-creation queries.

        This is the missing piece in the loop: new layouts are evaluated immediately (often with 0
        queries), but later accumulate traffic. Running `optimize` again should convert those into
        scored evaluations so the bandit can learn.
        """
        bandit = self.get_bandit(table_name)
        layouts = self.metadata_store.get_all_layouts(table_name)

        updated: List[str] = []
        skipped_insufficient: List[str] = []

        # Pre-check query counts here to avoid writing redundant NULL evaluations.
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=self.config.eval_window_hours)
        min_q = int(getattr(self.config, "min_queries_for_eval", 0) or 0)

        def _parse_created_at(value) -> datetime:
            # Keep parsing behavior consistent with RewardCalculator
            def _as_utc(dt: datetime) -> datetime:
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)

            if isinstance(value, str):
                try:
                    return _as_utc(
                        datetime.fromisoformat(value.replace("Z", "+00:00"))
                    )
                except ValueError:
                    try:
                        return _as_utc(
                            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                        )
                    except ValueError:
                        return datetime.now(timezone.utc)
            if isinstance(value, (int, float)):
                # epoch ms
                try:
                    return datetime.fromtimestamp(
                        float(value) / 1000.0, tz=timezone.utc
                    )
                except Exception:
                    return datetime.now(timezone.utc)
            if isinstance(value, datetime):
                return _as_utc(value)
            return datetime.now(timezone.utc)

        for layout in layouts:
            layout_id = layout["layout_id"]

            # Skip if latest evaluation is already scored
            latest = self.metadata_store.get_layout_evaluations(
                layout_id, limit=1
            )
            if latest and latest[0].get("reward_score") is not None:
                continue

            created_at = _parse_created_at(layout.get("created_at"))
            new_start_time = max(start_time, created_at)
            baseline_end_time = min(end_time, created_at)

            # Query counts in the same windows that RewardCalculator will use
            new_metrics = self.reward_calculator.metrics_calculator.calculate_layout_metrics(
                layout_id,
                new_start_time,
                end_time,
                table_name=table_name,
            )
            baseline_metrics = self.reward_calculator.metrics_calculator.calculate_layout_metrics(
                "initial",
                start_time,
                baseline_end_time,
                table_name=table_name,
            )

            if min_q > 0 and (
                int(new_metrics.get("queries_evaluated", 0) or 0) < min_q
                or int(baseline_metrics.get("queries_evaluated", 0) or 0)
                < min_q
            ):
                skipped_insufficient.append(layout_id)
                continue

            rewrite_cost = (
                latest[0].get("rewrite_cost_sec", 0.0) if latest else 0.0
            )
            scored = self.reward_calculator.evaluate_layout(
                layout_id=layout_id,
                eval_window_hours=self.config.eval_window_hours,
                baseline_layout_id=None,
                rewrite_cost_sec=rewrite_cost,
            )

            if scored.get("reward_score") is None:
                skipped_insufficient.append(layout_id)
                continue

            bandit.update_arm(layout_id, scored["reward_score"])
            updated.append(layout_id)

        # If we learned something new, consider activating the best layout
        best_arm = bandit.get_best_arm()
        active_layout = self.metadata_store.get_active_layout(table_name)
        if best_arm and (
            not active_layout or active_layout.get("layout_id") != best_arm
        ):
            self.metadata_store.activate_layout(best_arm, table_name)
            layout_info = self.metadata_store.get_active_layout(table_name)
            if layout_info:
                self.query_executor.register_layout(
                    table_name,
                    layout_info["layout_path"],
                    layout_info["layout_id"],
                )

        return {
            "updated_layouts": updated,
            "skipped_insufficient": skipped_insufficient,
            "best_arm": best_arm,
        }

    def optimize_table(
        self,
        table_name: str,
        force_new: bool = False,
    ) -> Dict:
        """
        Run one optimization cycle for a table.

        Steps:
        1. Analyze workload
        2. Propose new layout (if needed)
        3. Create and evaluate layout
        4. Update bandit

        Returns:
            Dictionary with optimization results
        """
        # First: backfill evaluations for existing layouts that now have enough data
        backfill_info = self.backfill_evaluations(table_name)

        # Check if we have any active layout
        active_layout = self.metadata_store.get_active_layout(table_name)

        # Propose new layout
        proposal = self.propose_new_layout(table_name)

        if not proposal and not force_new:
            # If no active layout exists, we should still create one
            if not active_layout:
                # Generate a layout even if similar one exists (we need an active one)
                layout_spec = self.layout_generator.generate_layout(table_name)
                layout_id = self.layout_generator.generate_layout_id()
            else:
                return {
                    "status": "no_new_layout",
                    "message": "No new layout proposed (may already exist)",
                    "backfill": backfill_info,
                }
        elif proposal:
            layout_id, layout_spec = proposal
        else:
            # Force new: generate a random variation
            layout_spec = self.layout_generator.generate_random_layout(
                table_name
            )
            layout_id = self.layout_generator.generate_layout_id()

        # Create and evaluate
        try:
            eval_result = self.create_and_evaluate_layout(
                table_name,
                layout_id,
                layout_spec,
            )

            # Activate layout if it's the best, or if no layout is currently active
            bandit = self.get_bandit(table_name)
            best_arm = bandit.get_best_arm()
            active_layout = self.metadata_store.get_active_layout(table_name)

            if best_arm == layout_id or not active_layout:
                # Activate the new layout (either it's best, or it's the first one)
                self.metadata_store.activate_layout(layout_id, table_name)
                # Re-register in query executor
                layout_info = self.metadata_store.get_active_layout(table_name)
                if layout_info:
                    self.query_executor.register_layout(
                        table_name,
                        layout_info["layout_path"],
                        layout_info["layout_id"],
                    )

            return {
                "status": "success",
                "layout_id": layout_id,
                "evaluation": eval_result,
                "best_layout": best_arm,
                "backfill": backfill_info,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "backfill": backfill_info,
            }

    def get_exploration_status(self, table_name: str) -> Dict:
        """Get current exploration status for a table."""
        bandit = self.get_bandit(table_name)
        active_layout = self.metadata_store.get_active_layout(table_name)

        return {
            "table_name": table_name,
            "active_layout": active_layout["layout_id"]
            if active_layout
            else None,
            "num_arms": len(bandit.arms),
            "arm_stats": bandit.get_arm_stats(),
            "best_arm": bandit.get_best_arm(),
        }
