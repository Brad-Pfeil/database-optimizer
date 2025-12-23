"""Reward function for layout evaluation."""

import random
import statistics
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional

from ..config import Config
from ..storage.metadata import MetadataStore
from .metrics import MetricsCalculator


class RewardCalculator:
    """Calculates reward scores for layout evaluations."""

    def __init__(
        self,
        metrics_calculator: MetricsCalculator,
        config: Config,
        metadata_store: MetadataStore,
    ):
        """Initialize reward calculator."""
        self.metrics_calculator = metrics_calculator
        self.config = config
        self.metadata_store = metadata_store

    def calculate_reward(
        self,
        layout_id: str,
        baseline_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any],
        rewrite_cost_sec: float,
    ) -> float:
        """
        Calculate reward for a layout change.

        Formula: reward = α * latency_improvement - β * rewrite_cost

        Args:
            layout_id: Layout identifier
            baseline_metrics: Metrics from baseline layout
            new_metrics: Metrics from new layout
            rewrite_cost_sec: Time taken to rewrite (in seconds)

        Returns:
            Reward score (higher is better)
        """
        breakdown = self.calculate_reward_breakdown(
            baseline_metrics=baseline_metrics,
            new_metrics=new_metrics,
            rewrite_cost_sec=rewrite_cost_sec,
        )
        return breakdown["reward"]

    def calculate_reward_breakdown(
        self,
        baseline_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any],
        rewrite_cost_sec: float,
        *,
        table_name: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate reward and expose the intermediate components for observability.

        Returns:
            Dictionary with latency_improvement, normalized_rewrite_cost, reward, and terms.
        """

        def _improvement(b: float, n: float) -> float:
            if b <= 0:
                return 0.0
            return (b - n) / b

        baseline_mean = float(
            baseline_metrics.get("avg_latency_ms", 0.0) or 0.0
        )
        new_mean = float(new_metrics.get("avg_latency_ms", 0.0) or 0.0)
        baseline_p95 = float(
            baseline_metrics.get("p95_latency_ms", 0.0) or 0.0
        )
        new_p95 = float(new_metrics.get("p95_latency_ms", 0.0) or 0.0)
        baseline_p99 = float(
            baseline_metrics.get("p99_latency_ms", 0.0) or 0.0
        )
        new_p99 = float(new_metrics.get("p99_latency_ms", 0.0) or 0.0)

        baseline_rows = float(
            baseline_metrics.get("avg_rows_scanned", 0.0) or 0.0
        )
        new_rows = float(new_metrics.get("avg_rows_scanned", 0.0) or 0.0)

        baseline_cv = float(baseline_metrics.get("latency_cv", 0.0) or 0.0)
        new_cv = float(new_metrics.get("latency_cv", 0.0) or 0.0)

        mean_impr = _improvement(baseline_mean, new_mean)
        p95_impr = _improvement(baseline_p95, new_p95)
        p99_impr = _improvement(baseline_p99, new_p99)
        rows_impr = _improvement(baseline_rows, new_rows)
        var_impr = _improvement(
            baseline_cv, new_cv
        )  # treat lower CV as better

        # Normalize rewrite cost (convert to a 0-1 scale)
        normalizer = float(
            getattr(self.config, "rewrite_cost_normalizer_sec_default", 3600.0)
            or 3600.0
        )
        mode = str(
            getattr(self.config, "rewrite_cost_normalizer_mode", "fixed")
            or "fixed"
        ).lower()
        if (
            mode == "historical_median"
            and table_name is not None
            and normalizer > 0
        ):
            normalizer = self.metadata_store.get_rewrite_cost_normalizer_sec(
                table_name=table_name,
                cluster_id=cluster_id,
                fallback_sec=normalizer,
                limit=int(
                    getattr(
                        self.config, "rewrite_cost_normalizer_history_n", 50
                    )
                    or 50
                ),
            )
        if normalizer <= 0:
            normalizer = 3600.0

        normalized_rewrite_cost = min(
            (rewrite_cost_sec or 0.0) / normalizer, 1.0
        )

        perf_term = (
            self.config.reward_w_mean * mean_impr
            + self.config.reward_w_p95 * p95_impr
            + self.config.reward_w_p99 * p99_impr
            + self.config.reward_w_rows_scanned * rows_impr
            + self.config.reward_w_variance * var_impr
        )
        rewrite_term = self.config.beta * normalized_rewrite_cost
        reward = perf_term - rewrite_term

        return {
            "latency_improvement": mean_impr,
            "mean_improvement": mean_impr,
            "p95_improvement": p95_impr,
            "p99_improvement": p99_impr,
            "rows_scanned_improvement": rows_impr,
            "cv_improvement": var_impr,
            "normalized_rewrite_cost": normalized_rewrite_cost,
            "performance_term": perf_term,
            "rewrite_term": rewrite_term,
            "reward": reward,
        }

    def _winsorize(
        self,
        xs: List[float],
        pctl: float,
    ) -> List[float]:
        if not xs:
            return []
        xs2 = sorted(xs)
        idx = int(len(xs2) * pctl)
        idx = min(max(idx, 0), len(xs2) - 1)
        cap = xs2[idx]
        return [min(x, cap) for x in xs2]

    def _bootstrap_ci_lower_bound(
        self,
        baseline: List[float],
        new: List[float],
        iters: int,
        alpha: float,
        seed: int,
    ) -> float:
        """
        Bootstrap a lower bound for mean latency improvement:
        improvement = (mean(b) - mean(n)) / mean(b)
        """
        if not baseline or not new:
            return 0.0
        b_mean = statistics.mean(baseline)
        if b_mean <= 0:
            return 0.0
        rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
        samples: List[float] = []
        for _ in range(max(1, iters)):
            b = [rng.choice(baseline) for _ in range(len(baseline))]
            n = [rng.choice(new) for _ in range(len(new))]
            b_m = statistics.mean(b)
            n_m = statistics.mean(n)
            if b_m <= 0:
                continue
            samples.append((b_m - n_m) / b_m)
        if not samples:
            return 0.0
        samples.sort()
        k = int(len(samples) * alpha)
        k = min(max(k, 0), len(samples) - 1)
        return float(samples[k])

    def _bootstrap_seed(
        self,
        *,
        layout_id: str,
        baseline_layout_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        s = (
            f"{layout_id}|{baseline_layout_id}|{start_time.isoformat()}|{end_time.isoformat()}"
        ).encode("utf-8")
        return int.from_bytes(sha256(s).digest()[:8], "big")

    def evaluate_layout(
        self,
        layout_id: str,
        eval_window_hours: int,
        baseline_layout_id: Optional[str] = None,
        rewrite_cost_sec: float = 0.0,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a layout and calculate reward.

        Args:
            layout_id: Layout to evaluate
            eval_window_hours: Hours of queries to evaluate
            baseline_layout_id: Baseline layout for comparison (if None, uses previous active)
            rewrite_cost_sec: Cost of rewriting to this layout

        Returns:
            Dictionary with evaluation results including reward_score
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=eval_window_hours)

        # Get layout info to determine table name
        layout_info = self.metadata_store.get_layout(layout_id)
        if not layout_info:
            raise ValueError(f"Layout {layout_id} not found")

        table_name = layout_info["table_name"]
        if cluster_id is None:
            cluster_id = layout_info.get("cluster_id")
        # Baseline resolution: if not explicitly provided, compare vs active layout if present
        # (else vs initial). Always compare on the exact same rolling window.
        baseline_used: Optional[str] = baseline_layout_id
        if baseline_used is None:
            active = self.metadata_store.get_active_layout(table_name)
            if active and active.get("layout_id") != layout_id:
                baseline_used = str(active["layout_id"])
            else:
                baseline_used = "initial"
        elif baseline_used == layout_id:
            baseline_used = "initial"

        baseline_metrics = self.metrics_calculator.calculate_layout_metrics(
            baseline_used,
            start_time,
            end_time,
            table_name=table_name,
            cluster_id=cluster_id,
        )

        new_metrics = self.metrics_calculator.calculate_layout_metrics(
            layout_id,
            start_time,
            end_time,
            table_name=table_name,
            cluster_id=cluster_id,
        )

        # Gate evaluation: don't score until we have enough data.
        min_q = int(getattr(self.config, "min_queries_for_eval", 0) or 0)
        new_q = int(new_metrics.get("queries_evaluated", 0) or 0)
        baseline_q = int(baseline_metrics.get("queries_evaluated", 0) or 0)

        reward_score: Optional[float]
        breakdown: Optional[Dict[str, float]]
        eval_status: str
        if min_q > 0 and (new_q < min_q or baseline_q < min_q):
            reward_score = None
            breakdown = None
            eval_status = "insufficient_data"
        else:
            breakdown = self.calculate_reward_breakdown(
                baseline_metrics=baseline_metrics,
                new_metrics=new_metrics,
                rewrite_cost_sec=rewrite_cost_sec,
                table_name=table_name,
                cluster_id=cluster_id,
            )
            reward_score = breakdown["reward"]
            eval_status = "scored"

            # Guardrail: require confidence before accepting a positive win.
            # If not confident, clamp to 0 (neutral) rather than rewarding noise.
            if reward_score > 0:
                b_samples = self.metrics_calculator.get_latency_samples(
                    baseline_used,
                    start_time,
                    end_time,
                    table_name=table_name,
                    cluster_id=cluster_id,
                )
                n_samples = self.metrics_calculator.get_latency_samples(
                    layout_id,
                    start_time,
                    end_time,
                    table_name=table_name,
                    cluster_id=cluster_id,
                )
                b_samples = self._winsorize(
                    b_samples, self.config.reward_outlier_winsor_pctl
                )
                n_samples = self._winsorize(
                    n_samples, self.config.reward_outlier_winsor_pctl
                )
                lb = self._bootstrap_ci_lower_bound(
                    baseline=b_samples,
                    new=n_samples,
                    iters=int(self.config.reward_bootstrap_iters),
                    alpha=float(self.config.reward_confidence_alpha),
                    seed=self._bootstrap_seed(
                        layout_id=layout_id,
                        baseline_layout_id=str(baseline_used),
                        start_time=start_time,
                        end_time=end_time,
                    ),
                )
                if lb <= 0:
                    reward_score = 0.0
                    eval_status = "guarded_unconfident"

        # Record evaluation
        eval_id = self.metadata_store.record_evaluation(
            layout_id=layout_id,
            table_name=table_name,
            cluster_id=cluster_id,
            baseline_layout_id=baseline_used,
            eval_mode="natural_window",
            eval_status=eval_status,
            eval_window_start=start_time,
            eval_window_end=end_time,
            avg_latency_ms=new_metrics["avg_latency_ms"],
            p95_latency_ms=new_metrics.get("p95_latency_ms"),
            p99_latency_ms=new_metrics.get("p99_latency_ms"),
            avg_rows_scanned=new_metrics.get("avg_rows_scanned"),
            queries_evaluated=new_metrics["queries_evaluated"],
            rewrite_cost_sec=rewrite_cost_sec,
            reward_score=reward_score,
        )

        return {
            "eval_id": eval_id,
            "layout_id": layout_id,
            "metrics": new_metrics,
            "baseline_metrics": baseline_metrics,
            "reward_score": reward_score,
            "latency_improvement": (
                breakdown["latency_improvement"] if breakdown else 0.0
            ),
            "reward_breakdown": breakdown,
            "eval_status": eval_status,
            "min_queries_for_eval": min_q,
            "eval_window": {
                "start": start_time,
                "end": end_time,
            },
            "baseline_layout_id": baseline_used,
            "eval_mode": "natural_window",
            "rewrite_cost_sec": rewrite_cost_sec,
        }

    def evaluate_layout_window(
        self,
        *,
        layout_id: str,
        baseline_layout_id: Optional[str],
        start_time: datetime,
        end_time: datetime,
        rewrite_cost_sec: float = 0.0,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a layout vs a baseline on the same rolling time window.

        Used by the Phase 5 scheduler.
        """
        layout_info = self.metadata_store.get_layout(layout_id)
        if not layout_info:
            raise ValueError(f"Layout {layout_id} not found")

        table_name = layout_info["table_name"]
        if cluster_id is None:
            cluster_id = layout_info.get("cluster_id")

        baseline_used = baseline_layout_id or "initial"

        if baseline_layout_id:
            baseline_metrics = (
                self.metrics_calculator.calculate_layout_metrics(
                    baseline_layout_id,
                    start_time,
                    end_time,
                    table_name=table_name,
                    cluster_id=cluster_id,
                )
            )
        else:
            baseline_metrics = (
                self.metrics_calculator.calculate_layout_metrics(
                    "initial",
                    start_time,
                    end_time,
                    table_name=table_name,
                    cluster_id=cluster_id,
                )
            )

        new_metrics = self.metrics_calculator.calculate_layout_metrics(
            layout_id,
            start_time,
            end_time,
            table_name=table_name,
            cluster_id=cluster_id,
        )

        min_q = int(getattr(self.config, "min_queries_for_eval", 0) or 0)
        new_q = int(new_metrics.get("queries_evaluated", 0) or 0)
        baseline_q = int(baseline_metrics.get("queries_evaluated", 0) or 0)

        reward_score: Optional[float]
        breakdown: Optional[Dict[str, float]]
        eval_status: str

        if min_q > 0 and (new_q < min_q or baseline_q < min_q):
            reward_score = None
            breakdown = None
            eval_status = "insufficient_data"
        else:
            breakdown = self.calculate_reward_breakdown(
                baseline_metrics=baseline_metrics,
                new_metrics=new_metrics,
                rewrite_cost_sec=rewrite_cost_sec,
                table_name=table_name,
                cluster_id=cluster_id,
            )
            reward_score = breakdown["reward"]
            eval_status = "scored"

            if reward_score > 0:
                b_samples = self.metrics_calculator.get_latency_samples(
                    "initial"
                    if not baseline_layout_id
                    else baseline_layout_id,
                    start_time,
                    end_time,
                    table_name=table_name,
                    cluster_id=cluster_id,
                )
                n_samples = self.metrics_calculator.get_latency_samples(
                    layout_id,
                    start_time,
                    end_time,
                    table_name=table_name,
                    cluster_id=cluster_id,
                )
                b_samples = self._winsorize(
                    b_samples, self.config.reward_outlier_winsor_pctl
                )
                n_samples = self._winsorize(
                    n_samples, self.config.reward_outlier_winsor_pctl
                )
                lb = self._bootstrap_ci_lower_bound(
                    baseline=b_samples,
                    new=n_samples,
                    iters=int(self.config.reward_bootstrap_iters),
                    alpha=float(self.config.reward_confidence_alpha),
                    seed=self._bootstrap_seed(
                        layout_id=layout_id,
                        baseline_layout_id=str(baseline_used),
                        start_time=start_time,
                        end_time=end_time,
                    ),
                )
                if lb <= 0:
                    reward_score = 0.0
                    breakdown = None
                    eval_status = "guarded_unconfident"

        eval_id = self.metadata_store.record_evaluation(
            layout_id=layout_id,
            table_name=table_name,
            cluster_id=cluster_id,
            baseline_layout_id=baseline_used,
            eval_mode="natural_window",
            eval_status=eval_status,
            eval_window_start=start_time,
            eval_window_end=end_time,
            avg_latency_ms=new_metrics["avg_latency_ms"],
            p95_latency_ms=new_metrics.get("p95_latency_ms"),
            p99_latency_ms=new_metrics.get("p99_latency_ms"),
            avg_rows_scanned=new_metrics.get("avg_rows_scanned"),
            queries_evaluated=new_metrics["queries_evaluated"],
            rewrite_cost_sec=rewrite_cost_sec,
            reward_score=reward_score,
        )

        return {
            "eval_id": eval_id,
            "layout_id": layout_id,
            "metrics": new_metrics,
            "baseline_metrics": baseline_metrics,
            "reward_score": reward_score,
            "reward_breakdown": breakdown,
            "eval_status": eval_status,
            "min_queries_for_eval": min_q,
            "eval_window": {"start": start_time, "end": end_time},
            "baseline_layout_id": baseline_used,
            "eval_mode": "natural_window",
            "rewrite_cost_sec": rewrite_cost_sec,
        }
