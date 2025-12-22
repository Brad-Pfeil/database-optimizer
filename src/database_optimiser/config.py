"""Configuration management for the adaptive layout generator."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration settings for the database optimizer."""

    # Data storage paths
    data_dir: Path = Path("data")
    metadata_db_path: Path = Path("metadata.db")

    # Layout settings
    default_file_size_mb: float = 128.0
    max_layout_versions: int = 5

    # Bandit algorithm settings
    exploration_rate: float = 0.2  # 20% traffic to experiments
    alpha: float = 1.0  # Weight for latency improvement in reward
    beta: float = 0.1  # Weight for rewrite cost in reward

    # Evaluation settings
    eval_window_hours: int = 24  # Hours to evaluate a layout
    min_queries_for_eval: int = 10  # Minimum queries before evaluating

    # Reward v2 settings (weights + guardrails)
    reward_w_mean: float = 1.0
    reward_w_p95: float = 0.25
    reward_w_p99: float = 0.25
    reward_w_rows_scanned: float = 0.1
    reward_w_variance: float = 0.0  # optional penalty for instability

    reward_confidence_alpha: float = 0.05  # significance level for guardrails
    reward_bootstrap_iters: int = (
        200  # iterations for bootstrap CI (mean latency improvement)
    )
    reward_outlier_winsor_pctl: float = (
        0.99  # winsorize latency samples at this percentile
    )

    # Workload analysis settings
    analysis_window_hours: int = 24  # Hours of queries to analyze

    # Clustering settings (Phase 3)
    num_clusters_per_table: int = 8

    # Candidate generation (Phase 4)
    candidate_beam_width: int = 30
    candidate_max_partition_cols: int = 2
    candidate_max_sort_cols: int = 3
    candidate_top_k_partition: int = 6
    candidate_top_k_sort: int = 6

    # Evaluation scheduler (Phase 5)
    eval_scheduler_interval_sec: int = 300  # loop sleep interval
    eval_scheduler_min_new_queries: int = (
        50  # per layout since last eval window
    )

    # Async/incremental migration (Phase 6)
    migration_batch_files: int = 10

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            metadata_db_path=Path(os.getenv("METADATA_DB", "metadata.db")),
            exploration_rate=float(os.getenv("EXPLORATION_RATE", "0.2")),
            alpha=float(os.getenv("ALPHA", "1.0")),
            beta=float(os.getenv("BETA", "0.1")),
        )

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
