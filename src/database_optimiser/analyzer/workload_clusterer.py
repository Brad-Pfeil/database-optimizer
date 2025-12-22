"""Deterministic workload clustering (dataset-agnostic).

We cluster queries by their query-shape signature derived from `QueryContext.key()`.
This is stable across runs and requires no training data.
"""

from __future__ import annotations

import hashlib


class WorkloadClusterer:
    def __init__(self, num_clusters: int = 8):
        if num_clusters <= 0:
            raise ValueError("num_clusters must be > 0")
        self.num_clusters = num_clusters

    def cluster_id_for_context_key(self, context_key: str) -> str:
        h = hashlib.sha1(context_key.encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % self.num_clusters
        return f"c{bucket}"
