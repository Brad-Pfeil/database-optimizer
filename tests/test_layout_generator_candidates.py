from __future__ import annotations

from database_optimiser.config import Config
from database_optimiser.layout.generator import LayoutGenerator


class _FakeAnalyzer:
    def get_partition_candidates(
        self, table_name: str, window_hours=None, cluster_id=None
    ):
        return [("event_date", 10.0), ("customer_id", 5.0)]

    def get_sort_candidates(
        self,
        table_name: str,
        window_hours=None,
        max_cols: int = 3,
        cluster_id=None,
    ):
        return ["customer_id", "event_date"][:max_cols]


def test_generate_candidate_layouts_produces_variation_and_dedupes(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
    )
    gen = LayoutGenerator(_FakeAnalyzer(), config)  # type: ignore[arg-type]

    candidates = gen.generate_candidate_layouts("events", window_hours=24)
    assert len(candidates) >= 3

    # Ensure no duplicates
    keys = {
        (tuple(c.partition_cols or ()), tuple(c.sort_cols or ()))
        for c in candidates
    }
    assert len(keys) == len(candidates)

    # Ensure at least one candidate is unpartitioned and one is partitioned
    assert any(c.partition_cols is None for c in candidates)
    assert any(c.partition_cols is not None for c in candidates)
