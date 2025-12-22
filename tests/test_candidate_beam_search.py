from __future__ import annotations

import json
from unittest.mock import Mock

from database_optimiser.adaptive.explorer import LayoutExplorer
from database_optimiser.analyzer.workload_analyzer import WorkloadAnalyzer
from database_optimiser.config import Config
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.layout.generator import LayoutGenerator
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.query.executor import QueryExecutor
from database_optimiser.storage.metadata import MetadataStore


class _FakeAnalyzer:
    def get_partition_candidates(
        self, table_name: str, window_hours=None, cluster_id=None
    ):
        return [("a", 10.0), ("b", 9.0), ("c", 1.0)]

    def get_sort_candidates(
        self,
        table_name: str,
        window_hours=None,
        max_cols: int = 3,
        cluster_id=None,
    ):
        return ["a", "b", "c"][:max_cols]


def test_beam_search_generates_multicolumn_candidates(tmp_path):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    gen = LayoutGenerator(_FakeAnalyzer(), config)  # type: ignore[arg-type]

    candidates = gen.generate_candidate_layouts("t", window_hours=24)
    assert candidates
    # Expect at least one multi-column partition or sort candidate given inputs.
    assert any(
        (c.partition_cols and len(c.partition_cols) >= 2)
        or (c.sort_cols and len(c.sort_cols) >= 2)
        for c in candidates
    )


def test_equivalence_dedupe_derived_partition_mapping(tmp_path):
    # Existing layout partitioned on derived col, but notes map it to base col.
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    store.create_layout(
        layout_id="layout_existing",
        table_name="t",
        partition_cols=["a_year_month"],
        sort_cols=None,
        layout_path=str(tmp_path / "x"),
        file_size_mb=1.0,
        notes=json.dumps({"derived_partition_cols": {"a_year_month": "a"}}),
    )

    class _Gen(LayoutGenerator):
        def generate_candidate_layouts(
            self,
            table_name: str,
            window_hours: int | None = None,
            cluster_id: str | None = None,
            max_partition_candidates: int = 3,
            max_sort_candidates: int = 3,
        ) -> list[LayoutSpec]:
            return [
                LayoutSpec(
                    partition_cols=["a"], sort_cols=None, file_size_mb=1.0
                )
            ]

    # Minimal explorer wiring (we won't execute queries/migrations).
    analyzer = WorkloadAnalyzer(store)
    gen = _Gen(analyzer, config)
    explorer = LayoutExplorer(
        config=config,
        metadata_store=store,
        layout_generator=gen,
        layout_migrator=Mock(spec=LayoutMigrator),  # not used in this test
        reward_calculator=Mock(spec=RewardCalculator),  # not used in this test
        query_executor=Mock(spec=QueryExecutor),  # not used in this test
    )

    proposed = explorer.propose_new_layout("t", window_hours=24)
    assert proposed is None
