from __future__ import annotations

from typing import Optional

from database_optimiser.adaptive.explorer import LayoutExplorer
from database_optimiser.analyzer.workload_clusterer import WorkloadClusterer
from database_optimiser.config import Config
from database_optimiser.evaluator.metrics import MetricsCalculator
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.evaluator.scheduler import EvaluationScheduler
from database_optimiser.query.context import extract_query_context
from database_optimiser.storage.metadata import MetadataStore
from database_optimiser.workload.runner import run_queries


class _Dummy:
    pass


class _ClusterAwareLoggingExecutor:
    """Logs query runtimes to metadata.db, with runtime depending on (layout_id, query family)."""

    def __init__(
        self,
        store: MetadataStore,
        clusterer: WorkloadClusterer,
        table_name: str,
    ):
        self.store = store
        self.clusterer = clusterer
        self.table_name = table_name
        self.current_layout_id: Optional[str] = None

    def register_layout(
        self,
        table_name: str,
        layout_path: str,
        layout_id: Optional[str] = None,
    ) -> None:
        assert table_name == self.table_name
        self.current_layout_id = layout_id

    def run_query(self, sql: str) -> object:
        ctx = extract_query_context(sql)
        cluster_id = self.clusterer.cluster_id_for_context_key(ctx.key())

        # Two synthetic families: filter on a vs filter on b.
        is_a = "WHERE a" in sql
        is_b = "WHERE b" in sql

        # Baseline is mediocre; correct layout is faster for its family.
        lid = self.current_layout_id or "initial"
        if is_a:
            runtime_ms = (
                40.0
                if lid == "layout_a"
                else 120.0
                if lid == "layout_b"
                else 80.0
            )
        elif is_b:
            runtime_ms = (
                40.0
                if lid == "layout_b"
                else 120.0
                if lid == "layout_a"
                else 80.0
            )
        else:
            runtime_ms = 80.0

        self.store.log_query(
            table_name=self.table_name,
            columns_used=[],
            predicates=[],
            joins=[],
            group_by_cols=[],
            order_by_cols=[],
            runtime_ms=runtime_ms,
            rows_scanned=None,
            rows_returned=0,
            query_text=sql,
            user_id=None,
            layout_id=None if lid == "initial" else lid,
            context_key=ctx.key(),
            cluster_id=cluster_id,
        )
        return object()


def _find_num_clusters_to_separate_keys(key1: str, key2: str) -> int:
    for k in range(2, 257):
        c = WorkloadClusterer(num_clusters=k)
        if c.cluster_id_for_context_key(key1) != c.cluster_id_for_context_key(
            key2
        ):
            return k
    raise AssertionError(
        "Unable to find num_clusters that separates the two contexts"
    )


def test_scenario_two_query_families_learn_separate_layouts(tmp_path):
    table = "t"
    sql_a = "SELECT * FROM t WHERE a = 1"
    sql_b = "SELECT * FROM t WHERE b = 1"
    key_a = extract_query_context(sql_a).key()
    key_b = extract_query_context(sql_b).key()

    num_clusters = _find_num_clusters_to_separate_keys(key_a, key_b)

    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        exploration_rate=0.0,  # deterministic routing via contextual bonus within cluster
        num_clusters_per_table=num_clusters,
        min_queries_for_eval=5,
        eval_window_hours=24,
        eval_scheduler_min_new_queries=1,
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    clusterer = WorkloadClusterer(num_clusters=config.num_clusters_per_table)

    cluster_a = clusterer.cluster_id_for_context_key(key_a)
    cluster_b = clusterer.cluster_id_for_context_key(key_b)
    assert cluster_a != cluster_b

    # Baseline layout (global) and two cluster-scoped layouts.
    store.create_layout(
        layout_id="baseline",
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "baseline"),
        file_size_mb=1.0,
        cluster_id=None,
    )
    store.activate_layout("baseline", table)
    store.create_layout(
        layout_id="layout_a",
        table_name=table,
        partition_cols=None,
        sort_cols=["a"],
        layout_path=str(tmp_path / "a"),
        file_size_mb=1.0,
        cluster_id=cluster_a,
    )
    store.create_layout(
        layout_id="layout_b",
        table_name=table,
        partition_cols=None,
        sort_cols=["b"],
        layout_path=str(tmp_path / "b"),
        file_size_mb=1.0,
        cluster_id=cluster_b,
    )

    # Seed baseline queries per cluster so baseline comparisons are meaningful.
    executor = _ClusterAwareLoggingExecutor(store, clusterer, table_name=table)
    executor.register_layout(table, str(tmp_path / "baseline"), "baseline")
    for i in range(8):
        executor.run_query(sql_a)
        executor.run_query(sql_b)

    metrics = MetricsCalculator(store)
    reward_calc = RewardCalculator(metrics, config, store)
    explorer = LayoutExplorer(
        config,
        store,
        _Dummy(),  # type: ignore[arg-type]
        _Dummy(),  # type: ignore[arg-type]
        reward_calc,
        executor,  # type: ignore[arg-type]
    )

    # Verify routing chooses the cluster-appropriate layout.
    assert explorer.select_layout_for_query(table, sql=sql_a) == "layout_a"
    assert explorer.select_layout_for_query(table, sql=sql_b) == "layout_b"

    # Run workload (mixed families); logs are written with cluster ids.
    queries = [sql_a] * 20 + [sql_b] * 20
    run_queries(
        table_name=table,
        queries=queries,
        query_executor=executor,
        metadata_store=store,
        explore=True,
        explorer=explorer,
    )

    # Scheduler evaluates on rolling window; expect both layouts to score and be positive.
    scheduler = EvaluationScheduler(config, store, reward_calc)
    n = scheduler.evaluate_table(table_name=table)
    assert n >= 1

    eval_a = store.get_latest_evaluation("layout_a")
    eval_b = store.get_latest_evaluation("layout_b")
    assert (
        eval_a is not None
        and eval_a["reward_score"] is not None
        and float(eval_a["reward_score"]) > 0
    )
    assert (
        eval_b is not None
        and eval_b["reward_score"] is not None
        and float(eval_b["reward_score"]) > 0
    )
