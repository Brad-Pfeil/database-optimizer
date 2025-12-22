from __future__ import annotations

from database_optimiser.analyzer.workload_clusterer import WorkloadClusterer
from database_optimiser.config import Config
from database_optimiser.storage.metadata import MetadataStore
from database_optimiser.storage.query_logger import QueryLogger


def test_clusterer_stable_cluster_for_same_context_key():
    c = WorkloadClusterer(num_clusters=8)
    key = "t|a|||noagg"
    assert c.cluster_id_for_context_key(key) == c.cluster_id_for_context_key(
        key
    )


def test_clusterer_different_keys_can_map_different_clusters():
    c = WorkloadClusterer(num_clusters=8)
    a = c.cluster_id_for_context_key("t|a|||noagg")
    b = c.cluster_id_for_context_key("t|b|||noagg")
    # Not guaranteed, but extremely likely. If collision occurs, increase clusters.
    assert a != b or c.num_clusters == 1


def test_query_logger_persists_cluster_id_and_context_key(tmp_path):
    config = Config(
        data_dir=tmp_path / "data", metadata_db_path=tmp_path / "meta.db"
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    logger = QueryLogger(store, config)

    sql = "SELECT * FROM t WHERE a = 1"
    logger.log_query_execution(
        sql=sql,
        runtime_ms=1.0,
        rows_returned=0,
        rows_scanned=None,
        layout_id=None,
    )

    logs = store.get_query_logs(table_name="t", limit=1)
    assert logs
    assert logs[0].get("context_key") is not None
    assert logs[0].get("cluster_id") is not None
