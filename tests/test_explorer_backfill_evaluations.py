from __future__ import annotations

from datetime import datetime, timedelta, timezone

from database_optimiser.adaptive.explorer import LayoutExplorer
from database_optimiser.config import Config
from database_optimiser.evaluator.metrics import MetricsCalculator
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.layout.spec import LayoutSpec
from database_optimiser.storage.metadata import MetadataStore


class _NoopLayoutGenerator:
    def generate_candidate_layouts(self, table_name: str, window_hours=None):
        return []

    def generate_layout(self, table_name: str, window_hours=None):
        return LayoutSpec(
            partition_cols=None, sort_cols=None, file_size_mb=1.0
        )

    def generate_layout_id(self) -> str:
        return "layout_new"

    def generate_random_layout(self, table_name: str, window_hours=None):
        return LayoutSpec(
            partition_cols=None, sort_cols=None, file_size_mb=1.0
        )

    def compare_layouts(
        self, layout1: LayoutSpec, layout2: LayoutSpec
    ) -> bool:
        return True


class _NoopMigrator:
    def get_source_path(self, table_name: str):
        return None


class _NoopQueryExecutor:
    def __init__(self):
        self.registered = []

    def register_layout(
        self, table_name: str, layout_path: str, layout_id: str | None = None
    ) -> None:
        self.registered.append((table_name, layout_path, layout_id))


def _insert_query_log(
    store: MetadataStore,
    *,
    ts: datetime,
    table_name: str,
    layout_id: str | None,
    runtime_ms: float,
):
    with store._connection() as conn:  # noqa: SLF001
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_log (
              ts, user_id, table_name, layout_id, columns_used, predicates, joins,
              group_by_cols, order_by_cols, runtime_ms, rows_scanned, rows_returned, query_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000),
                None,
                table_name,
                layout_id,
                "[]",
                "[]",
                "[]",
                "[]",
                "[]",
                runtime_ms,
                None,
                0,
                "SELECT 1",
            ),
        )
        conn.commit()


def _set_layout_created_at(
    store: MetadataStore, layout_id: str, created_at: datetime
) -> None:
    with store._connection() as conn:  # noqa: SLF001
        cur = conn.cursor()
        cur.execute(
            "UPDATE table_layout SET created_at = ? WHERE layout_id = ?",
            (
                int(
                    created_at.replace(tzinfo=timezone.utc).timestamp() * 1000
                ),
                layout_id,
            ),
        )
        conn.commit()


def test_optimize_backfills_scored_evals_and_updates_bandit(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        eval_window_hours=24,
        min_queries_for_eval=1,
    )
    config.ensure_dirs()
    store = MetadataStore(config)
    metrics = MetricsCalculator(store)
    rewards = RewardCalculator(metrics, config, store)
    executor = _NoopQueryExecutor()

    explorer = LayoutExplorer(
        config,
        store,
        _NoopLayoutGenerator(),  # type: ignore[arg-type]
        _NoopMigrator(),  # type: ignore[arg-type]
        rewards,
        executor,  # type: ignore[arg-type]
    )

    table = "nyc_taxi"
    layout_id = "layout_backfill"
    store.create_layout(
        layout_id=layout_id,
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "layout_backfill"),
        file_size_mb=1.0,
    )
    store.activate_layout(layout_id, table)

    now = datetime.now(timezone.utc)
    created_at = now - timedelta(hours=1)
    _set_layout_created_at(store, layout_id, created_at)

    # Baseline query before layout creation
    _insert_query_log(
        store,
        ts=now - timedelta(hours=2),
        table_name=table,
        layout_id=None,
        runtime_ms=100.0,
    )
    # Layout query after creation
    _insert_query_log(
        store,
        ts=now - timedelta(minutes=30),
        table_name=table,
        layout_id=layout_id,
        runtime_ms=50.0,
    )

    # First optimize should backfill and return no_new_layout (since generator proposes none)
    result = explorer.optimize_table(table)
    assert result["status"] == "no_new_layout"

    # We should now have a scored evaluation for the existing layout
    evals = store.get_layout_evaluations(layout_id)
    assert any(e.get("reward_score") is not None for e in evals)

    bandit = explorer.get_bandit(table)
    stats = bandit.get_arm_stats()
    assert stats[layout_id]["pulls"] > 0
