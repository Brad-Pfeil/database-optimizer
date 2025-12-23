from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from database_optimiser.adaptive.bandit import MultiArmedBandit
from database_optimiser.adaptive.explorer import LayoutExplorer
from database_optimiser.config import Config
from database_optimiser.evaluator.metrics import MetricsCalculator
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.storage.metadata import MetadataStore
from database_optimiser.workload.runner import run_queries


class _Dummy:
    pass


class _LoggingQueryExecutor:
    """
    Minimal executor that doesn't run SQL; it just records query logs with the current layout id.
    """

    def __init__(
        self,
        store: MetadataStore,
        table_name: str,
        runtimes_ms: dict[str, float],
    ):
        self.store = store
        self.table_name = table_name
        self.runtimes_ms = runtimes_ms
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
        runtime_ms = self.runtimes_ms.get(
            self.current_layout_id or "initial", 100.0
        )
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
            layout_id=self.current_layout_id,
        )
        return object()


def _set_layout_created_at(
    store: MetadataStore, layout_id: str, created_at: datetime
) -> None:
    with store._connection() as conn:  # noqa: SLF001 - test helper
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


def test_end_to_end_exploration_produces_rewards_and_bandit_pulls(tmp_path):
    config = Config(
        data_dir=tmp_path / "data",
        metadata_db_path=tmp_path / "meta.db",
        exploration_rate=0.5,  # encourage variety
        min_queries_for_eval=1,
        eval_window_hours=24,
    )
    config.ensure_dirs()
    store = MetadataStore(config)

    table = "nyc_taxi"
    now = datetime.now(timezone.utc)

    # Seed baseline (initial) queries before any layout creation
    with store._connection() as conn:  # noqa: SLF001 - test helper
        cur = conn.cursor()
        for i in range(5):
            cur.execute(
                """
                INSERT INTO query_log (
                  ts, user_id, table_name, layout_id, columns_used, predicates, joins,
                  group_by_cols, order_by_cols, runtime_ms, rows_scanned, rows_returned, query_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(
                        (now - timedelta(hours=2, minutes=i))
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                        * 1000
                    ),
                    None,
                    table,
                    None,
                    "[]",
                    "[]",
                    "[]",
                    "[]",
                    "[]",
                    100.0,
                    None,
                    0,
                    "SELECT 1",
                ),
            )
        conn.commit()

    # Create two layouts
    store.create_layout(
        layout_id="layout_a",
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "a"),
        file_size_mb=128.0,
    )
    store.create_layout(
        layout_id="layout_b",
        table_name=table,
        partition_cols=None,
        sort_cols=None,
        layout_path=str(tmp_path / "b"),
        file_size_mb=128.0,
    )
    _set_layout_created_at(store, "layout_a", now - timedelta(hours=1))
    _set_layout_created_at(store, "layout_b", now - timedelta(hours=1))

    # Route queries across the two layouts; make layout_b faster.
    executor = _LoggingQueryExecutor(
        store,
        table_name=table,
        runtimes_ms={"layout_a": 90.0, "layout_b": 40.0, "initial": 100.0},
    )

    metrics = MetricsCalculator(store)
    reward_calc = RewardCalculator(metrics, config, store)

    # Explorer is only used here for routing decisions.
    explorer = LayoutExplorer(
        config,
        store,
        _Dummy(),  # type: ignore[arg-type]
        _Dummy(),  # type: ignore[arg-type]
        reward_calc,
        executor,  # type: ignore[arg-type]
    )

    run_queries(
        table_name=table,
        queries=[f"SELECT {i}" for i in range(50)],
        query_executor=executor,
        metadata_store=store,
        explore=True,
        explorer=explorer,
    )

    # Evaluate both layouts; at least one should score (min_queries_for_eval=1)
    eval_a = reward_calc.evaluate_layout(
        "layout_a", eval_window_hours=24, rewrite_cost_sec=0.0
    )
    eval_b = reward_calc.evaluate_layout(
        "layout_b", eval_window_hours=24, rewrite_cost_sec=0.0
    )
    assert eval_a["reward_score"] is not None
    assert eval_b["reward_score"] is not None

    bandit = MultiArmedBandit(metadata_store=store)
    bandit.load_evaluations(table)
    stats = bandit.get_arm_stats()
    assert stats["layout_a"]["pulls"] > 0 or stats["layout_b"]["pulls"] > 0
    assert bandit.get_best_arm() in {"layout_a", "layout_b"}
