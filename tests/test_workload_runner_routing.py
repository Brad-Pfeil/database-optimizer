from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from database_optimiser.workload.runner import run_queries


@dataclass
class _FakeQueryExecutor:
    registered: list[tuple[str, str, Optional[str]]]
    executed: list[str]

    def register_layout(
        self,
        table_name: str,
        layout_path: str,
        layout_id: Optional[str] = None,
    ) -> None:
        self.registered.append((table_name, layout_path, layout_id))

    def run_query(self, sql: str) -> object:
        self.executed.append(sql)
        return object()


class _FakeMetadataStore:
    def __init__(self, layouts: dict[str, dict]):
        self.layouts = layouts

    def get_layout(self, layout_id: str) -> Optional[dict]:
        return self.layouts.get(layout_id)


class _SeqExplorer:
    def __init__(self, seq: list[Optional[str]]):
        self.seq = list(seq)
        self.i = 0

    def select_layout_for_query(
        self, table_name: str, sql: Optional[str] = None
    ) -> Optional[str]:
        if self.i >= len(self.seq):
            return None
        v = self.seq[self.i]
        self.i += 1
        return v


def test_runner_registers_only_on_layout_change():
    layouts = {
        "layout_a": {"layout_id": "layout_a", "layout_path": "/tmp/a"},
        "layout_b": {"layout_id": "layout_b", "layout_path": "/tmp/b"},
    }
    metadata = _FakeMetadataStore(layouts)
    executor = _FakeQueryExecutor(registered=[], executed=[])
    explorer = _SeqExplorer(["layout_a", "layout_a", "layout_b", None])

    out = run_queries(
        table_name="nyc_taxi",
        queries=["SELECT 1", "SELECT 2", "SELECT 3", "SELECT 4"],
        query_executor=executor,
        metadata_store=metadata,
        explore=True,
        explorer=explorer,
    )

    assert out.success_count == 4
    assert out.error_count == 0
    assert executor.registered == [
        ("nyc_taxi", "/tmp/a", "layout_a"),
        ("nyc_taxi", "/tmp/b", "layout_b"),
    ]
    assert executor.executed == [
        "SELECT 1",
        "SELECT 2",
        "SELECT 3",
        "SELECT 4",
    ]
