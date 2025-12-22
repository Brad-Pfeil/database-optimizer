"""Workload execution utilities.

The core optimizer components are fairly heavyweight. For running workloads (especially from
CLI/scripts) we want a small, testable helper that optionally routes each query to a layout
selected by the adaptive explorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol


class _QueryExecutor(Protocol):
    def register_layout(
        self,
        table_name: str,
        layout_path: str,
        layout_id: Optional[str] = None,
    ) -> None: ...

    def run_query(self, sql: str) -> object: ...


class _MetadataStore(Protocol):
    def get_layout(self, layout_id: str) -> Optional[dict]: ...


class _LayoutExplorer(Protocol):
    def select_layout_for_query(
        self, table_name: str, sql: Optional[str] = None
    ) -> Optional[str]: ...


@dataclass
class WorkloadRunResult:
    success_count: int
    error_count: int


def run_queries(
    *,
    table_name: str,
    queries: Iterable[str],
    query_executor: _QueryExecutor,
    metadata_store: _MetadataStore,
    explore: bool = False,
    explorer: Optional[_LayoutExplorer] = None,
    on_error: Optional[Callable[[int, Exception], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> WorkloadRunResult:
    """
    Run queries, optionally routing each query to a selected layout.

    Notes:
    - This function does not generate queries; it only executes them.
    - Layout selection happens per query. We re-register only when the selected layout changes.
    """
    current_layout_id: Optional[str] = None
    success_count = 0
    error_count = 0

    total_queries = 0
    for query in queries:
        sql = query.strip()
        if not sql:
            continue
        total_queries += 1

        if explore and explorer is not None:
            selected_layout_id = explorer.select_layout_for_query(
                table_name, sql=sql
            )
            if selected_layout_id and selected_layout_id != current_layout_id:
                layout_info = metadata_store.get_layout(selected_layout_id)
                if not layout_info:
                    raise ValueError(
                        f"Layout {selected_layout_id} not found in metadata store"
                    )
                query_executor.register_layout(
                    table_name,
                    layout_info["layout_path"],
                    layout_info["layout_id"],
                )
                current_layout_id = selected_layout_id

        try:
            query_executor.run_query(sql)
            success_count += 1
        except Exception as e:
            error_count += 1
            if on_error is not None:
                on_error(total_queries, e)

        if on_progress is not None:
            on_progress(total_queries, success_count + error_count)

    return WorkloadRunResult(
        success_count=success_count, error_count=error_count
    )
