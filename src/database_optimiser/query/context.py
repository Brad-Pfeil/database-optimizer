"""Query context extraction for contextual routing.

This module intentionally uses lightweight heuristics (regex) so it can run without heavy SQL
parsers. The context is meant to be stable and low-cardinality enough to support online learning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class QueryContext:
    table_name: Optional[str]
    filter_cols: FrozenSet[str]
    join_cols: FrozenSet[str]
    group_by_cols: FrozenSet[str]
    order_by_cols: FrozenSet[str]
    has_agg: bool

    def key(self) -> str:
        """A stable string key suitable for bucketing."""
        return "|".join(
            [
                self.table_name or "",
                ",".join(sorted(self.filter_cols)),
                ",".join(sorted(self.join_cols)),
                ",".join(sorted(self.group_by_cols)),
                ",".join(sorted(self.order_by_cols)),
                "agg" if self.has_agg else "noagg",
            ]
        )


AGG_FUNCS = ("COUNT", "SUM", "AVG", "MIN", "MAX")


def extract_query_context(sql: str) -> QueryContext:
    sql_str = sql.strip()

    # Table name (simple heuristic)
    table_name = None
    from_match = re.search(r"\bFROM\s+(\w+)", sql_str, re.IGNORECASE)
    if from_match:
        table_name = from_match.group(1)

    # Agg heuristic: if SELECT contains an aggregate function token
    select_match = re.search(
        r"\bSELECT\s+(.*?)\s+FROM\b", sql_str, re.IGNORECASE | re.DOTALL
    )
    has_agg = False
    if select_match:
        select_clause = select_match.group(1).upper()
        has_agg = any(f"{fn}(" in select_clause for fn in AGG_FUNCS)

    filter_cols = set()
    where_match = re.search(
        r"\bWHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)",
        sql_str,
        re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        where_clause = where_match.group(1)
        # column operator value pattern (lightweight)
        # Note: don't use \b after operators like '=' (not a word char).
        for m in re.finditer(
            r"\b(?:\w+\.)?(\w+)\s*(=|<|>|<=|>=|!=|LIKE|IN|BETWEEN)\s+",
            where_clause,
            re.IGNORECASE,
        ):
            filter_cols.add(m.group(1))

    join_cols = set()
    for m in re.finditer(
        r"\bJOIN\s+(\w+)\s+ON\s+(?:\w+\.)?(\w+)\s*=\s*(?:\w+\.)?(\w+)",
        sql_str,
        re.IGNORECASE,
    ):
        join_cols.add(m.group(2))
        join_cols.add(m.group(3))

    group_by_cols = set()
    group_by_match = re.search(
        r"\bGROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|$)",
        sql_str,
        re.IGNORECASE | re.DOTALL,
    )
    if group_by_match:
        group_clause = group_by_match.group(1)
        for m in re.finditer(r"\b(?:\w+\.)?(\w+)\b", group_clause):
            col = m.group(1)
            if col.upper() not in ("AND", "OR"):
                group_by_cols.add(col)

    order_by_cols = set()
    order_by_match = re.search(
        r"\bORDER\s+BY\s+(.*?)$", sql_str, re.IGNORECASE | re.DOTALL
    )
    if order_by_match:
        order_clause = order_by_match.group(1)
        for m in re.finditer(r"\b(?:\w+\.)?(\w+)\b", order_clause):
            col = m.group(1)
            if col.upper() not in ("ASC", "DESC"):
                order_by_cols.add(col)

    return QueryContext(
        table_name=table_name,
        filter_cols=frozenset(filter_cols),
        join_cols=frozenset(join_cols),
        group_by_cols=frozenset(group_by_cols),
        order_by_cols=frozenset(order_by_cols),
        has_agg=has_agg,
    )
