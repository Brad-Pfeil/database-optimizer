from __future__ import annotations

from database_optimiser.query.context import (
    WARN_CTE_PRESENT,
    WARN_NO_BASE_TABLE,
    WARN_POSITIONAL_GROUP_ORDER,
    WARN_SUBQUERY_FROM,
    extract_query_context_result,
)


def test_sqlglot_alias_primary_table() -> None:
    sql = "SELECT * FROM trips t WHERE t.a = 1"
    res = extract_query_context_result(sql)
    assert res.parse_success is True
    assert res.context.table_name == "trips"
    assert res.parse_confidence >= 0.7
    assert res.parse_warnings == ()


def test_sqlglot_schema_qualified_and_quoted_from() -> None:
    sql = 'SELECT * FROM "S"."T"'
    res = extract_query_context_result(sql)
    assert res.parse_success is True
    assert res.context.table_name == "T"
    assert res.parse_confidence >= 0.7


def test_sqlglot_cte_from_has_no_base_table_and_low_confidence() -> None:
    sql = "WITH x AS (SELECT * FROM trips) SELECT * FROM x"
    res = extract_query_context_result(sql)
    assert (
        res.parse_success is True
    )  # parse succeeded, but ambiguous base table
    assert res.context.table_name is None
    assert WARN_CTE_PRESENT in res.parse_warnings
    assert WARN_NO_BASE_TABLE in res.parse_warnings
    assert res.parse_confidence <= 0.6


def test_regex_fallback_subquery_from_warns_and_low_confidence() -> None:
    # Force regex fallback by using an invalid sqlglot dialect.
    sql = "SELECT * FROM (SELECT * FROM trips) t"
    res = extract_query_context_result(sql, dialect="__nope__")
    assert res.context.table_name is None
    assert WARN_SUBQUERY_FROM in res.parse_warnings
    assert res.parse_confidence <= 0.4


def test_regex_fallback_group_order_filters_positionals_keywords_and_functions() -> (
    None
):
    sql = "SELECT * FROM t GROUP BY 1,2 ORDER BY lower(a) NULLS LAST"
    res = extract_query_context_result(sql, dialect="__nope__")
    assert WARN_POSITIONAL_GROUP_ORDER in res.parse_warnings
    assert res.context.group_by_cols == frozenset()
    assert res.context.order_by_cols == frozenset({"a"})
