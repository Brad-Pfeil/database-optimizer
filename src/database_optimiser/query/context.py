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

# Stable warning codes (do not include raw SQL fragments in these).
WARN_CTE_PRESENT = "CTE_PRESENT"
WARN_NO_BASE_TABLE = "NO_BASE_TABLE"
WARN_MULTI_TABLE = "MULTI_TABLE"
WARN_UNRESOLVED_QUALIFIER = "UNRESOLVED_QUALIFIER"
WARN_SUBQUERY_FROM = "SUBQUERY_FROM"
WARN_WHERE_UNPARSED = "WHERE_UNPARSED"
WARN_POSITIONAL_GROUP_ORDER = "POSITIONAL_GROUP_ORDER"
WARN_GROUP_ORDER_NOISY = "GROUP_ORDER_NOISY"

_CLAUSE_KEYWORD_DENYLIST = {
    # ordering modifiers / null ordering
    "NULLS",
    "FIRST",
    "LAST",
    # boolean/null literals
    "NULL",
    "TRUE",
    "FALSE",
    # case expression keywords (avoid being misclassified as columns)
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
}


@dataclass(frozen=True)
class QueryContextParseResult:
    """Parse result for query context extraction (used for routing + learning gates)."""

    context: QueryContext
    parse_success: bool
    parse_confidence: float
    parse_warnings: tuple[str, ...] = ()
    parser_version: Optional[str] = None


def extract_query_context_result(
    sql: str, *, dialect: str = "duckdb"
) -> QueryContextParseResult:
    """Extract query context using a real SQL parser (sqlglot), with confidence scoring.

    Falls back to regex heuristics if parsing fails, but reports low confidence.
    """
    sql_str = sql.strip()

    # Default fallback result: regex-based extraction + diagnostics.
    fallback_res = _extract_query_context_regex_result(sql_str)

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        # sqlglot missing/unavailable: treat as no-parse.
        return fallback_res

    # If dialect is unknown to sqlglot, force fallback (useful for tests and safety).
    try:
        from sqlglot.dialects import Dialect

        Dialect.get_or_raise(dialect)
    except Exception:
        return QueryContextParseResult(
            context=fallback_res.context,
            parse_success=fallback_res.parse_success,
            parse_confidence=fallback_res.parse_confidence,
            parse_warnings=fallback_res.parse_warnings,
            parser_version=getattr(sqlglot, "__version__", None),
        )

    try:
        tree = sqlglot.parse_one(sql_str, read=dialect)
    except Exception:
        return QueryContextParseResult(
            context=fallback_res.context,
            parse_success=fallback_res.parse_success,
            parse_confidence=fallback_res.parse_confidence,
            parse_warnings=fallback_res.parse_warnings,
            parser_version=getattr(sqlglot, "__version__", None),
        )

    # Collect CTE names (these are not base tables).
    cte_names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        try:
            alias = cte.alias_or_name
        except Exception:
            alias = None
        if alias:
            cte_names.add(str(alias))

    # Alias mapping for qualified columns (best-effort).
    alias_to_table: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        if t.alias and t.name:
            alias_to_table[str(t.alias)] = str(t.name)

    # Determine top-level FROM/JOIN sources (do NOT treat CTE-internal tables as base tables).
    base_tables: list[str] = []
    from_arg = tree.args.get("from_") or tree.args.get("from")
    if from_arg is not None:
        sources = []
        if getattr(from_arg, "this", None) is not None:
            sources.append(from_arg.this)
        sources.extend(getattr(from_arg, "expressions", []) or [])
        for src in sources:
            if isinstance(src, exp.Subquery):
                base_tables = []
                break
            if isinstance(src, exp.Table) and src.name:
                n = str(src.name)
                if n in cte_names:
                    continue
                if n not in base_tables:
                    base_tables.append(n)

    # Include top-level JOIN tables as additional sources.
    for j in tree.args.get("joins") or []:
        jt = getattr(j, "this", None)
        if isinstance(jt, exp.Subquery):
            base_tables = []
            break
        if isinstance(jt, exp.Table) and jt.name:
            n = str(jt.name)
            if n in cte_names:
                continue
            if n not in base_tables:
                base_tables.append(n)

    # Resolve primary table if unambiguous.
    table_name: Optional[str]
    if len(base_tables) == 1:
        table_name = base_tables[0]
    else:
        table_name = None

    def _col_name(c: exp.Column) -> str:
        return str(c.name)

    def _qualifier_known(c: exp.Column) -> bool:
        if not c.table:
            return True  # unqualified; assume resolvable
        q = str(c.table)
        return q in alias_to_table or q in base_tables

    # Extract clause-specific column sets.
    filter_cols: set[str] = set()
    join_cols: set[str] = set()
    group_by_cols: set[str] = set()
    order_by_cols: set[str] = set()

    where_expr = tree.args.get("where")
    if where_expr is not None:
        for c in where_expr.find_all(exp.Column):
            filter_cols.add(_col_name(c))

    for j in tree.find_all(exp.Join):
        on_expr = j.args.get("on")
        if on_expr is not None:
            for c in on_expr.find_all(exp.Column):
                join_cols.add(_col_name(c))

    group_expr = tree.args.get("group")
    if group_expr is not None:
        for c in group_expr.find_all(exp.Column):
            group_by_cols.add(_col_name(c))

    order_expr = tree.args.get("order")
    if order_expr is not None:
        for c in order_expr.find_all(exp.Column):
            order_by_cols.add(_col_name(c))

    # Aggregate detection
    has_agg = any(
        isinstance(node, exp.AggFunc)
        for node in tree.walk()
        if node is not None
    )
    if not has_agg:
        # Fallback: any explicit COUNT/SUM/... function calls
        has_agg = any(
            isinstance(node, exp.Anonymous)
            and str(node.name).upper() in AGG_FUNCS
            for node in tree.find_all(exp.Anonymous)
        )

    warnings: list[str] = []

    # Confidence scoring
    confidence = 1.0
    if cte_names:
        warnings.append(WARN_CTE_PRESENT)
        confidence -= 0.2
    if re.search(r"\bFROM\s*\(", sql_str, re.IGNORECASE):
        warnings.append(WARN_SUBQUERY_FROM)
        confidence -= 0.5
    if len(base_tables) == 0:
        warnings.append(WARN_NO_BASE_TABLE)
        confidence -= 0.5
    elif len(base_tables) > 1:
        warnings.append(WARN_MULTI_TABLE)
        confidence -= 0.4

    unresolved_qualifiers = 0
    total_qualified = 0
    for c in tree.find_all(exp.Column):
        if c.table:
            total_qualified += 1
            if not _qualifier_known(c):
                unresolved_qualifiers += 1
    if total_qualified > 0 and unresolved_qualifiers > 0:
        warnings.append(WARN_UNRESOLVED_QUALIFIER)
        confidence -= 0.2

    confidence = max(0.0, min(1.0, confidence))

    ctx = QueryContext(
        table_name=table_name,
        filter_cols=frozenset(filter_cols),
        join_cols=frozenset(join_cols),
        group_by_cols=frozenset(group_by_cols),
        order_by_cols=frozenset(order_by_cols),
        has_agg=bool(has_agg),
    )

    return QueryContextParseResult(
        context=ctx,
        parse_success=True,
        parse_confidence=confidence,
        parse_warnings=tuple(sorted(set(warnings))),
        parser_version=getattr(sqlglot, "__version__", None),
    )


def extract_query_context(sql: str) -> QueryContext:
    """Backwards-compatible API: returns only the QueryContext."""
    return extract_query_context_result(sql).context


def _extract_query_context_regex_result(
    sql_str: str,
) -> QueryContextParseResult:
    """Regex-based extraction with diagnostics (fallback when sqlglot unavailable)."""

    warnings: list[str] = []

    # For CTE-heavy queries, restrict parsing to the main SELECT (best-effort).
    sql_main = sql_str
    if re.search(r"\bWITH\b", sql_str, re.IGNORECASE):
        last_main = None
        for m in re.finditer(r"\)\s*SELECT\b", sql_str, re.IGNORECASE):
            last_main = m
        if last_main is not None:
            sql_main = sql_str[last_main.start() + 1 :]

    # Detect CTE names (best-effort; regex-only)
    cte_names: set[str] = set()
    if re.search(r"\bWITH\b", sql_str, re.IGNORECASE):
        for m in re.finditer(
            r'\b("([^"]+)"|\w+)\s+AS\s*\(',
            sql_str,
            re.IGNORECASE,
        ):
            raw = m.group(1)
            name = raw.strip('"') if raw else None
            if name:
                cte_names.add(name)
        if cte_names:
            warnings.append(WARN_CTE_PRESENT)

    # If FROM is a subquery, we can't reliably extract a base table.
    if re.search(r"\bFROM\s*\(", sql_main, re.IGNORECASE):
        warnings.append(WARN_SUBQUERY_FROM)

    # Identifier patterns: schema-qualified + quoted identifiers.
    ident = r'(?:"[^"]+"|\w+)'
    qualified = rf"{ident}(?:\.{ident})?"

    def _norm_ident(x: str) -> str:
        # strip quotes, pick last component if schema-qualified
        parts = [p.strip() for p in x.split(".")]
        last = parts[-1] if parts else x
        return last.strip('"')

    # Base table candidates from FROM/JOIN (excluding CTE refs)
    from_ref = None
    from_match = re.search(
        rf"\bFROM\s+({qualified})",
        sql_main,
        re.IGNORECASE,
    )
    if from_match:
        from_ref = _norm_ident(from_match.group(1))

    join_refs: list[str] = []
    for m in re.finditer(rf"\bJOIN\s+({qualified})", sql_main, re.IGNORECASE):
        join_refs.append(_norm_ident(m.group(1)))

    base_tables = []
    for t in [from_ref, *join_refs]:
        if not t:
            continue
        if t in cte_names:
            continue
        if t not in base_tables:
            base_tables.append(t)

    # Table name resolution: only if unambiguous and not a subquery/cte-only.
    if len(base_tables) == 1 and WARN_SUBQUERY_FROM not in warnings:
        table_name: Optional[str] = base_tables[0]
    else:
        table_name = None
        if len(base_tables) > 1:
            warnings.append(WARN_MULTI_TABLE)
        if len(base_tables) == 0:
            warnings.append(WARN_NO_BASE_TABLE)

    # Agg heuristic: if SELECT contains an aggregate function token
    select_match = re.search(
        r"\bSELECT\s+(.*?)\s+FROM\b", sql_str, re.IGNORECASE | re.DOTALL
    )
    has_agg = False
    if select_match:
        select_clause = select_match.group(1).upper()
        has_agg = any(f"{fn}(" in select_clause for fn in AGG_FUNCS)

    filter_cols = set()
    where_present = bool(re.search(r"\bWHERE\b", sql_main, re.IGNORECASE))
    where_match = re.search(
        r"\bWHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)",
        sql_main,
        re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        where_clause = where_match.group(1)
        # column operator value pattern (lightweight)
        for m in re.finditer(
            r"\b(?:\w+\.)?(\w+)\s*(=|<|>|<=|>=|!=|LIKE|IN|BETWEEN)\s+",
            where_clause,
            re.IGNORECASE,
        ):
            filter_cols.add(m.group(1))
    elif where_present:
        warnings.append(WARN_WHERE_UNPARSED)

    join_cols = set()
    for m in re.finditer(
        r"\bJOIN\s+(?:\w+|\"[^\"]+\")(?:\.(?:\w+|\"[^\"]+\"))?\s+ON\s+(?:\w+\.)?(\w+)\s*=\s*(?:\w+\.)?(\w+)",
        sql_main,
        re.IGNORECASE,
    ):
        join_cols.add(m.group(1))
        join_cols.add(m.group(2))

    def _extract_clause_cols(clause: str) -> tuple[set[str], int, int, int]:
        # Returns: (cols, dropped_numeric, dropped_keywords, dropped_functions)
        cols: set[str] = set()
        dropped_numeric = 0
        dropped_keywords = 0
        dropped_functions = 0

        for m in re.finditer(r'("([^"]+)"|\w+)(?:\.(?:"[^"]+"|\w+))?', clause):
            raw = m.group(0)
            token = _norm_ident(raw)
            upper = token.upper()
            if token.isdigit():
                dropped_numeric += 1
                continue
            if upper in _CLAUSE_KEYWORD_DENYLIST:
                dropped_keywords += 1
                continue
            # Skip ordering keywords that appear in ORDER BY tails
            if upper in ("ASC", "DESC", "AND", "OR"):
                continue

            # Skip function names: token followed by '(' (ignoring whitespace)
            j = m.end()
            while j < len(clause) and clause[j].isspace():
                j += 1
            if j < len(clause) and clause[j] == "(":
                dropped_functions += 1
                continue

            cols.add(token)

        return (cols, dropped_numeric, dropped_keywords, dropped_functions)

    group_by_cols: set[str] = set()
    dropped_g_num = dropped_g_kw = dropped_g_fn = 0
    group_by_match = re.search(
        r"\bGROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|$)",
        sql_main,
        re.IGNORECASE | re.DOTALL,
    )
    if group_by_match:
        group_clause = group_by_match.group(1)
        group_by_cols, dropped_g_num, dropped_g_kw, dropped_g_fn = (
            _extract_clause_cols(group_clause)
        )

    order_by_cols: set[str] = set()
    dropped_o_num = dropped_o_kw = dropped_o_fn = 0
    order_by_match = re.search(
        r"\bORDER\s+BY\s+(.*?)$",
        sql_main,
        re.IGNORECASE | re.DOTALL,
    )
    if order_by_match:
        order_clause = order_by_match.group(1)
        order_by_cols, dropped_o_num, dropped_o_kw, dropped_o_fn = (
            _extract_clause_cols(order_clause)
        )

    dropped_numeric = dropped_g_num + dropped_o_num
    dropped_keywords = dropped_g_kw + dropped_o_kw
    dropped_functions = dropped_g_fn + dropped_o_fn
    if dropped_numeric > 0:
        warnings.append(WARN_POSITIONAL_GROUP_ORDER)

    extracted_clause_cols = len(group_by_cols) + len(order_by_cols)
    total_seen = (
        extracted_clause_cols
        + dropped_numeric
        + dropped_keywords
        + dropped_functions
    )
    if total_seen == 0:
        group_order_sane = True
    else:
        group_order_sane = (extracted_clause_cols / total_seen) >= 0.6
        if not group_order_sane:
            warnings.append(WARN_GROUP_ORDER_NOISY)

    # Heuristic confidence (review-guided)
    confidence = 0.0
    if table_name is not None:
        confidence += 0.3
    if (not where_present) or (where_match is not None):
        confidence += 0.2
    if dropped_functions == 0:
        confidence += 0.2
    if group_order_sane:
        confidence += 0.3
    # If FROM is a subquery, cap confidence (too ambiguous).
    if WARN_SUBQUERY_FROM in warnings:
        confidence = min(confidence, 0.4)
    if WARN_NO_BASE_TABLE in warnings:
        confidence = min(confidence, 0.6)
    confidence = max(0.0, min(1.0, confidence))

    # parse_success reflects whether the heuristic parser found a credible, unambiguous base table.
    parse_success = (
        table_name is not None
        and WARN_SUBQUERY_FROM not in warnings
        and WARN_MULTI_TABLE not in warnings
        and WARN_NO_BASE_TABLE not in warnings
    )

    ctx = QueryContext(
        table_name=table_name,
        filter_cols=frozenset(filter_cols),
        join_cols=frozenset(join_cols),
        group_by_cols=frozenset(group_by_cols),
        order_by_cols=frozenset(order_by_cols),
        has_agg=has_agg,
    )

    return QueryContextParseResult(
        context=ctx,
        parse_success=parse_success,
        parse_confidence=confidence,
        parse_warnings=tuple(sorted(set(warnings))),
        parser_version=None,
    )


def _extract_query_context_regex(sql_str: str) -> QueryContext:
    """Legacy API for regex extraction (context only)."""
    return _extract_query_context_regex_result(sql_str).context
