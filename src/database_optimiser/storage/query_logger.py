"""Query logging with SQL parsing to extract metadata."""

import re
from typing import Any, Dict, List, Optional

from ..analyzer.workload_clusterer import WorkloadClusterer
from ..config import Config
from ..query.context import extract_query_context_result
from .metadata import MetadataStore


class QueryLogger:
    """Parses SQL queries and logs metadata to the metadata store."""

    def __init__(
        self, metadata_store: MetadataStore, config: Optional[Config] = None
    ):
        """Initialize query logger."""
        self.metadata_store = metadata_store
        self.config = config or Config()
        self.clusterer = WorkloadClusterer(
            num_clusters=self.config.num_clusters_per_table
        )

    def parse_query(self, sql: str) -> Dict[str, Any]:
        """Parse SQL query to extract metadata."""
        # Extract table names (simple heuristic: FROM/JOIN clauses)
        table_names = self._extract_tables(sql)
        primary_table = table_names[0] if table_names else None

        # Extract columns used
        columns_used = self._extract_columns(sql, table_names)

        # Extract predicates (WHERE clause conditions)
        predicates = self._extract_predicates(sql)

        # Extract joins
        joins = self._extract_joins(sql)

        # Extract GROUP BY columns
        group_by_cols = self._extract_group_by(sql)

        # Extract ORDER BY columns
        order_by_cols = self._extract_order_by(sql)

        return {
            "table_name": primary_table,
            "columns_used": columns_used,
            "predicates": predicates,
            "joins": joins,
            "group_by_cols": group_by_cols,
            "order_by_cols": order_by_cols,
        }

    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL."""
        tables = []

        # Match FROM table_name
        from_match = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))

        # Match JOIN table_name
        join_matches = re.finditer(r"\bJOIN\s+(\w+)", sql, re.IGNORECASE)
        for match in join_matches:
            table = match.group(1)
            if table not in tables:
                tables.append(table)

        return tables

    def _extract_columns(self, sql: str, table_names: List[str]) -> List[str]:
        """Extract column names from SQL."""
        columns = set()

        # Extract from SELECT clause
        select_match = re.search(
            r"\bSELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL
        )
        if select_match:
            select_clause = select_match.group(1)
            # Match column references (table.col or just col)
            col_pattern = r"\b(?:\w+\.)?(\w+)\b"
            for match in re.finditer(col_pattern, select_clause):
                col = match.group(1)
                # Skip SQL keywords and functions
                if col.upper() not in [
                    "AS",
                    "COUNT",
                    "SUM",
                    "AVG",
                    "MAX",
                    "MIN",
                    "DISTINCT",
                    "CASE",
                    "WHEN",
                    "THEN",
                    "ELSE",
                    "END",
                ]:
                    columns.add(col)

        # Extract from WHERE clause
        where_match = re.search(
            r"\bWHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if where_match:
            where_clause = where_match.group(1)
            for match in re.finditer(r"\b(?:\w+\.)?(\w+)\b", where_clause):
                col = match.group(1)
                if col.upper() not in [
                    "AND",
                    "OR",
                    "NOT",
                    "IN",
                    "LIKE",
                    "BETWEEN",
                    "IS",
                    "NULL",
                ]:
                    columns.add(col)

        # Extract from GROUP BY
        group_by_match = re.search(
            r"\bGROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if group_by_match:
            group_clause = group_by_match.group(1)
            for match in re.finditer(r"\b(?:\w+\.)?(\w+)\b", group_clause):
                columns.add(match.group(1))

        # Extract from ORDER BY
        order_by_match = re.search(
            r"\bORDER\s+BY\s+(.*?)$", sql, re.IGNORECASE | re.DOTALL
        )
        if order_by_match:
            order_clause = order_by_match.group(1)
            for match in re.finditer(r"\b(?:\w+\.)?(\w+)\b", order_clause):
                columns.add(match.group(1))

        return sorted(list(columns))

    def _extract_predicates(self, sql: str) -> List[Dict[str, Any]]:
        """Extract WHERE clause predicates."""
        predicates = []

        where_match = re.search(
            r"\bWHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if not where_match:
            return predicates

        where_clause = where_match.group(1)

        # Match patterns like: column operator value
        # e.g., date > '2025-01-01', customer_id = 123
        pattern = r"\b(?:\w+\.)?(\w+)\s*([=<>!]+|LIKE|IN|BETWEEN|IS)\s*([^\s]+(?:\s+[^\s]+)*?)(?:\s+AND|\s+OR|$)"

        for match in re.finditer(pattern, where_clause, re.IGNORECASE):
            col = match.group(1)
            op = match.group(2).strip()
            value = match.group(3).strip()

            # Clean up value (remove quotes, trailing operators)
            value = re.sub(r'^[\'"]|[\'"]$', "", value)
            value = re.sub(r"\s+(AND|OR).*$", "", value, flags=re.IGNORECASE)

            predicates.append(
                {
                    "col": col,
                    "op": op,
                    "value": value,
                }
            )

        return predicates

    def _extract_joins(self, sql: str) -> List[Dict[str, Any]]:
        """Extract JOIN conditions."""
        joins = []

        # Match JOIN table ON left = right
        join_pattern = (
            r"\bJOIN\s+(\w+)\s+ON\s+(?:\w+\.)?(\w+)\s*=\s*(?:\w+\.)?(\w+)"
        )

        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            right_table = match.group(1)
            left_col = match.group(2)
            right_col = match.group(3)

            joins.append(
                {
                    "left": left_col,
                    "right": f"{right_table}.{right_col}",
                }
            )

        return joins

    def _extract_group_by(self, sql: str) -> List[str]:
        """Extract GROUP BY columns."""
        group_by_match = re.search(
            r"\bGROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if not group_by_match:
            return []

        group_clause = group_by_match.group(1)
        columns = []

        for match in re.finditer(r"\b(?:\w+\.)?(\w+)\b", group_clause):
            col = match.group(1)
            if col.upper() not in ["AND", "OR"]:
                columns.append(col)

        return columns

    def _extract_order_by(self, sql: str) -> List[str]:
        """Extract ORDER BY columns."""
        order_by_match = re.search(
            r"\bORDER\s+BY\s+(.*?)$", sql, re.IGNORECASE | re.DOTALL
        )
        if not order_by_match:
            return []

        order_clause = order_by_match.group(1)
        columns = []

        # Extract columns, handling ASC/DESC
        for match in re.finditer(r"\b(?:\w+\.)?(\w+)\b", order_clause):
            col = match.group(1)
            if col.upper() not in ["ASC", "DESC"]:
                columns.append(col)

        return columns

    def log_query_execution(
        self,
        sql: str,
        runtime_ms: float,
        rows_returned: int,
        rows_scanned: Optional[int] = None,
        user_id: Optional[str] = None,
        layout_id: Optional[str] = None,
    ) -> int:
        """Parse and log a query execution."""
        parsed = self.parse_query(sql)

        ctx_res = extract_query_context_result(
            sql, dialect=self.config.sql_parser_dialect
        )
        context_key = ctx_res.context.key()
        cluster_id: Optional[str]
        if (
            ctx_res.parse_success
            and ctx_res.parse_confidence
            >= self.config.parse_confidence_threshold
        ):
            cluster_id = self.clusterer.cluster_id_for_context_key(context_key)
        else:
            # Exclude low-confidence parses from learning (no cluster assignment)
            cluster_id = None

        return self.metadata_store.log_query(
            table_name=parsed["table_name"],
            columns_used=parsed["columns_used"],
            predicates=parsed["predicates"],
            joins=parsed["joins"],
            group_by_cols=parsed["group_by_cols"],
            order_by_cols=parsed["order_by_cols"],
            runtime_ms=runtime_ms,
            rows_scanned=rows_scanned,
            rows_returned=rows_returned,
            query_text=sql,
            user_id=user_id,
            layout_id=layout_id,
            context_key=context_key,
            cluster_id=cluster_id,
            parse_success=ctx_res.parse_success,
            parse_confidence=ctx_res.parse_confidence,
            parser_version=ctx_res.parser_version,
        )
