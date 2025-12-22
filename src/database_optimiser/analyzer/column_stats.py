"""Column usage statistics from query logs."""

from typing import Any, Dict


class ColumnStats:
    """Statistics about column usage in queries."""

    def __init__(
        self,
        table_name: str,
        column_name: str,
        filter_freq: int = 0,
        join_freq: int = 0,
        groupby_freq: int = 0,
        orderby_freq: int = 0,
        avg_selectivity: float = 1.0,
        total_queries: int = 0,
    ):
        """Initialize column statistics."""
        self.table_name = table_name
        self.column_name = column_name
        self.filter_freq = filter_freq
        self.join_freq = join_freq
        self.groupby_freq = groupby_freq
        self.orderby_freq = orderby_freq
        self.avg_selectivity = avg_selectivity
        self.total_queries = total_queries

    @property
    def filter_ratio(self) -> float:
        """Ratio of queries that filter on this column."""
        if self.total_queries == 0:
            return 0.0
        return self.filter_freq / self.total_queries

    @property
    def join_ratio(self) -> float:
        """Ratio of queries that join on this column."""
        if self.total_queries == 0:
            return 0.0
        return self.join_freq / self.total_queries

    @property
    def groupby_ratio(self) -> float:
        """Ratio of queries that group by this column."""
        if self.total_queries == 0:
            return 0.0
        return self.groupby_freq / self.total_queries

    @property
    def orderby_ratio(self) -> float:
        """Ratio of queries that order by this column."""
        if self.total_queries == 0:
            return 0.0
        return self.orderby_freq / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_name": self.table_name,
            "column_name": self.column_name,
            "filter_freq": self.filter_freq,
            "join_freq": self.join_freq,
            "groupby_freq": self.groupby_freq,
            "orderby_freq": self.orderby_freq,
            "avg_selectivity": self.avg_selectivity,
            "total_queries": self.total_queries,
            "filter_ratio": self.filter_ratio,
            "join_ratio": self.join_ratio,
            "groupby_ratio": self.groupby_ratio,
            "orderby_ratio": self.orderby_ratio,
        }
