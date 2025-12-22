"""Layout specification data structures."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LayoutSpec:
    """Specification for a data layout."""

    partition_cols: Optional[List[str]] = None
    sort_cols: Optional[List[str]] = None
    index_strategy: Optional[Dict[str, Any]] = None
    compression: Optional[Dict[str, Any]] = None
    file_size_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partition_cols": self.partition_cols,
            "sort_cols": self.sort_cols,
            "index_strategy": self.index_strategy,
            "compression": self.compression,
            "file_size_mb": self.file_size_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutSpec":
        """Create from dictionary."""
        return cls(
            partition_cols=data.get("partition_cols"),
            sort_cols=data.get("sort_cols"),
            index_strategy=data.get("index_strategy"),
            compression=data.get("compression"),
            file_size_mb=data.get("file_size_mb"),
        )

    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.partition_cols:
            parts.append(f"partition={','.join(self.partition_cols)}")
        if self.sort_cols:
            parts.append(f"sort={','.join(self.sort_cols)}")
        return "|".join(parts) if parts else "default"
