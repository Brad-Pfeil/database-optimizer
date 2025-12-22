"""CLI reporting helpers (history tables, CSV/JSON export, report artifacts)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        # Accept ISO strings like "2025-12-21T04:48:24.432882"
        return datetime.fromisoformat(v)
    except ValueError:
        # Also accept sqlite-style "YYYY-mm-dd HH:MM:SS(.ffffff)"
        try:
            return datetime.fromisoformat(v.replace(" ", "T"))
        except ValueError:
            raise


def to_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)

    data = [[_fmt(r.get(c)) for c in columns] for r in rows]
    widths = [len(c) for c in columns]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "  ".join("-" * widths[i] for i in range(len(columns)))
    lines = [header, sep]
    for row in data:
        lines.append(
            "  ".join(row[i].ljust(widths[i]) for i in range(len(columns)))
        )
    return "\n".join(lines)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str) + "\n")


def write_csv(
    path: Path, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


@dataclass(frozen=True)
class ReportPaths:
    root: Path
    summary_md: Path
    eval_csv: Path
    eval_json: Path
    routing_csv: Path


def make_report_paths(out_dir: Path) -> ReportPaths:
    return ReportPaths(
        root=out_dir,
        summary_md=out_dir / "summary.md",
        eval_csv=out_dir / "layout_eval.csv",
        eval_json=out_dir / "layout_eval.json",
        routing_csv=out_dir / "query_routing_summary.csv",
    )


def summarize_markdown(
    *,
    table_name: str,
    eval_rows: list[dict[str, Any]],
    routing_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Optimiser report: {table_name}")
    lines.append("")

    # Choose best layout based on each layout's latest scored evaluation.
    latest_scored_by_layout: dict[str, dict[str, Any]] = {}
    for r in eval_rows:
        if r.get("reward_score") is None:
            continue
        lid = str(r.get("layout_id"))
        if lid not in latest_scored_by_layout:
            latest_scored_by_layout[lid] = r

    best = None
    if latest_scored_by_layout:
        best = max(
            latest_scored_by_layout.values(),
            key=lambda r: float(r["reward_score"]),
        )

    if best:
        lines.append("## Best scored layout (latest window)")
        lines.append("")
        lines.append(f"- layout_id: `{best.get('layout_id')}`")
        lines.append(f"- reward_score: `{best.get('reward_score')}`")
        lines.append(f"- avg_latency_ms: `{best.get('avg_latency_ms')}`")
        lines.append(f"- p95_latency_ms: `{best.get('p95_latency_ms')}`")
        lines.append(f"- p99_latency_ms: `{best.get('p99_latency_ms')}`")
        lines.append(f"- queries_evaluated: `{best.get('queries_evaluated')}`")
        lines.append(f"- eval_window_end: `{best.get('eval_window_end')}`")
        lines.append("")
    else:
        lines.append("## Best scored layout")
        lines.append("")
        lines.append("- No scored evaluations yet (reward_score is NULL).")
        lines.append("")

    lines.append("## Routing distribution (query_log)")
    lines.append("")
    if not routing_rows:
        lines.append("- No query logs found.")
        lines.append("")
        return "\n".join(lines) + "\n"

    # Keep top 20 for brevity.
    top = routing_rows[:20]
    cols = ["cluster_id", "layout_id", "query_count"]
    lines.append("```")
    lines.append(to_table(top, cols))
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"
