"""Small UI layer for CLI output.

Uses Rich when available and output is a TTY; falls back to plain text otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import click

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
except Exception:  # pragma: no cover - optional at runtime
    Console = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


def _wants_plain() -> bool:
    v = os.environ.get("DBOPT_PLAIN", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _is_tty() -> bool:
    try:
        return bool(click.get_text_stream("stdout").isatty())
    except Exception:
        return False


@dataclass(frozen=True)
class UI:
    rich: bool
    _console: Any

    @staticmethod
    def create() -> "UI":
        enable = (not _wants_plain()) and _is_tty() and Console is not None
        if enable:
            return UI(rich=True, _console=Console())
        return UI(rich=False, _console=None)

    def echo(self, text: str = "") -> None:
        if self.rich:
            self._console.print(text)
        else:
            click.echo(text)

    def rule(self, title: str) -> None:
        if self.rich:
            self._console.rule(title)
        else:
            click.echo(f"\n== {title} ==")

    def table(
        self,
        *,
        title: Optional[str],
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            self.echo("(no rows)")
            return

        if self.rich and Table is not None:
            t = Table(title=title, show_lines=False)
            for c in columns:
                t.add_column(c, overflow="fold")
            for r in rows:
                t.add_row(*(self._fmt_cell(r.get(c)) for c in columns))
            self._console.print(t)
            return

        # Plain fallback
        from .reporting import to_table

        if title:
            self.echo(title)
        self.echo(to_table(rows, columns))

    def _fmt_cell(self, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)

    def progress(self, total: int) -> "ProgressUI":
        if self.rich and Progress is not None:
            p = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
            )
            return ProgressUI(
                rich=True, _progress=p, _task_id=None, total=total
            )
        return ProgressUI(
            rich=False, _progress=None, _task_id=None, total=total
        )


@dataclass
class ProgressUI:
    rich: bool
    _progress: Any
    _task_id: Any
    total: int

    def __enter__(self) -> "ProgressUI":
        if self.rich:
            self._progress.__enter__()
            self._task_id = self._progress.add_task(
                "Running", total=self.total
            )
        return self

    def advance(self, n: int = 1, description: Optional[str] = None) -> None:
        if not self.rich:
            return
        if description is not None:
            self._progress.update(self._task_id, description=description)
        self._progress.advance(self._task_id, n)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.rich:
            self._progress.__exit__(exc_type, exc, tb)
