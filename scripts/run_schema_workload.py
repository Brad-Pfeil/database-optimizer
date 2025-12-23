#!/usr/bin/env python3
"""Script to run schema-driven workload on a loaded dataset.

This is a thin wrapper around:
  dbopt run-workload --workload schema
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database_optimiser.cli.main import cli


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run schema-driven workload on a loaded dataset"
    )
    parser.add_argument("--table-name", default="dataset", help="Table name")
    parser.add_argument(
        "--num-queries", type=int, default=100, help="Number of queries"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--metadata-db", default="metadata.db", help="Metadata database"
    )
    parser.add_argument(
        "--explore/--no-explore",
        default=True,
        help="Enable/disable exploration routing across layouts",
    )
    parser.add_argument(
        "--exploration-rate",
        type=float,
        default=None,
        help="Override exploration rate (0..1)",
    )
    args = parser.parse_args()

    cmd = [
        "--data-dir",
        args.data_dir,
        "--metadata-db",
        args.metadata_db,
        "run-workload",
        "--table-name",
        args.table_name,
        "--num-queries",
        str(args.num_queries),
        "--workload",
        "schema",
        "--explore" if args.explore else "--no-explore",
    ]
    if args.exploration_rate is not None:
        cmd += ["--exploration-rate", str(args.exploration_rate)]

    cli(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
