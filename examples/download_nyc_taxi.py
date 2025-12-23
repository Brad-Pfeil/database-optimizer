#!/usr/bin/env python3
"""Example: download NYC taxi parquet and store it under data/<table>/layout_initial/.

This lives in `examples/` on purpose: it is dataset-specific and not part of the
core optimiser API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NYC taxi parquet (yellow/green) into a dbopt dataset directory."
    )
    parser.add_argument(
        "--table-name", default="nyc_taxi", help="Destination table name"
    )
    parser.add_argument(
        "--data-dir", default="data", help="dbopt data directory"
    )
    parser.add_argument("--year", type=int, default=2023, help="Year")
    parser.add_argument("--month", type=int, default=1, help="Month (1..12)")
    parser.add_argument(
        "--color",
        default="yellow",
        choices=["yellow", "green"],
        help="Taxi color",
    )
    args = parser.parse_args()

    url = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"{args.color}_tripdata_{args.year}-{args.month:02d}.parquet"
    )

    out_dir = Path(args.data_dir) / args.table_name / "layout_initial"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "part-0.parquet"

    conn = duckdb.connect()
    try:
        conn.execute(
            """
            COPY (SELECT * FROM read_parquet(?))
            TO ? (FORMAT PARQUET)
            """,
            (url, str(out_file)),
        )
    finally:
        conn.close()

    print(
        f"✓ Downloaded {args.color} {args.year}-{args.month:02d} to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
