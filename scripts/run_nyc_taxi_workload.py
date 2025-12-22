#!/usr/bin/env python3
"""Script to run workload on NYC taxi data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database_optimiser.adaptive.explorer import LayoutExplorer
from database_optimiser.analyzer.workload_analyzer import WorkloadAnalyzer
from database_optimiser.config import Config
from database_optimiser.data.generator import DatasetGenerator
from database_optimiser.evaluator.metrics import MetricsCalculator
from database_optimiser.evaluator.reward import RewardCalculator
from database_optimiser.layout.generator import LayoutGenerator
from database_optimiser.layout.migrator import LayoutMigrator
from database_optimiser.query.executor import QueryExecutor
from database_optimiser.storage.metadata import MetadataStore
from database_optimiser.storage.query_logger import QueryLogger
from database_optimiser.workload.runner import run_queries


def main():
    """Run workload on NYC taxi data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run query workload on NYC taxi data"
    )
    parser.add_argument("--table-name", default="nyc_taxi", help="Table name")
    parser.add_argument(
        "--num-queries", type=int, default=100, help="Number of queries"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--metadata-db", default="metadata.db", help="Metadata database"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Enable exploration routing across layouts",
    )
    parser.add_argument(
        "--exploration-rate",
        type=float,
        default=None,
        help="Override exploration rate (0..1)",
    )

    args = parser.parse_args()

    # Initialize
    config = Config(
        data_dir=Path(args.data_dir),
        metadata_db_path=Path(args.metadata_db),
    )
    config.ensure_dirs()
    if args.exploration_rate is not None:
        config.exploration_rate = args.exploration_rate

    metadata_store = MetadataStore(config)
    query_logger = QueryLogger(metadata_store, config)
    query_executor = QueryExecutor(config, metadata_store, query_logger)

    workload_analyzer = WorkloadAnalyzer(metadata_store)
    layout_generator = LayoutGenerator(workload_analyzer, config)
    layout_migrator = LayoutMigrator(config, metadata_store)
    metrics_calculator = MetricsCalculator(metadata_store)
    reward_calculator = RewardCalculator(
        metrics_calculator, config, metadata_store
    )
    explorer = LayoutExplorer(
        config,
        metadata_store,
        layout_generator,
        layout_migrator,
        reward_calculator,
        query_executor,
    )

    # Register table
    taxi_path = config.data_dir / args.table_name / "layout_initial"
    if not taxi_path.exists():
        print(f"Error: No data found at {taxi_path}")
        print("Please download NYC taxi data first:")
        print(
            f"  uv run main.py download-nyc-taxi --table-name {args.table_name}"
        )
        return 1

    # Prefer active layout if present
    active_layout = metadata_store.get_active_layout(args.table_name)
    if active_layout:
        query_executor.register_layout(
            args.table_name,
            active_layout["layout_path"],
            active_layout["layout_id"],
        )
        print(
            f"Registered {args.table_name} table with active layout: {active_layout['layout_id']}"
        )
    else:
        query_executor.register_table(args.table_name, str(taxi_path))
        query_executor.current_layouts[args.table_name] = None
        print(f"Registered {args.table_name} table (initial layout)")

    # Generate and run queries
    generator = DatasetGenerator(config)
    queries = generator.generate_nyc_taxi_queries(
        table_name=args.table_name,
        num_queries=args.num_queries,
    )

    print(f"Running {len(queries)} queries...")
    print(
        f"Exploration routing: {'enabled' if args.explore else 'disabled'} (exploration_rate={config.exploration_rate})"
    )

    shown_errors = 0

    def _on_error(i: int, e: Exception) -> None:
        nonlocal shown_errors
        if shown_errors < 5:
            shown_errors += 1
            print(f"  Error in query {i}: {str(e)}")

    def _on_progress(i: int, _total_processed: int) -> None:
        if i % 10 == 0:
            print(f"  Processed {i}/{len(queries)} queries...")

    result = run_queries(
        table_name=args.table_name,
        queries=queries,
        query_executor=query_executor,
        metadata_store=metadata_store,
        explore=args.explore,
        explorer=explorer,
        on_error=_on_error,
        on_progress=_on_progress,
    )

    print(
        f"✓ Completed: {result.success_count} successful, {result.error_count} errors"
    )

    query_executor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
