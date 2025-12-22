"""End-to-end integration test for the adaptive layout generator."""

import sys
from pathlib import Path

# Add src to path
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


def run_integration_test():
    """Run end-to-end integration test."""
    print("=" * 60)
    print("Adaptive Layout Generator - Integration Test")
    print("=" * 60)

    # Initialize config
    config = Config(
        data_dir=Path("test_data"),
        metadata_db_path=Path("test_metadata.db"),
    )
    config.ensure_dirs()

    # Clean up test database if it exists
    if config.metadata_db_path.exists():
        config.metadata_db_path.unlink()

    table_name = "events"

    # Step 1: Generate dataset
    print("\n[1/5] Generating synthetic dataset...")
    generator = DatasetGenerator(config)
    events_path = generator.generate_events_table(
        table_name=table_name,
        num_rows=1_000_000,  # Smaller for faster testing
    )
    print(f"✓ Generated {table_name} table at {events_path}")

    # Step 2: Initialize components
    print("\n[2/5] Initializing components...")
    metadata_store = MetadataStore(config)
    query_logger = QueryLogger(metadata_store, config)
    query_executor = QueryExecutor(config, metadata_store, query_logger)

    # Register table
    query_executor.register_table(table_name, str(events_path))
    print(f"✓ Registered {table_name} table")

    # Step 3: Run workload
    print("\n[3/5] Running query workload...")
    queries = generator.generate_workload_queries(
        table_name=table_name,
        num_queries=50,  # Smaller for faster testing
    )

    success_count = 0
    for i, query in enumerate(queries):
        try:
            query_executor.run_query(query.strip())
            success_count += 1
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(queries)} queries...")
        except Exception as e:
            print(f"  Error in query {i + 1}: {str(e)}")

    print(f"✓ Completed {success_count}/{len(queries)} queries successfully")

    # Step 4: Analyze workload
    print("\n[4/5] Analyzing workload...")
    workload_analyzer = WorkloadAnalyzer(metadata_store)
    column_stats = workload_analyzer.analyze_table(table_name)

    print(f"✓ Analyzed {len(column_stats)} columns")
    print("  Top columns by filter frequency:")
    for col, stats in sorted(
        column_stats.items(), key=lambda x: x[1].filter_freq, reverse=True
    )[:5]:
        print(f"    {col}: {stats.filter_freq} filters")

    # Step 5: Optimize
    print("\n[5/5] Running optimization...")
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

    # Generate and evaluate a new layout
    proposal = explorer.propose_new_layout(table_name)
    if proposal:
        layout_id, layout_spec = proposal
        print(f"  Proposed layout: {layout_spec}")

        try:
            eval_result = explorer.create_and_evaluate_layout(
                table_name,
                layout_id,
                layout_spec,
            )
            print("✓ Optimization complete!")
            print(f"  Layout ID: {layout_id}")
            print(f"  Reward score: {eval_result['reward_score']:.4f}")
            print(
                f"  Latency improvement: {eval_result['latency_improvement'] * 100:.2f}%"
            )
        except Exception as e:
            print(f"✗ Optimization failed: {str(e)}")
            import traceback

            traceback.print_exc()
    else:
        print("  No new layout proposed")

    # Show final status
    print("\n" + "=" * 60)
    print("Final Status:")
    print("=" * 60)
    status = explorer.get_exploration_status(table_name)
    print(f"Active layout: {status['active_layout']}")
    print(f"Number of layouts: {status['num_arms']}")
    print(f"Best layout: {status['best_arm']}")

    query_executor.close()

    print("\n✓ Integration test completed!")
    print(f"Test data: {config.data_dir}")
    print(f"Test metadata: {config.metadata_db_path}")


if __name__ == "__main__":
    run_integration_test()
