"""CLI interface for the database optimizer."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import click

from ..adaptive.explorer import LayoutExplorer
from ..analyzer.workload_analyzer import WorkloadAnalyzer
from ..cli.reporting import (
    make_report_paths,
    parse_dt,
    summarize_markdown,
    write_csv,
    write_json,
)
from ..cli.ui import UI
from ..config import Config
from ..data.generator import DatasetGenerator
from ..evaluator.metrics import MetricsCalculator
from ..evaluator.reward import RewardCalculator
from ..evaluator.scheduler import EvaluationScheduler
from ..layout.generator import LayoutGenerator
from ..layout.migration_worker import MigrationWorker, new_job_id
from ..layout.migrator import LayoutMigrator
from ..query.executor import QueryExecutor
from ..storage.metadata import MetadataStore
from ..storage.query_logger import QueryLogger
from ..workload.runner import run_queries


@dataclass(frozen=True)
class ExplorerComponents:
    metadata_store: MetadataStore
    query_logger: QueryLogger
    query_executor: QueryExecutor
    workload_analyzer: WorkloadAnalyzer
    layout_generator: LayoutGenerator
    layout_migrator: LayoutMigrator
    metrics_calculator: MetricsCalculator
    reward_calculator: RewardCalculator
    explorer: LayoutExplorer


def _build_explorer_components(config: Config) -> ExplorerComponents:
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
    return ExplorerComponents(
        metadata_store=metadata_store,
        query_logger=query_logger,
        query_executor=query_executor,
        workload_analyzer=workload_analyzer,
        layout_generator=layout_generator,
        layout_migrator=layout_migrator,
        metrics_calculator=metrics_calculator,
        reward_calculator=reward_calculator,
        explorer=explorer,
    )


@click.group()
@click.option("--data-dir", default="data", help="Data directory")
@click.option(
    "--metadata-db", default="metadata.db", help="Metadata database path"
)
@click.pass_context
def cli(ctx, data_dir, metadata_db):
    """Adaptive Layout Generator - Self-rewriting database optimizer."""
    ctx.ensure_object(dict)
    config = Config(
        data_dir=Path(data_dir),
        metadata_db_path=Path(metadata_db),
    )
    config.ensure_dirs()
    ctx.obj["config"] = config
    ctx.obj["ui"] = UI.create()


@cli.command()
@click.option("--table-name", default="events", help="Table name")
@click.option(
    "--num-rows", default=100_000_000, help="Number of rows to generate"
)
@click.pass_context
def generate(ctx, table_name, num_rows):
    """Generate synthetic dataset."""
    config = ctx.obj["config"]
    generator = DatasetGenerator(config)

    click.echo(f"Generating {num_rows:,} rows for table '{table_name}'...")

    # Generate events table
    output_path = generator.generate_events_table(
        table_name=table_name,
        num_rows=num_rows,
    )

    click.echo(f"✓ Generated dataset at {output_path}")

    # Also generate customers table for joins
    if table_name == "events":
        customers_path = generator.generate_customers_table(
            table_name="customers",
            num_customers=1_000_000,
        )
        click.echo(f"✓ Generated customers table at {customers_path}")


@cli.command()
@click.option("--table-name", required=True, help="Table name")
@click.option(
    "--source-path",
    required=True,
    help="Path to source Parquet file(s) or directory",
)
@click.pass_context
def load_dataset(ctx, table_name, source_path):
    """Load an external Parquet dataset (Parquet file(s) or directory)."""
    config = ctx.obj["config"]
    generator = DatasetGenerator(config)

    click.echo(f"Loading dataset from {source_path}...")

    try:
        output_path = generator.load_external_dataset(
            table_name=table_name,
            source_path=source_path,
        )
        click.echo(f"✓ Dataset loaded at {output_path}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--table-name", default="events", help="Table name")
@click.option("--num-queries", default=100, help="Number of queries to run")
@click.option(
    "--workload",
    "workload_mode",
    type=click.Choice(["auto", "events", "schema"]),
    default="auto",
    show_default=True,
    help=(
        "Workload generator: "
        "events=events-like synthetic queries; "
        "schema=generic schema-driven queries; "
        "auto=events for table 'events', else schema."
    ),
)
@click.option(
    "--explore/--no-explore",
    default=True,
    help="Route queries across layouts using exploration/exploitation",
)
@click.option(
    "--exploration-rate",
    default=None,
    type=float,
    help="Override exploration rate for this run (0..1)",
)
@click.pass_context
def run_workload(
    ctx, table_name, num_queries, workload_mode, explore, exploration_rate
):
    """Run a query workload and log telemetry."""
    config = ctx.obj["config"]

    if exploration_rate is not None:
        config.exploration_rate = exploration_rate

    # Initialize components
    components = _build_explorer_components(config)
    metadata_store = components.metadata_store
    query_executor = components.query_executor
    explorer = components.explorer

    # Register tables
    ui: UI = ctx.obj["ui"]

    active_layout = metadata_store.get_active_layout(table_name)
    if active_layout:
        query_executor.register_layout(
            table_name,
            active_layout["layout_path"],
            active_layout["layout_id"],
        )
        ui.echo(
            f"Registered {table_name} table with active layout: {active_layout['layout_id']}"
        )
    else:
        events_path = config.data_dir / table_name / "layout_initial"
        if events_path.exists():
            query_executor.register_table(table_name, str(events_path))
            query_executor.current_layouts[table_name] = None
            ui.echo(f"Registered {table_name} table (initial layout)")
        else:
            ui.echo(f"✗ Error: No data found at {events_path}")
            query_executor.close()
            raise click.Abort()

    customers_path = config.data_dir / "customers" / "layout_initial"
    if customers_path.exists():
        query_executor.register_table("customers", str(customers_path))
        ui.echo("Registered customers table")

    # Generate queries
    generator = DatasetGenerator(config)
    if workload_mode == "auto":
        workload_mode = "events" if table_name == "events" else "schema"

    if workload_mode == "events":
        queries = generator.generate_workload_queries(
            table_name=table_name,
            num_queries=num_queries,
        )
    else:
        queries = generator.generate_schema_workload_queries(
            table_name=table_name,
            num_queries=num_queries,
            query_executor=query_executor,
        )

    ui.echo(f"Running {len(queries)} queries...")

    if ui.rich:
        prog = ui.progress(total=len(queries))
    else:
        prog = None

    if prog:
        prog.__enter__()

    def _on_progress(i: int, _total: int) -> None:
        if prog:
            if i:
                prog.advance(10 if i % 10 == 0 else 0)
            return
        if i % 10 == 0:
            ui.echo(f"  Processed {i}/{len(queries)} queries...")

    result = run_queries(
        table_name=table_name,
        queries=queries,
        query_executor=query_executor,
        metadata_store=metadata_store,
        explore=explore,
        explorer=explorer,
        on_progress=_on_progress,
    )

    ui.echo(
        f"✓ Completed: {result.success_count} successful, {result.error_count} errors"
    )

    if prog:
        prog.__exit__(None, None, None)

    query_executor.close()


@cli.command()
@click.option("--table-name", default="events", help="Table name")
@click.pass_context
def optimize(ctx, table_name):
    """Run optimization cycle for a table."""
    config = ctx.obj["config"]

    # Initialize all components
    components = _build_explorer_components(config)
    metadata_store = components.metadata_store
    query_executor = components.query_executor
    explorer = components.explorer

    # Register current table
    active_layout = metadata_store.get_active_layout(table_name)
    if active_layout:
        query_executor.register_layout(
            table_name,
            active_layout["layout_path"],
            active_layout["layout_id"],
        )
    else:
        initial_path = config.data_dir / table_name / "layout_initial"
        if initial_path.exists():
            query_executor.register_table(table_name, str(initial_path))
            # No layout_id for initial layout
            query_executor.current_layouts[table_name] = None

    click.echo(f"Optimizing table '{table_name}'...")

    # Run optimization
    result = explorer.optimize_table(table_name)

    if result["status"] == "success":
        click.echo("✓ Optimization successful!")
        click.echo(f"  Layout ID: {result['layout_id']}")
        evaluation = result.get("evaluation") or {}
        reward_score = evaluation.get("reward_score")
        latency_improvement = evaluation.get("latency_improvement", 0.0) or 0.0

        if reward_score is None:
            click.echo("  Reward score: (insufficient data)")
        else:
            click.echo(f"  Reward score: {reward_score:.4f}")

        click.echo(f"  Latency improvement: {latency_improvement * 100:.2f}%")
        click.echo(f"  Best layout: {result['best_layout']}")

        # Extra breakdown for debugging/observability
        breakdown = evaluation.get("reward_breakdown") or {}
        metrics = evaluation.get("metrics") or {}
        baseline_metrics = evaluation.get("baseline_metrics") or {}

        if breakdown:
            click.echo("  Reward breakdown:")
            click.echo(
                f"    Baseline queries: {baseline_metrics.get('queries_evaluated', 0)}"
            )
            click.echo(
                f"    New layout queries: {metrics.get('queries_evaluated', 0)}"
            )
            click.echo(
                f"    Baseline avg latency: {baseline_metrics.get('avg_latency_ms', 0.0):.2f} ms"
            )
            click.echo(
                f"    New avg latency: {metrics.get('avg_latency_ms', 0.0):.2f} ms"
            )
            click.echo(
                f"    Normalized rewrite cost: {breakdown.get('normalized_rewrite_cost', 0.0):.4f}"
            )
            click.echo(
                f"    Latency term (alpha*impr): {breakdown.get('latency_term', 0.0):.4f}"
            )
            click.echo(
                f"    Rewrite term (beta*cost): {breakdown.get('rewrite_term', 0.0):.4f}"
            )
            click.echo(
                f"    Rewrite cost sec: {evaluation.get('rewrite_cost_sec', 0.0):.2f}"
            )
    elif result["status"] == "no_new_layout":
        click.echo(f"ℹ {result['message']}")
    else:
        click.echo(f"✗ Error: {result.get('error', 'Unknown error')}")

    query_executor.close()


@cli.command()
@click.option("--table-name", default="events", help="Table name")
@click.pass_context
def status(ctx, table_name):
    """Show optimization status for a table."""
    config = ctx.obj["config"]

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

    # Get status
    status_info = explorer.get_exploration_status(table_name)
    workload_summary = workload_analyzer.get_workload_summary(table_name)

    active = metadata_store.get_active_layout(table_name)
    baseline_layout_id = active["layout_id"] if active else None

    ui: UI = ctx.obj["ui"]
    ui.rule(f"Status: {table_name}")
    header_rows = [
        {
            "active_layout": status_info["active_layout"],
            "num_layouts": status_info["num_arms"],
            "best_layout": status_info["best_arm"],
            "baseline": baseline_layout_id or "(initial)",
            "total_queries": workload_summary["total_queries"],
            "avg_ms": f"{workload_summary['avg_latency_ms']:.2f}",
            "p95_ms": f"{workload_summary['p95_latency_ms']:.2f}",
        }
    ]
    ui.table(
        title=None,
        columns=[
            "active_layout",
            "num_layouts",
            "best_layout",
            "baseline",
            "total_queries",
            "avg_ms",
            "p95_ms",
        ],
        rows=header_rows,
    )

    # Get all layouts and their evaluations
    all_layouts = metadata_store.get_all_layouts(table_name)
    if all_layouts:
        ui.rule("Layout performance")

        # Baseline for the "Improvement vs baseline" section is the initial
        # (layout_id NULL) queries.
        with metadata_store._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(1) AS c, AVG(runtime_ms) AS avg_ms
                FROM query_log
                WHERE table_name = ? AND layout_id IS NULL
                """,
                (table_name,),
            )
            row = cur.fetchone()
            baseline_count = (
                int(row["c"]) if row and row["c"] is not None else 0
            )
            baseline_latency = (
                float(row["avg_ms"])
                if row and row["avg_ms"] is not None
                else 0.0
            )

        perf_rows: list[dict[str, object]] = []
        for layout in all_layouts:
            layout_id = layout["layout_id"]
            is_active = layout.get("is_active", False)
            active_marker = " (ACTIVE)" if is_active else ""

            # Prefer the latest *scored* evaluation (reward_score not NULL).
            scored_eval = metadata_store.get_latest_scored_eval(layout_id)
            latest_eval = metadata_store.get_latest_evaluation(layout_id)

            with metadata_store._connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT COUNT(1) AS c, AVG(runtime_ms) AS avg_ms
                    FROM query_log
                    WHERE table_name = ? AND layout_id = ?
                    """,
                    (table_name, layout_id),
                )
                row = cur.fetchone()
                layout_count = (
                    int(row["c"]) if row and row["c"] is not None else 0
                )
                layout_latency = (
                    float(row["avg_ms"])
                    if row and row["avg_ms"] is not None
                    else 0.0
                )

            improvement = None
            if baseline_latency > 0 and layout_latency > 0:
                improvement = (
                    (baseline_latency - layout_latency) / baseline_latency
                ) * 100

            eval_data = scored_eval or latest_eval
            eval_status = None
            reward = None
            eval_end = None
            eval_avg = None
            eval_q = None
            if eval_data:
                reward = eval_data.get("reward_score")
                eval_status = (
                    "scored" if reward is not None else "insufficient_data"
                )
                eval_end = eval_data.get("eval_window_end")
                eval_avg = eval_data.get("avg_latency_ms")
                eval_q = eval_data.get("queries_evaluated")

            perf_rows.append(
                {
                    "layout_id": f"{layout_id}{active_marker}",
                    "queries": layout_count,
                    "avg_ms": f"{layout_latency:.2f}",
                    "impr_%": f"{improvement:+.2f}"
                    if improvement is not None
                    else "",
                    "eval_status": eval_status or "",
                    "reward": f"{reward:.4f}"
                    if isinstance(reward, float)
                    else "",
                    "eval_end": str(eval_end) if eval_end is not None else "",
                    "eval_q": eval_q if eval_q is not None else "",
                    "eval_avg": f"{eval_avg:.2f}"
                    if isinstance(eval_avg, float)
                    else "",
                }
            )

        ui.table(
            title=None,
            columns=[
                "layout_id",
                "queries",
                "avg_ms",
                "impr_%",
                "eval_status",
                "reward",
                "eval_q",
                "eval_avg",
                "eval_end",
            ],
            rows=perf_rows,
        )
        if baseline_count:
            ui.echo(
                f"Baseline for improvement: initial (layout_id NULL), queries={baseline_count}"
            )

    if status_info["arm_stats"]:
        click.echo("\nBandit Statistics:")
        for layout_id, stats in status_info["arm_stats"].items():
            click.echo(f"  {layout_id}:")
            click.echo(f"    Pulls: {stats['pulls']}")
            click.echo(f"    Mean reward: {stats['mean_reward']:.4f}")

    query_executor.close()


@cli.command()
@click.option("--table-name", required=True, help="Table name")
@click.option("--layout-id", default=None, help="Optional layout id to filter")
@click.option(
    "--cluster-id", default=None, help="Optional cluster id to filter"
)
@click.option(
    "--since",
    default=None,
    help="Only include evaluations with eval_window_end >= this timestamp (ISO)",
)
@click.option("--limit", default=50, type=int, help="Max rows to show")
@click.option(
    "--only-scored/--all",
    default=False,
    help="Only show scored evaluations (reward_score not NULL)",
)
@click.option(
    "--format",
    "out_format",
    default="table",
    type=click.Choice(["table", "json", "csv"]),
)
@click.pass_context
def history(
    ctx,
    table_name,
    layout_id,
    cluster_id,
    since,
    limit,
    only_scored,
    out_format,
):
    """Show evaluation history (time series) for a table/layout."""
    config = ctx.obj["config"]
    ui: UI = ctx.obj["ui"]
    metadata_store = MetadataStore(config)

    since_dt = None
    if since is not None:
        try:
            since_dt = parse_dt(since)
        except ValueError:
            click.echo(f"✗ Error: invalid --since value: {since}")
            raise click.Abort()

    rows = metadata_store.get_layout_eval_history(
        table_name=table_name,
        layout_id=layout_id,
        cluster_id=cluster_id,
        since=since_dt,
        limit=limit,
        only_scored=only_scored,
    )

    for r in rows:
        r["eval_status"] = (
            "scored"
            if r.get("reward_score") is not None
            else "insufficient_data"
        )

    columns = [
        "eval_window_end",
        "layout_id",
        "cluster_id",
        "eval_status",
        "reward_score",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "avg_rows_scanned",
        "queries_evaluated",
        "rewrite_cost_sec",
    ]

    if out_format == "json":
        click.echo(json.dumps(rows, indent=2, default=str))
        return

    if out_format == "csv":
        click.echo(",".join(columns))
        for r in rows:
            click.echo(
                ",".join(
                    "" if r.get(c) is None else str(r.get(c)) for c in columns
                )
            )
        return

    ui.rule(f"History: {table_name}")
    ui.table(title=None, columns=columns, rows=rows)


@cli.command()
@click.option("--table-name", required=True, help="Table name")
@click.option("--cluster-id", default=None, help="Optional cluster id")
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for report artifacts",
)
@click.option(
    "--since",
    default=None,
    help="Optional lower bound for history (ISO timestamp for eval_window_end/query ts)",
)
@click.pass_context
def report(ctx, table_name, cluster_id, out_dir, since):
    """Generate shareable report artifacts (Markdown + CSV/JSON)."""
    config = ctx.obj["config"]
    metadata_store = MetadataStore(config)

    since_dt = None
    if since is not None:
        try:
            since_dt = parse_dt(since)
        except ValueError:
            click.echo(f"✗ Error: invalid --since value: {since}")
            raise click.Abort()

    eval_rows = metadata_store.get_layout_eval_history(
        table_name=table_name,
        cluster_id=cluster_id,
        since=since_dt,
        limit=None,
        only_scored=False,
    )
    routing_rows = metadata_store.get_layout_query_counts(
        table_name=table_name,
        cluster_id=cluster_id,
        since=since_dt,
    )

    paths = make_report_paths(out_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    eval_cols = [
        "layout_id",
        "cluster_id",
        "eval_window_start",
        "eval_window_end",
        "reward_score",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "avg_rows_scanned",
        "queries_evaluated",
        "rewrite_cost_sec",
    ]
    routing_cols = ["cluster_id", "layout_id", "query_count"]

    write_json(paths.eval_json, eval_rows)
    write_csv(paths.eval_csv, eval_rows, eval_cols)
    write_csv(paths.routing_csv, routing_rows, routing_cols)

    md = summarize_markdown(
        table_name=table_name, eval_rows=eval_rows, routing_rows=routing_rows
    )
    paths.summary_md.write_text(md)

    ui: UI = ctx.obj["ui"]
    ui.echo(f"✓ Wrote report to {paths.root}")


@cli.command()
@click.option("--table-name", default="events", help="Table name")
@click.pass_context
def analyze(ctx, table_name):
    """Analyze workload and show column statistics."""
    config = ctx.obj["config"]

    metadata_store = MetadataStore(config)
    workload_analyzer = WorkloadAnalyzer(metadata_store)

    click.echo(f"Analyzing workload for table '{table_name}'...")

    column_stats = workload_analyzer.analyze_table(table_name)

    if not column_stats:
        click.echo("No query data found for this table.")
        return

    click.echo("\nColumn Statistics:")
    click.echo(
        f"{'Column':<20} {'Filter':<10} {'Join':<10} {'GroupBy':<10} {'OrderBy':<10} {'Selectivity':<12}"
    )
    click.echo("-" * 80)

    for col, stats in sorted(
        column_stats.items(), key=lambda x: x[1].filter_freq, reverse=True
    ):
        click.echo(
            f"{col:<20} {stats.filter_freq:<10} {stats.join_freq:<10} "
            f"{stats.groupby_freq:<10} {stats.orderby_freq:<10} {stats.avg_selectivity:<12.3f}"
        )

    # Show recommendations
    partition_candidates = workload_analyzer.get_partition_candidates(
        table_name
    )
    sort_candidates = workload_analyzer.get_sort_candidates(table_name)

    click.echo("\nPartition Candidates:")
    for col, score in partition_candidates[:5]:
        click.echo(f"  {col}: {score:.2f}")

    click.echo("\nSort Candidates:")
    for col in sort_candidates[:5]:
        click.echo(f"  {col}")


@cli.command()
@click.option("--table-name", required=True, help="Table name to evaluate")
@click.option(
    "--cluster-id", default=None, help="Optional cluster id (e.g. c0)"
)
@click.option(
    "--eval-window-hours",
    default=None,
    type=int,
    help="Rolling evaluation window size",
)
@click.option(
    "--min-new-queries",
    default=None,
    type=int,
    help="Min new queries since last eval to re-evaluate a layout",
)
@click.option(
    "--loop/--once", default=False, help="Run continuously (loop) or one-shot"
)
@click.option(
    "--interval-sec", default=None, type=int, help="Loop interval seconds"
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Show which layouts would be evaluated and why, without recording evals",
)
@click.option(
    "--explain/--no-explain",
    default=False,
    help="Print evaluation window and skip reasons",
)
@click.pass_context
def evaluate(
    ctx,
    table_name,
    cluster_id,
    eval_window_hours,
    min_new_queries,
    loop,
    interval_sec,
    dry_run,
    explain,
):
    """Evaluate layouts on a rolling window (Phase 5 scheduler)."""
    config = ctx.obj["config"]
    ui: UI = ctx.obj["ui"]
    metadata_store = MetadataStore(config)
    metrics_calculator = MetricsCalculator(metadata_store)
    reward_calculator = RewardCalculator(
        metrics_calculator, config, metadata_store
    )
    scheduler = EvaluationScheduler(config, metadata_store, reward_calculator)

    sleep_s = (
        interval_sec
        if interval_sec is not None
        else config.eval_scheduler_interval_sec
    )

    if not loop:
        if dry_run or explain:
            from datetime import timedelta

            end_time = scheduler._now()
            window_hours = eval_window_hours or config.eval_window_hours
            start_time = end_time - timedelta(hours=window_hours)
            min_new = (
                min_new_queries
                if min_new_queries is not None
                else config.eval_scheduler_min_new_queries
            )
            active = metadata_store.get_active_layout(table_name)
            baseline_layout_id = active["layout_id"] if active else None

            ui.rule("Evaluate plan")
            ui.echo(
                f"Evaluate plan for table={table_name} "
                f"cluster={cluster_id or '(all)'} "
                f"window_hours={window_hours} "
                f"min_new_queries={min_new}"
            )
            ui.echo(
                "Baseline layout: "
                + (baseline_layout_id if baseline_layout_id else "(initial)")
            )
            ui.echo(
                f"Window: start={start_time.isoformat()} end={end_time.isoformat()}"
            )

            layouts = (
                metadata_store.get_all_layouts(
                    table_name, cluster_id=cluster_id
                )
                if cluster_id
                else metadata_store.get_all_layouts(table_name)
            )
            for layout in layouts:
                lid = layout["layout_id"]
                if baseline_layout_id and lid == baseline_layout_id:
                    ui.echo(f"- {lid}: skip (baseline)")
                    continue

                latest = metadata_store.get_latest_evaluation(lid)
                last_end = latest["eval_window_end"] if latest else None
                if last_end is not None:
                    new_q = scheduler._count_queries_in_window(
                        table_name=table_name,
                        layout_id=lid,
                        cluster_id=cluster_id or layout.get("cluster_id"),
                        start_time=last_end,
                        end_time=end_time,
                    )
                    if new_q < min_new:
                        ui.echo(
                            f"- {lid}: skip (new_queries={new_q} < {min_new})"
                        )
                        continue
                    ui.echo(f"- {lid}: evaluate (new_queries={new_q})")
                else:
                    ui.echo(f"- {lid}: evaluate (no prior eval)")

            if dry_run:
                return

        n = scheduler.evaluate_table(
            table_name=table_name,
            cluster_id=cluster_id,
            eval_window_hours=eval_window_hours,
            min_new_queries=min_new_queries,
        )
        ui.echo(f"Recorded {n} evaluation(s).")
        return

    import time

    ui.rule("Evaluation loop")
    ui.echo(
        f"Starting evaluation loop for table={table_name} cluster={cluster_id or '(all)'} every {sleep_s}s"
    )
    while True:
        n = scheduler.evaluate_table(
            table_name=table_name,
            cluster_id=cluster_id,
            eval_window_hours=eval_window_hours,
            min_new_queries=min_new_queries,
        )
        if n:
            ui.echo(
                f"[{datetime.now(timezone.utc).isoformat()}] Recorded {n} evaluation(s)."
            )
        time.sleep(sleep_s)


@cli.command(name="enqueue-migration")
@click.option("--table-name", required=True, help="Table name")
@click.option(
    "--mode", type=click.Choice(["full", "incremental"]), default="full"
)
@click.option("--cluster-id", default=None, help="Optional cluster id")
@click.pass_context
def enqueue_migration(ctx, table_name, mode, cluster_id):
    """Enqueue a migration job that creates a new layout in the background (Phase 6)."""
    config = ctx.obj["config"]
    components = _build_explorer_components(config)
    metadata_store = components.metadata_store
    layout_migrator = components.layout_migrator
    explorer = components.explorer

    proposed = explorer.propose_new_layout(
        table_name, window_hours=config.analysis_window_hours
    )
    if not proposed:
        click.echo("No novel layout candidate found to migrate.")
        return
    layout_id, spec = proposed

    src = layout_migrator.get_source_path(table_name)
    if not src:
        raise click.ClickException(f"No source path for table {table_name}")
    total_files = len(layout_migrator.list_parquet_files(src))

    jid = new_job_id()
    metadata_store.enqueue_migration_job(
        job_id=jid,
        table_name=table_name,
        layout_id=layout_id,
        mode=mode,
        requested_spec_json=json.dumps(spec.to_dict()),
        cluster_id=cluster_id,
        total_files=total_files,
    )
    click.echo(
        f"Enqueued job {jid} for layout {layout_id} mode={mode} (files={total_files})"
    )


@cli.command(name="migration-worker")
@click.option(
    "--once/--loop",
    default=False,
    help="Run one job then exit, or loop forever",
)
@click.option(
    "--sleep-sec", default=2, type=int, help="Loop sleep seconds when idle"
)
@click.option(
    "--batch-files",
    default=None,
    type=int,
    help="Files per batch (incremental mode)",
)
@click.pass_context
def migration_worker(ctx, once, sleep_sec, batch_files):
    """Run the migration worker that processes queued jobs (Phase 6)."""
    config = ctx.obj["config"]
    metadata_store = MetadataStore(config)
    layout_migrator = LayoutMigrator(config, metadata_store)
    worker = MigrationWorker(
        config=config,
        metadata_store=metadata_store,
        migrator=layout_migrator,
        batch_files=(
            batch_files
            if batch_files is not None
            else config.migration_batch_files
        ),
    )

    if once:
        did = worker.run_once()
        click.echo("Processed a job." if did else "No queued jobs.")
        return
    click.echo("Starting migration worker loop.")
    worker.run_loop(sleep_sec=sleep_sec)


if __name__ == "__main__":
    cli()
