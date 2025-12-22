# Adaptive Layout Generator

A self-rewriting database system that observes query workloads and automatically optimizes data layouts (partitioning, sorting) to minimize query latency and cost over time.

## Overview

This project implements an adaptive database optimizer that:

- **Observes** query workloads and extracts patterns
- **Proposes** new data layouts based on workload analysis
- **Migrates** data to optimized layouts (partitioning, sorting)
- **Evaluates** performance improvements
- **Learns** from feedback using multi-armed bandit algorithms

Think: AutoML, but for physical data design.

## Architecture

### Components

1. **Storage Layer**: SQLite metadata store for query logs, layouts, and evaluations
2. **Query Executor**: DuckDB wrapper that logs all query telemetry
3. **Workload Analyzer**: Aggregates query patterns and computes column statistics
4. **Layout Generator**: Heuristic algorithm that proposes layouts based on workload
5. **Layout Migrator**: Rewrites Parquet datasets with new layouts
6. **Evaluator**: Compares performance and calculates reward scores
7. **Adaptive Explorer**: Multi-armed bandit for layout exploration/exploitation

### Data Flow

```
Queries → Query Logger → Metadata Store
                              ↓
                    Workload Analyzer
                              ↓
                    Layout Generator
                              ↓
                    Layout Migrator → New Parquet Files
                              ↓
                    Evaluator → Reward Calculator
                              ↓
                    Multi-Armed Bandit → Next Layout
```

## Installation

```bash
# Recommended: install the CLI as a real tool (production-style)
uv tool install -e .

# Verify
dbopt --help
```

## Usage

### Quickstart (command overview)

```bash
# Generate synthetic dataset
dbopt generate --table-name events --num-rows 1000000

# Run query workload (routes queries across layouts if exploration is enabled)
dbopt run-workload --table-name events --num-queries 1000 --explore --exploration-rate 0.3

# Analyze workload
dbopt analyze --table-name events

# Propose + migrate a new layout (full rewrite by default; async job + worker)
dbopt enqueue-migration --table-name events --mode full
dbopt migration-worker --loop --sleep-sec 2

# Phase 5: scheduled evaluation (rolling window)
dbopt evaluate --table-name events --once
dbopt evaluate --table-name events --loop --interval-sec 60

# Status / diagnostics
dbopt status --table-name events
dbopt history --table-name events --limit 50 --format table

# Shareable report artifacts (Markdown + CSV/JSON)
dbopt report --table-name events --out reports/run1
```

### Rich terminal UI (and plain fallback)

When run in an interactive terminal, the CLI uses Rich tables and progress indicators.

- Force plain output (useful for logs/CI/piping):

```bash
DBOPT_PLAIN=1 dbopt status --table-name events
```

### Example workflow (synthetic dataset, end-to-end)

This is the recommended “happy path” to see the optimiser learn.

#### 0) Install the CLI

```bash
cd /path/to/database_optimiser
uv tool install -e .
dbopt --help
```

#### 1) Generate a dataset

```bash
dbopt generate --table-name events --num-rows 200000
```

Expected:
- `data/events/layout_initial/` is created
- `data/customers/layout_initial/` is created (for join queries)

#### 2) Run a routed workload (collect telemetry)

```bash
dbopt run-workload --table-name events --num-queries 2000 --explore --exploration-rate 0.3
```

Expected:
- `metadata.db` gets populated with `query_log` rows
- queries are routed across existing layouts (initially mostly “initial”)

#### 3) Create a new candidate layout (async migration)

```bash
dbopt enqueue-migration --table-name events --mode full
dbopt migration-worker --loop --sleep-sec 2
```

Expected:
- a new layout directory appears under `data/events/layout_<id>/`
- the worker prints partition/sort decisions and completes

Tip: you can stop the worker with Ctrl+C once it finishes the queued job(s).

#### 4) Run more routed workload (gather samples for the new layout)

```bash
dbopt run-workload --table-name events --num-queries 2000 --explore --exploration-rate 0.3
```

Expected:
- the new layout starts receiving some queries (exploration traffic)

#### 5) Evaluate layouts (rolling window) and inspect status

```bash
dbopt evaluate --table-name events --once
dbopt status --table-name events
```

Expected:
- `evaluate` records `layout_eval` rows
- `status` shows per-layout: query counts, avg latency, improvement vs baseline, and reward score
- one layout should start to emerge as “Best layout” as samples accumulate

#### 6) Inspect learning history + export a report

```bash
dbopt history --table-name events --limit 50 --format table
dbopt history --table-name events --format json --only-scored > eval_history.json
dbopt report --table-name events --out reports/events_run1
```

Expected:
- `history` shows a time series of evaluations (reward, latency, rows scanned, window end timestamps)
- `report` writes:
  - `reports/events_run1/summary.md`
  - `reports/events_run1/layout_eval.csv` and `layout_eval.json`
  - `reports/events_run1/query_routing_summary.csv`

### Example workflow (NYC taxi dataset)

This workflow is the same loop, but starts from real data. The tooling still stays generic; NYC taxi is just a convenient example dataset.

#### 1) Download or load the dataset

```bash
dbopt download-nyc-taxi --table-name nyc_taxi --year 2023 --month 1 --color yellow
```

Or load an existing Parquet dataset:

```bash
dbopt load-dataset --table-name nyc_taxi --source-path /path/to/parquet_dir
```

#### 2) Run routed workload

```bash
dbopt run-nyc-taxi-workload --table-name nyc_taxi --num-queries 2000 --explore --exploration-rate 0.3
```

#### 3) Create new layouts + evaluate

```bash
dbopt enqueue-migration --table-name nyc_taxi --mode incremental
dbopt migration-worker --loop --sleep-sec 2
dbopt evaluate --table-name nyc_taxi --once
dbopt status --table-name nyc_taxi
```

### Debugging and “why didn’t it score?”

If you see `reward_score` missing / “insufficient data”, it usually means the evaluator didn’t have enough samples in the evaluation window.

Use:

```bash
dbopt evaluate --table-name events --explain --dry-run
```

This will print:
- the evaluation window bounds
- the baseline layout used (`(initial)` if no active layout is set)
- which layouts would be evaluated or skipped (and why)

### Python API

```python
from database_optimiser.config import Config
from database_optimiser.storage.metadata import MetadataStore
from database_optimiser.query.executor import QueryExecutor
from database_optimiser.adaptive.explorer import LayoutExplorer

# Initialize
config = Config()
metadata_store = MetadataStore(config)
query_executor = QueryExecutor(config, metadata_store, query_logger)

# Run queries (automatically logged)
result = query_executor.run_query("SELECT * FROM events WHERE event_date > '2024-01-01'")

# Optimize
explorer = LayoutExplorer(...)
result = explorer.optimize_table("events")
```

## Features

### Workload Analysis

- Tracks column usage (filters, joins, group by, order by)
- Computes selectivity estimates
- Identifies "hot" columns for optimization

### Layout Generation

- **Partitioning**: Automatically chooses partition keys based on filter frequency
- **Sorting**: Selects sort keys based on filter + group by patterns
- Heuristic algorithm prioritizes time columns and high-frequency filters

### Adaptive Learning

- **Multi-Armed Bandit**: UCB1 and Thompson Sampling algorithms
- **Exploration/Exploitation**: 80% traffic to best layout, 20% to experiments
- **Reward Function**: `reward = α * latency_improvement - β * rewrite_cost`

### Workload clustering (generic)

- Queries are assigned a **stable cluster_id** derived from their query-shape signature.
- Layout pools can be scoped per cluster to avoid one global layout dominating heterogeneous workloads.

### Online evaluation (Phase 5)

- `evaluate` runs a rolling-window evaluation loop that records `layout_eval` rows when there are enough new samples.
- This decouples learning from manual `optimize` runs.

### Async / incremental rewrites (Phase 6)

- Migration jobs are queued in SQLite (`migration_job`) and processed by a worker.
- Incremental mode rewrites datasets in batches of parquet files to reduce latency spikes.

## Extension points

This repo is designed to stay dataset-agnostic. The easiest places to swap implementations are:

- **Context extraction**: `src/database_optimiser/query/context.py`
- **Clustering**: `src/database_optimiser/analyzer/workload_clusterer.py`
- **Routing policy**: `src/database_optimiser/adaptive/contextual_bandit.py`
- **Candidate generation**: `src/database_optimiser/layout/generator.py`
- **Migration**: `src/database_optimiser/layout/migrator.py` and `src/database_optimiser/layout/migration_worker.py`

### Evaluation

- Compares query performance before/after layout changes
- Calculates latency improvements (avg, p95, p99)
- Tracks rows scanned and query counts

## Project Structure

```
database_optimiser/
├── src/database_optimiser/
│   ├── storage/          # Metadata storage (SQLite)
│   ├── query/            # Query execution and logging
│   ├── data/             # Dataset generation
│   ├── analyzer/         # Workload analysis
│   ├── layout/           # Layout generation and migration
│   ├── evaluator/        # Performance evaluation
│   ├── adaptive/         # Multi-armed bandit
│   └── cli/              # Command-line interface
├── scripts/              # Utility scripts
├── tests/                # Unit tests
└── benchmarks/           # Performance benchmarks
```

## Configuration

Configuration is managed via `Config` class or environment variables:

- `DATA_DIR`: Data directory (default: "data")
- `METADATA_DB`: Metadata database path (default: "metadata.db")
- `EXPLORATION_RATE`: Traffic fraction for exploration (default: 0.2)
- `ALPHA`: Weight for latency improvement in reward (default: 1.0)
- `BETA`: Weight for rewrite cost in reward (default: 0.1)

## Note on examples (deduplication)

The end-to-end workflows above are the canonical examples. They cover the same steps as older
`python -m ...` examples, but using the production-style CLI (`dbopt`) and including the new
observability commands (`history`, `report`).

## Testing

Run the integration test:

```bash
python scripts/integration_test.py
```

This will:
- Generate a test dataset
- Run a query workload
- Analyze the workload
- Propose and evaluate a new layout
- Show optimization results

## Future Enhancements

- [ ] Incremental layout rewrites (instead of full rewrites)
- [ ] Support for more layout features (compression, indexing)
- [ ] Distributed execution (Spark integration)
- [ ] Real-time query routing
- [ ] Advanced RL algorithms (DQN, PPO)
- [ ] Cost-aware optimization (storage costs, compute costs)

## References

- Multi-Armed Bandit: UCB1, Thompson Sampling
- Data Layout Optimization: Partitioning, Sorting, Clustering
- Adaptive Systems: Online learning, exploration/exploitation

