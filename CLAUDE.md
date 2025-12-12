# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Model Context Protocol (MCP) server that provides intelligent analysis of trading strategy backtest results stored in a SQLite database. It enables Claude Code to perform sophisticated strategy research, cross-period robustness analysis, and benchmark comparisons.

## Prerequisites

- Python 3.10+
- Access to strategy_research.db (path set via `STRATEGY_DB_PATH` environment variable)

## Environment Variables

```bash
# Required: Path to the SQLite strategy research database
export STRATEGY_DB_PATH=/path/to/cache/strategy_research.db

# Example typical location (from trading-lab project)
export STRATEGY_DB_PATH=/Volumes/Work/development/projects/git/trading-lab/cache/strategy_research.db
```

## Development Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register with Claude Code (user level - available in all projects)
claude mcp add -s user strategy-research-db \
    '/Volumes/Work/development/projects/git/mcp-strategy-research-db/venv/bin/python3' \
    '/Volumes/Work/development/projects/git/mcp-strategy-research-db/strategy_research_mcp_server.py'

# Verify registration
claude mcp list

# Debug mode for troubleshooting
claude --mcp-debug
```

## Architecture

The server (`strategy_research_mcp_server.py`) is a single-file MCP server using `mcp.server` with direct SQLite queries for maximum flexibility.

### Tools (17 total)

**Database Overview:**
- `get_database_status`: Overview statistics (runs, strategies, symbols)
- `list_search_runs`: List backtest runs with filters
- `get_run_details`: Details for a specific run

**Strategy Analysis:**
- `get_top_strategies`: Ranked strategies with customizable sorting
- `get_strategy_details`: Full details for a strategy
- `compare_strategy_across_periods`: Cross-period performance analysis

**Robustness Analysis:**
- `find_robust_strategies`: Strategies that work across ALL periods
- `get_benchmark_comparison`: Compare strategies vs Buy & Hold

**Benchmark Comparison:**
- `find_alpha_generators`: Strategies beating Buy & Hold
- `get_risk_adjusted_rankings`: Calmar ratio rankings

**Symbol Analysis:**
- `get_strategy_symbol_breakdown`: Per-symbol performance breakdown
- `get_signal_performance_summary`: Aggregate performance by signal type

**Capital Deployment Analysis:**
- `get_capital_deployment_analysis`: Portfolio capital utilization across periods
- `get_daily_position_counts`: Exact daily position counts (requires Dec 2025+ data)
- `compare_timing_modes`: Compare Conservative vs Aggressive timing

**Advanced:**
- `run_custom_query`: Custom SQL (read-only)
- `get_schema`: Database schema documentation
- `list_strategy_ids`: List strategy IDs with pattern filtering

### Resources

- `strategy://status`: Database status and statistics
- `strategy://schema`: Database schema information

## Database Schema

The SQLite database contains four key tables:

### search_runs
```sql
- id: Primary key
- study_name: Study class used (e.g., AVWAPDensityStudy)
- period_name: Market period name (e.g., "2008 Financial Crisis")
- start_date, end_date: Backtest date range
- trading_days, years_duration: Period duration
- buy_hold_median_return: B&H benchmark return
- symbols_tested, strategies_tested: Counts
- created_at: Timestamp
```

### aggregated_results
```sql
- strategy_id: Strategy identifier (buy_X__sell_Y format)
- trade_timing: 'conservative' or 'aggressive'
- median_*: MEDIAN metrics across all symbols
  - expectancy, win_rate, profit_factor
  - annualized_return, cagr, calmar_ratio
  - max_drawdown_pct, time_in_market_pct
  - alpha (vs Buy & Hold)
- consistency_score: % symbols profitable
- symbols_beating_benchmark: Count
```

### symbol_results
```sql
- Per-symbol detailed backtest results
- Includes all trade-level metrics
- Links to search_run via search_run_id
```

### trade_results (Dec 2025+)
```sql
- Individual trade records for position count analysis
- entry_date, exit_date: Trade duration for daily position counting
- strategy_id, symbol_code, trade_timing: Identifies the trade
- pnl_pct, duration_days: Trade performance
- Enables exact portfolio-level capital deployment tracking
```

## Key Concepts

### Strategy ID Format
Strategies follow the pattern: `buy_<entry_signals>__sell_<exit_signals>`
- Entry signals: bullish conditions (e.g., `adm_momentum_low`)
- Exit signals: bearish conditions (e.g., `adm_momentum_high`)
- Multiple signals combined with `+` (e.g., `buy_sig1+sig2__sell_sig3`)

### Market Periods
Standard periods tested:
1. 2008 Financial Crisis (Bear + Recovery)
2. 2022 Bear Market (Inflation Bear)
3. COVID Crash & Recovery (V-shaped)
4. 2021 Bull Market (Strong Bull)
5. Full Cycle 2018-2024 (Complete Cycle)
6. Dot-Com Bust (Tech Crash)

### Trade Timing
- `conservative`: Enter next bar after signal, exit next bar after exit signal
- `aggressive`: Enter/exit on same bar as signal (requires intraday capability)

### Median Aggregation
All metrics use MEDIAN aggregation across symbols to reduce outlier impact.

## Common Queries

```python
# Find robust strategies (work in all 6 periods)
find_robust_strategies(min_periods=6, min_consistency=0.7)

# Get top strategies for bear market
get_top_strategies(period_name="2008 Financial Crisis", sort_by="median_calmar_ratio")

# Find alpha generators
find_alpha_generators(min_alpha=5.0, min_beat_rate=0.6)

# Compare strategy across periods
compare_strategy_across_periods(strategy_id="buy_adm_momentum_low__sell_adm_momentum_high")
```

## Related Projects

- `trading-lab`: Strategy backtesting platform that generates the database
- `mcp-marketdata-db`: Market data MCP server for EOD price data
