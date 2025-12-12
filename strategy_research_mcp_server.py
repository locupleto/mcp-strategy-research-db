#!/usr/bin/env python3
"""
MCP Server for Strategy Research Database

Provides Claude Code access to strategy backtest results stored in a SQLite database.
Enables intelligent analysis of trading strategies across multiple market regimes.
"""

import os
import json
import sqlite3
import statistics
from typing import Any, Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
)


# Environment variables
STRATEGY_DB_PATH = os.environ.get("STRATEGY_DB_PATH", "")

# Initialize MCP server
server = Server("strategy-research-db")


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory for dict-like access."""
    if not STRATEGY_DB_PATH:
        raise ValueError("STRATEGY_DB_PATH environment variable not set")
    if not os.path.exists(STRATEGY_DB_PATH):
        raise FileNotFoundError(f"Database not found: {STRATEGY_DB_PATH}")

    conn = sqlite3.connect(STRATEGY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def format_pct(value: Optional[float], decimals: int = 2) -> str:
    """Format a percentage value for display (value is already in percentage form, e.g., 50.0 for 50%)."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_ratio_as_pct(value: Optional[float], decimals: int = 0) -> str:
    """Format a ratio (0.0-1.0) as a percentage for display."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: Optional[float], decimals: int = 2) -> str:
    """Format a number for display."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        # Database Overview Tools
        Tool(
            name="get_database_status",
            description="Get database statistics: total runs, strategies, symbols tested, date range, and size. Essential first call to understand the data available.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_search_runs",
            description="List all search runs in the database with their key parameters (period, study, symbols, date range). Use this to discover what backtests have been performed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "study_name": {
                        "type": "string",
                        "description": "Filter by study name (optional)"
                    },
                    "period_name": {
                        "type": "string",
                        "description": "Filter by period name (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum runs to return (default: 20)",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="get_run_details",
            description="Get detailed information about a specific search run including all parameters, benchmark data, and summary statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "The search run ID"
                    }
                },
                "required": ["run_id"]
            }
        ),

        # Strategy Discovery Tools
        Tool(
            name="get_top_strategies",
            description="Get top-performing strategies ranked by a metric. Supports filtering by consistency, trade frequency, and time-in-market.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Filter by specific run ID (optional, shows all runs if not specified)"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Metric to sort by: median_expectancy, median_win_rate, median_profit_factor, median_calmar_ratio, median_alpha, median_risk_adjusted_alpha, consistency_score",
                        "default": "median_expectancy"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of strategies to return (default: 20)",
                        "default": 20
                    },
                    "min_trades_per_year": {
                        "type": "number",
                        "description": "Minimum median trades per year (default: 0)",
                        "default": 0
                    },
                    "min_consistency": {
                        "type": "number",
                        "description": "Minimum consistency score 0.0-1.0 (default: 0)",
                        "default": 0
                    },
                    "max_time_in_market": {
                        "type": "number",
                        "description": "Maximum time-in-market percentage (default: 100)",
                        "default": 100
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' (T+1 Open), 'aggressive' (same-day Close), or 'both'",
                        "default": "conservative"
                    }
                }
            }
        ),
        Tool(
            name="get_strategy_details",
            description="Get detailed performance metrics for a specific strategy including all symbol results, risk metrics, and benchmark comparison.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "The strategy ID to look up"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional run ID to filter results"
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    }
                },
                "required": ["strategy_id"]
            }
        ),
        Tool(
            name="search_strategies_by_signal",
            description="Find strategies that use specific entry or exit signals.",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_name": {
                        "type": "string",
                        "description": "Signal name to search for (e.g., 'adm_momentum_rising', 'asi_overbought')"
                    },
                    "signal_type": {
                        "type": "string",
                        "description": "Filter by signal type: 'buy', 'sell', or 'any'",
                        "default": "any"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "default": 20
                    }
                },
                "required": ["signal_name"]
            }
        ),

        # Cross-Period Analysis Tools
        Tool(
            name="compare_strategy_across_periods",
            description="Compare a strategy's performance across different market regimes/periods. Essential for validating robustness.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "The strategy ID to compare"
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    }
                },
                "required": ["strategy_id"]
            }
        ),
        Tool(
            name="find_robust_strategies",
            description="Find strategies that perform consistently well across ALL tested market periods. Returns strategies with high consistency across different regimes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_periods": {
                        "type": "integer",
                        "description": "Minimum number of periods the strategy must appear in (default: 3)",
                        "default": 3
                    },
                    "min_avg_expectancy": {
                        "type": "number",
                        "description": "Minimum average expectancy across periods (default: 1.0)",
                        "default": 1.0
                    },
                    "max_expectancy_variance": {
                        "type": "number",
                        "description": "Maximum variance in expectancy across periods - lower means more consistent (default: 5.0)",
                        "default": 5.0
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "default": 20
                    }
                }
            }
        ),

        # Benchmark Comparison Tools
        Tool(
            name="get_benchmark_comparison",
            description="Compare strategy performance against Buy & Hold benchmark for a specific run or period.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "The search run ID"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of strategies to compare (default: 10)",
                        "default": 10
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    }
                },
                "required": ["run_id"]
            }
        ),
        Tool(
            name="find_alpha_generators",
            description="Find strategies that consistently generate alpha (outperform Buy & Hold) across symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_alpha": {
                        "type": "number",
                        "description": "Minimum median alpha vs B&H in percentage points (default: 5.0)",
                        "default": 5.0
                    },
                    "min_beat_rate": {
                        "type": "number",
                        "description": "Minimum percentage of symbols that must beat B&H (default: 50)",
                        "default": 50
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "default": 20
                    }
                }
            }
        ),

        # Risk Analysis Tools
        Tool(
            name="get_risk_adjusted_rankings",
            description="Rank strategies by risk-adjusted metrics (Calmar ratio, return/MaxDD). Identifies strategies with best risk/reward profiles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Optional run ID to filter"
                    },
                    "min_calmar": {
                        "type": "number",
                        "description": "Minimum Calmar ratio (default: 1.0)",
                        "default": 1.0
                    },
                    "max_drawdown": {
                        "type": "number",
                        "description": "Maximum acceptable drawdown percentage (default: 20)",
                        "default": 20
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "default": 20
                    }
                }
            }
        ),

        # Symbol-Level Analysis Tools
        Tool(
            name="get_strategy_symbol_breakdown",
            description="Get per-symbol performance breakdown for a strategy. Shows which symbols the strategy works best/worst on.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "The strategy ID"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional run ID to filter"
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort by: 'expectancy', 'alpha', 'win_rate', 'calmar_ratio'",
                        "default": "expectancy"
                    }
                },
                "required": ["strategy_id"]
            }
        ),

        # Aggregation and Summary Tools
        Tool(
            name="get_signal_performance_summary",
            description="Aggregate performance by signal type. Shows which individual signals appear in the best-performing strategies.",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_type": {
                        "type": "string",
                        "description": "Filter by 'buy' (entry) or 'sell' (exit) signals",
                        "default": "buy"
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    }
                }
            }
        ),
        Tool(
            name="get_schema",
            description="Get the database schema showing all tables, their columns, and data types. Essential for understanding the database structure before writing custom queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Optional: specific table name to get schema for. If not provided, returns schema for all tables."
                    }
                }
            }
        ),
        Tool(
            name="list_strategy_ids",
            description="List strategy IDs in the database with optional pattern filtering. Use this to discover valid strategy IDs before using other tools that require a strategy_id parameter. Strategy IDs follow the format: buy_signal1+signal2__sell_signal1+signal2",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Optional: filter strategy IDs containing this pattern (case-insensitive). E.g., 'momentum_rising' to find all strategies using that signal."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 50)",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="run_custom_query",
            description="Execute a custom SQL query against the strategy database. For advanced users who need specific analysis not covered by other tools. Supports SELECT queries and PRAGMA commands for schema inspection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query or PRAGMA command to execute (read-only)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return (default: 100, max: 500)",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),

        # Timing Mode Comparison Tool
        Tool(
            name="compare_timing_modes",
            description="Compare strategy performance between Conservative (T+1 Open entry) and Aggressive (same-day Close entry) timing modes. Shows side-by-side comparison with delta calculations for each metric.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Optional run ID to filter (shows all runs if not specified)"
                    },
                    "strategy_id": {
                        "type": "string",
                        "description": "Optional specific strategy to compare. If not provided, shows top strategies with biggest timing impact."
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort by: 'expectancy_delta' (biggest change), 'cons_expectancy', 'aggr_expectancy', 'win_rate_delta'",
                        "default": "expectancy_delta"
                    },
                    "min_trades_per_year": {
                        "type": "number",
                        "description": "Minimum median trades per year (applies to both timings)",
                        "default": 5
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of strategies to return (default: 20)",
                        "default": 20
                    }
                }
            }
        ),

        # Capital Deployment Analysis Tool
        Tool(
            name="get_capital_deployment_analysis",
            description="Analyze portfolio-level capital deployment for a strategy. Shows estimated simultaneous positions, time-in-market stats, and compares two strategies side-by-side across market periods. Useful for understanding how much capital a strategy keeps deployed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Primary strategy ID to analyze"
                    },
                    "compare_strategy_id": {
                        "type": "string",
                        "description": "Optional second strategy ID for side-by-side comparison"
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional run ID to filter to a specific period"
                    }
                },
                "required": ["strategy_id"]
            }
        ),

        # Daily Position Count Tool (for time-series graphs)
        Tool(
            name="get_daily_position_counts",
            description="Get daily position counts for a strategy across all symbols. Returns time-series data showing how many positions were open each trading day - perfect for graphing capital deployment over time. Requires trade-level data (available for runs after Dec 2025).",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy ID to analyze"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Filter by specific run ID (required to get consistent date range)"
                    },
                    "trade_timing": {
                        "type": "string",
                        "description": "Filter by timing: 'conservative' or 'aggressive'",
                        "default": "conservative"
                    },
                    "compare_strategy_id": {
                        "type": "string",
                        "description": "Optional second strategy for side-by-side comparison"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'summary' (stats only), 'daily' (full time-series), 'weekly' (aggregated by week)",
                        "default": "summary"
                    }
                },
                "required": ["strategy_id", "run_id"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "get_database_status":
            return await handle_get_database_status(arguments)
        elif name == "list_search_runs":
            return await handle_list_search_runs(arguments)
        elif name == "get_run_details":
            return await handle_get_run_details(arguments)
        elif name == "get_top_strategies":
            return await handle_get_top_strategies(arguments)
        elif name == "get_strategy_details":
            return await handle_get_strategy_details(arguments)
        elif name == "search_strategies_by_signal":
            return await handle_search_strategies_by_signal(arguments)
        elif name == "compare_strategy_across_periods":
            return await handle_compare_strategy_across_periods(arguments)
        elif name == "find_robust_strategies":
            return await handle_find_robust_strategies(arguments)
        elif name == "get_benchmark_comparison":
            return await handle_get_benchmark_comparison(arguments)
        elif name == "find_alpha_generators":
            return await handle_find_alpha_generators(arguments)
        elif name == "get_risk_adjusted_rankings":
            return await handle_get_risk_adjusted_rankings(arguments)
        elif name == "get_strategy_symbol_breakdown":
            return await handle_get_strategy_symbol_breakdown(arguments)
        elif name == "get_signal_performance_summary":
            return await handle_get_signal_performance_summary(arguments)
        elif name == "get_schema":
            return await handle_get_schema(arguments)
        elif name == "list_strategy_ids":
            return await handle_list_strategy_ids(arguments)
        elif name == "run_custom_query":
            return await handle_run_custom_query(arguments)
        elif name == "compare_timing_modes":
            return await handle_compare_timing_modes(arguments)
        elif name == "get_capital_deployment_analysis":
            return await handle_get_capital_deployment_analysis(arguments)
        elif name == "get_daily_position_counts":
            return await handle_get_daily_position_counts(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_get_database_status(arguments: dict[str, Any]) -> list[TextContent]:
    """Get database statistics and status."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Get run count
        cursor.execute("SELECT COUNT(*) as count FROM search_runs")
        run_count = cursor.fetchone()['count']

        # Get strategy count
        cursor.execute("SELECT COUNT(*) as count FROM strategies")
        strategy_count = cursor.fetchone()['count']

        # Get aggregated results count
        cursor.execute("SELECT COUNT(*) as count FROM aggregated_results")
        agg_count = cursor.fetchone()['count']

        # Get symbol results count
        cursor.execute("SELECT COUNT(*) as count FROM symbol_results")
        symbol_results_count = cursor.fetchone()['count']

        # Get unique symbols tested
        cursor.execute("SELECT COUNT(DISTINCT symbol_code) as count FROM symbol_results")
        unique_symbols = cursor.fetchone()['count']

        # Get periods tested
        cursor.execute("SELECT DISTINCT period_name FROM search_runs WHERE period_name IS NOT NULL")
        periods = [row['period_name'] for row in cursor.fetchall()]

        # Get studies tested
        cursor.execute("SELECT DISTINCT study_name FROM search_runs WHERE study_name IS NOT NULL")
        studies = [row['study_name'] for row in cursor.fetchall()]

        # Get date range
        cursor.execute("""
            SELECT MIN(run_date) as min_date, MAX(run_date) as max_date
            FROM search_runs
        """)
        date_row = cursor.fetchone()

        # Database file size
        db_size = os.path.getsize(STRATEGY_DB_PATH) / (1024 * 1024)  # MB

        lines = [
            "Strategy Research Database Status",
            "=" * 50,
            f"Database Path: {STRATEGY_DB_PATH}",
            f"Database Size: {db_size:.2f} MB",
            "",
            "Statistics:",
            f"  Search Runs: {run_count}",
            f"  Unique Strategies: {strategy_count}",
            f"  Aggregated Results: {agg_count}",
            f"  Symbol Results: {symbol_results_count}",
            f"  Unique Symbols Tested: {unique_symbols}",
            "",
            f"Date Range: {date_row['min_date'] or 'N/A'} to {date_row['max_date'] or 'N/A'}",
            "",
            "Market Periods Tested:",
        ]
        for period in periods:
            lines.append(f"  - {period}")

        lines.append("")
        lines.append("Studies Used:")
        for study in studies:
            lines.append(f"  - {study}")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_list_search_runs(arguments: dict[str, Any]) -> list[TextContent]:
    """List all search runs."""
    study_name = arguments.get("study_name")
    period_name = arguments.get("period_name")
    limit = arguments.get("limit", 20)

    conn = get_connection()
    try:
        query = """
            SELECT
                id,
                run_date,
                csv_file,
                period_name,
                study_name,
                years_back,
                start_date,
                end_date,
                symbols_count,
                strategies_count,
                trading_days,
                years_duration,
                buy_hold_median_return,
                status,
                duration_seconds
            FROM search_runs
            WHERE 1=1
        """
        params = []

        if study_name:
            query += " AND study_name LIKE ?"
            params.append(f"%{study_name}%")
        if period_name:
            query += " AND period_name LIKE ?"
            params.append(f"%{period_name}%")

        query += " ORDER BY run_date DESC LIMIT ?"
        params.append(limit)

        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text="No search runs found matching criteria.")]

        lines = [
            f"Search Runs ({len(rows)} results)",
            "=" * 80,
        ]

        for row in rows:
            period = row['period_name'] or 'Custom'
            study = row['study_name'] or 'Default'
            bh_return = format_pct(row['buy_hold_median_return']) if row['buy_hold_median_return'] else 'N/A'
            duration = f"{row['duration_seconds']:.0f}s" if row['duration_seconds'] else 'N/A'

            lines.append("")
            lines.append(f"[{row['id']}] {period}")
            lines.append(f"  Study: {study}")
            lines.append(f"  Date Range: {row['start_date'] or 'N/A'} to {row['end_date'] or 'N/A'}")
            lines.append(f"  Symbols: {row['symbols_count'] or 0} | Strategies: {row['strategies_count'] or 0}")
            lines.append(f"  B&H Median: {bh_return} | Duration: {duration}")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_run_details(arguments: dict[str, Any]) -> list[TextContent]:
    """Get detailed information about a search run."""
    run_id = arguments["run_id"]

    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM search_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()

        if not row:
            return [TextContent(type="text", text=f"Run not found: {run_id}")]

        # Parse JSON fields
        buy_signals = json.loads(row['buy_signals_used']) if row['buy_signals_used'] else []
        sell_signals = json.loads(row['sell_signals_used']) if row['sell_signals_used'] else []

        lines = [
            f"Search Run Details: {run_id}",
            "=" * 60,
            "",
            "Configuration:",
            f"  Period Name: {row['period_name'] or 'N/A'}",
            f"  Study Name: {row['study_name'] or 'N/A'}",
            f"  CSV File: {row['csv_file']}",
            f"  Date Range: {row['start_date'] or 'N/A'} to {row['end_date'] or 'N/A'}",
            f"  Years Back: {row['years_back'] or 'N/A'}",
            "",
            "Test Parameters:",
            f"  Quorum Mode: {row['quorum_mode']}",
            f"  Max Quorum Size: {row['max_quorum_size']}",
            f"  Symbols Count: {row['symbols_count'] or 0}",
            f"  Strategies Count: {row['strategies_count'] or 0}",
            "",
            "Duration Metrics:",
            f"  Trading Days: {row['trading_days'] or 'N/A'}",
            f"  Years Duration: {format_number(row['years_duration'])}",
            "",
            "Buy & Hold Benchmark:",
            f"  Median Return: {format_pct(row['buy_hold_median_return'])}",
            f"  Average Return: {format_pct(row['buy_hold_avg_return'])}",
            f"  Min Return: {format_pct(row['buy_hold_min_return'])}",
            f"  Max Return: {format_pct(row['buy_hold_max_return'])}",
            f"  Std Dev: {format_pct(row['buy_hold_stddev'])}",
            "",
            "Signals Used:",
            f"  Buy Signals ({len(buy_signals)}): {', '.join(buy_signals[:5])}{'...' if len(buy_signals) > 5 else ''}",
            f"  Sell Signals ({len(sell_signals)}): {', '.join(sell_signals[:5])}{'...' if len(sell_signals) > 5 else ''}",
            "",
            "Execution:",
            f"  Run Date: {row['run_date']}",
            f"  Status: {row['status']}",
            f"  Duration: {row['duration_seconds']:.1f}s" if row['duration_seconds'] else "  Duration: N/A",
        ]

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_top_strategies(arguments: dict[str, Any]) -> list[TextContent]:
    """Get top-performing strategies."""
    run_id = arguments.get("run_id")
    sort_by = arguments.get("sort_by", "median_expectancy")
    limit = arguments.get("limit", 20)
    min_trades_per_year = arguments.get("min_trades_per_year", 0)
    min_consistency = arguments.get("min_consistency", 0)
    max_time_in_market = arguments.get("max_time_in_market", 100)
    trade_timing = arguments.get("trade_timing", "conservative")

    # Validate sort column
    valid_sort_cols = [
        'median_expectancy', 'median_win_rate', 'median_profit_factor',
        'consistency_score', 'median_calmar_ratio', 'median_alpha',
        'median_risk_adjusted_alpha', 'median_annualized_return', 'symbols_beating_benchmark'
    ]
    if sort_by not in valid_sort_cols:
        sort_by = 'median_expectancy'

    conn = get_connection()
    try:
        query = """
            SELECT
                ar.strategy_id,
                ar.trade_timing,
                s.buy_signals,
                s.sell_signals,
                ar.median_expectancy,
                ar.median_win_rate,
                ar.median_profit_factor,
                ar.median_max_drawdown_pct,
                ar.median_time_in_market_pct,
                ar.median_avg_trades_per_year,
                ar.median_annualized_return,
                ar.median_calmar_ratio,
                ar.median_alpha,
                ar.median_risk_adjusted_alpha,
                ar.symbols_beating_benchmark,
                ar.symbols_tested,
                ar.symbols_profitable,
                ar.consistency_score,
                ar.total_trades_all_symbols,
                sr.period_name
            FROM aggregated_results ar
            JOIN strategies s ON ar.strategy_id = s.strategy_id
            JOIN search_runs sr ON ar.search_run_id = sr.id
            WHERE COALESCE(ar.median_avg_trades_per_year, 0) >= ?
              AND COALESCE(ar.consistency_score, 0) >= ?
              AND COALESCE(ar.median_time_in_market_pct, 0) <= ?
        """
        params = [min_trades_per_year, min_consistency, max_time_in_market]

        if trade_timing != 'both':
            query += " AND ar.trade_timing = ?"
            params.append(trade_timing)

        if run_id:
            query += " AND ar.search_run_id = ?"
            params.append(run_id)

        query += f" ORDER BY ar.{sort_by} DESC NULLS LAST LIMIT ?"
        params.append(limit)

        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text="No strategies found matching criteria.")]

        lines = [
            f"Top Strategies by {sort_by.replace('_', ' ').title()}",
            "=" * 80,
            ""
        ]

        for i, row in enumerate(rows, 1):
            buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
            sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []

            timing_label = f"[{row['trade_timing'][:4].upper()}]" if row['trade_timing'] else ""
            period = f"({row['period_name']})" if row['period_name'] else ""

            lines.append(f"#{i} {timing_label} {period}")
            lines.append(f"  BUY:  {' + '.join(buy_signals)}")
            lines.append(f"  SELL: {' + '.join(sell_signals)}")
            lines.append(f"  ─" * 35)
            lines.append(
                f"  Expectancy: {format_pct(row['median_expectancy'])} | "
                f"Win Rate: {format_pct(row['median_win_rate'])} | "
                f"PF: {format_number(row['median_profit_factor'])}"
            )
            lines.append(
                f"  Max DD: {format_pct(row['median_max_drawdown_pct'])} | "
                f"Time in Mkt: {format_pct(row['median_time_in_market_pct'])} | "
                f"Consistency: {format_ratio_as_pct(row['consistency_score']) if row['consistency_score'] else 'N/A'}"
            )
            if row['median_annualized_return'] is not None:
                beat_pct = (row['symbols_beating_benchmark'] / row['symbols_tested'] * 100) if row['symbols_tested'] else 0
                lines.append(
                    f"  CAGR: {format_pct(row['median_annualized_return'])} | "
                    f"Calmar: {format_number(row['median_calmar_ratio'])} | "
                    f"Alpha: {format_pct(row['median_alpha'])}"
                )
                risk_adj_alpha = row['median_risk_adjusted_alpha'] if 'median_risk_adjusted_alpha' in row.keys() else None
                if risk_adj_alpha is not None:
                    lines.append(
                        f"  Risk-Adj Alpha: {format_pct(risk_adj_alpha)} | "
                        f"Beat B&H: {row['symbols_beating_benchmark'] or 0}/{row['symbols_tested']} ({beat_pct:.0f}%)"
                    )
                else:
                    lines.append(
                        f"  Beat B&H: {row['symbols_beating_benchmark'] or 0}/{row['symbols_tested']} ({beat_pct:.0f}%)"
                    )
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_strategy_details(arguments: dict[str, Any]) -> list[TextContent]:
    """Get detailed strategy information."""
    strategy_id = arguments["strategy_id"]
    run_id = arguments.get("run_id")
    trade_timing = arguments.get("trade_timing", "conservative")

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Get strategy definition
        cursor.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,))
        strategy_row = cursor.fetchone()

        if not strategy_row:
            # Try to suggest similar strategies
            cursor.execute("""
                SELECT strategy_id FROM strategies
                WHERE strategy_id LIKE ?
                LIMIT 5
            """, (f"%{strategy_id.split('__')[0].replace('buy_', '')}%",))
            suggestions = [r['strategy_id'] for r in cursor.fetchall()]

            msg = f"Strategy not found: {strategy_id}\n\n"
            msg += "Expected format: buy_signal1+signal2__sell_signal1+signal2\n"
            msg += "Example: buy_adm_momentum_rising__sell_adm_acceleration_falling\n"
            if suggestions:
                msg += f"\nSimilar strategies found:\n"
                for s in suggestions:
                    msg += f"  - {s}\n"
            msg += "\nUse list_strategy_ids tool to discover valid strategy IDs."
            return [TextContent(type="text", text=msg)]

        buy_signals = json.loads(strategy_row['buy_signals']) if strategy_row['buy_signals'] else []
        sell_signals = json.loads(strategy_row['sell_signals']) if strategy_row['sell_signals'] else []

        # Get aggregated results
        agg_query = """
            SELECT ar.*, sr.period_name, sr.start_date, sr.end_date, sr.buy_hold_median_return
            FROM aggregated_results ar
            JOIN search_runs sr ON ar.search_run_id = sr.id
            WHERE ar.strategy_id = ? AND ar.trade_timing = ?
        """
        params = [strategy_id, trade_timing]
        if run_id:
            agg_query += " AND ar.search_run_id = ?"
            params.append(run_id)
        agg_query += " ORDER BY sr.run_date DESC"

        cursor.execute(agg_query, params)
        agg_rows = cursor.fetchall()

        lines = [
            f"Strategy Details: {strategy_id}",
            "=" * 70,
            "",
            "Signal Composition:",
            f"  BUY:  {' + '.join(buy_signals)}",
            f"  SELL: {' + '.join(sell_signals)}",
            "",
        ]

        if not agg_rows:
            lines.append(f"No aggregated results found for timing={trade_timing}")
        else:
            lines.append(f"Performance Across {len(agg_rows)} Period(s) [{trade_timing}]:")
            lines.append("-" * 70)

            for row in agg_rows:
                period = row['period_name'] or 'Custom'
                lines.append("")
                lines.append(f"Period: {period} ({row['start_date']} to {row['end_date']})")
                lines.append(f"  Expectancy: {format_pct(row['median_expectancy'])} | Win Rate: {format_pct(row['median_win_rate'])}")
                lines.append(f"  Profit Factor: {format_number(row['median_profit_factor'])} | Max DD: {format_pct(row['median_max_drawdown_pct'])}")
                lines.append(f"  CAGR: {format_pct(row['median_annualized_return'])} | Calmar: {format_number(row['median_calmar_ratio'])}")
                lines.append(f"  Symbols: {row['symbols_tested']} tested, {row['symbols_profitable']} profitable")
                lines.append(f"  Consistency: {format_ratio_as_pct(row['consistency_score'])}")
                if row['median_alpha'] is not None:
                    lines.append(f"  Alpha vs B&H: {format_pct(row['median_alpha'])} (B&H baseline: {format_pct(row['buy_hold_median_return'])})")
                    risk_adj = row['median_risk_adjusted_alpha'] if 'median_risk_adjusted_alpha' in row.keys() else None
                    if risk_adj is not None:
                        lines.append(f"  Risk-Adjusted Alpha: {format_pct(risk_adj)} (accounts for time-in-market)")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_search_strategies_by_signal(arguments: dict[str, Any]) -> list[TextContent]:
    """Search strategies by signal name."""
    signal_name = arguments["signal_name"]
    signal_type = arguments.get("signal_type", "any")
    limit = arguments.get("limit", 20)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT
                s.strategy_id,
                s.buy_signals,
                s.sell_signals,
                ar.median_expectancy,
                ar.median_win_rate,
                ar.consistency_score,
                sr.period_name
            FROM strategies s
            JOIN aggregated_results ar ON s.strategy_id = ar.strategy_id
            JOIN search_runs sr ON ar.search_run_id = sr.id
            WHERE 1=1
        """
        params = []

        if signal_type in ('buy', 'any'):
            query += " AND s.buy_signals LIKE ?"
            params.append(f'%{signal_name}%')
        if signal_type in ('sell', 'any'):
            if signal_type == 'any':
                query = query.rstrip(" AND s.buy_signals LIKE ?")
                params = []
                query += " AND (s.buy_signals LIKE ? OR s.sell_signals LIKE ?)"
                params.extend([f'%{signal_name}%', f'%{signal_name}%'])
            else:
                query += " AND s.sell_signals LIKE ?"
                params.append(f'%{signal_name}%')

        query += " ORDER BY ar.median_expectancy DESC NULLS LAST LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text=f"No strategies found using signal: {signal_name}")]

        lines = [
            f"Strategies Using '{signal_name}'",
            "=" * 70,
            ""
        ]

        for row in rows:
            buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
            sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []

            # Highlight the matching signal
            buy_str = ' + '.join(
                f"**{s}**" if signal_name.lower() in s.lower() else s
                for s in buy_signals
            )
            sell_str = ' + '.join(
                f"**{s}**" if signal_name.lower() in s.lower() else s
                for s in sell_signals
            )

            lines.append(f"BUY: {buy_str}")
            lines.append(f"SELL: {sell_str}")
            lines.append(f"  Expectancy: {format_pct(row['median_expectancy'])} | Win Rate: {format_pct(row['median_win_rate'])} | Consistency: {format_ratio_as_pct(row['consistency_score'])}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_compare_strategy_across_periods(arguments: dict[str, Any]) -> list[TextContent]:
    """Compare strategy performance across different periods."""
    strategy_id = arguments["strategy_id"]
    trade_timing = arguments.get("trade_timing", "conservative")

    conn = get_connection()
    try:
        cursor = conn.cursor()

        query = """
            SELECT
                sr.period_name,
                sr.start_date,
                sr.end_date,
                sr.years_duration,
                sr.buy_hold_median_return,
                ar.median_expectancy,
                ar.median_win_rate,
                ar.median_profit_factor,
                ar.median_max_drawdown_pct,
                ar.median_annualized_return,
                ar.median_calmar_ratio,
                ar.median_alpha,
                ar.median_risk_adjusted_alpha,
                ar.symbols_tested,
                ar.symbols_profitable,
                ar.symbols_beating_benchmark,
                ar.consistency_score
            FROM aggregated_results ar
            JOIN search_runs sr ON ar.search_run_id = sr.id
            WHERE ar.strategy_id = ?
              AND ar.trade_timing = ?
            ORDER BY sr.start_date
        """
        cursor.execute(query, (strategy_id, trade_timing))
        rows = cursor.fetchall()

        if not rows:
            # Check if strategy exists at all
            cursor.execute("SELECT COUNT(*) as cnt FROM strategies WHERE strategy_id = ?", (strategy_id,))
            exists = cursor.fetchone()['cnt'] > 0

            if not exists:
                # Try to suggest similar strategies
                cursor.execute("""
                    SELECT strategy_id FROM strategies
                    WHERE strategy_id LIKE ?
                    LIMIT 5
                """, (f"%{strategy_id.split('__')[0].replace('buy_', '')}%",))
                suggestions = [r['strategy_id'] for r in cursor.fetchall()]

                msg = f"Strategy not found: {strategy_id}\n\n"
                msg += "Expected format: buy_signal1+signal2__sell_signal1+signal2\n"
                msg += "Example: buy_adm_momentum_rising__sell_adm_acceleration_falling\n"
                if suggestions:
                    msg += f"\nSimilar strategies found:\n"
                    for s in suggestions:
                        msg += f"  - {s}\n"
                msg += "\nUse list_strategy_ids tool to discover valid strategy IDs."
                return [TextContent(type="text", text=msg)]
            else:
                return [TextContent(type="text", text=f"Strategy exists but no results for timing={trade_timing}. Try trade_timing='aggressive'.")]

        # Calculate cross-period statistics
        expectancies = [r['median_expectancy'] for r in rows if r['median_expectancy'] is not None]
        win_rates = [r['median_win_rate'] for r in rows if r['median_win_rate'] is not None]
        consistencies = [r['consistency_score'] for r in rows if r['consistency_score'] is not None]

        lines = [
            f"Cross-Period Analysis: {strategy_id}",
            f"Trade Timing: {trade_timing}",
            "=" * 80,
            "",
        ]

        # Summary statistics
        if expectancies:
            lines.append("Summary Across All Periods:")
            lines.append(f"  Expectancy - Mean: {statistics.mean(expectancies):.2f}%, Std: {statistics.stdev(expectancies):.2f}%" if len(expectancies) > 1 else f"  Expectancy: {expectancies[0]:.2f}%")
            lines.append(f"  Win Rate - Mean: {statistics.mean(win_rates):.2f}%")
            lines.append(f"  Consistency - Mean: {statistics.mean(consistencies)*100:.0f}%")
            lines.append("")

        lines.append("Period-by-Period Breakdown:")
        lines.append("-" * 80)

        for row in rows:
            period = row['period_name'] or 'Custom'
            beat_pct = (row['symbols_beating_benchmark'] / row['symbols_tested'] * 100) if row['symbols_tested'] and row['symbols_beating_benchmark'] else 0

            lines.append("")
            lines.append(f"[{period}] {row['start_date']} to {row['end_date']}")
            lines.append(f"  B&H Baseline: {format_pct(row['buy_hold_median_return'])}")
            lines.append(f"  Expectancy: {format_pct(row['median_expectancy'])} | Win Rate: {format_pct(row['median_win_rate'])} | PF: {format_number(row['median_profit_factor'])}")
            lines.append(f"  CAGR: {format_pct(row['median_annualized_return'])} | Calmar: {format_number(row['median_calmar_ratio'])} | Max DD: {format_pct(row['median_max_drawdown_pct'])}")
            risk_adj = row['median_risk_adjusted_alpha'] if 'median_risk_adjusted_alpha' in row.keys() else None
            if risk_adj is not None:
                lines.append(f"  Alpha: {format_pct(row['median_alpha'])} | Risk-Adj Alpha: {format_pct(risk_adj)}")
                lines.append(f"  Beat B&H: {row['symbols_beating_benchmark'] or 0}/{row['symbols_tested']} ({beat_pct:.0f}%)")
            else:
                lines.append(f"  Alpha: {format_pct(row['median_alpha'])} | Beat B&H: {row['symbols_beating_benchmark'] or 0}/{row['symbols_tested']} ({beat_pct:.0f}%)")
            lines.append(f"  Consistency: {format_ratio_as_pct(row['consistency_score'])} ({row['symbols_profitable']}/{row['symbols_tested']} profitable)")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_find_robust_strategies(arguments: dict[str, Any]) -> list[TextContent]:
    """Find strategies that perform consistently across all periods."""
    min_periods = arguments.get("min_periods", 3)
    min_avg_expectancy = arguments.get("min_avg_expectancy", 1.0)
    max_expectancy_variance = arguments.get("max_expectancy_variance", 5.0)
    trade_timing = arguments.get("trade_timing", "conservative")
    limit = arguments.get("limit", 20)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Get strategies with their cross-period statistics
        query = """
            SELECT
                ar.strategy_id,
                s.buy_signals,
                s.sell_signals,
                COUNT(DISTINCT sr.period_name) as period_count,
                AVG(ar.median_expectancy) as avg_expectancy,
                MIN(ar.median_expectancy) as min_expectancy,
                MAX(ar.median_expectancy) as max_expectancy,
                AVG(ar.median_win_rate) as avg_win_rate,
                AVG(ar.consistency_score) as avg_consistency,
                AVG(ar.median_alpha) as avg_alpha,
                GROUP_CONCAT(sr.period_name || ':' || ROUND(ar.median_expectancy, 2), ' | ') as period_details
            FROM aggregated_results ar
            JOIN strategies s ON ar.strategy_id = s.strategy_id
            JOIN search_runs sr ON ar.search_run_id = sr.id
            WHERE ar.trade_timing = ?
              AND ar.median_expectancy IS NOT NULL
            GROUP BY ar.strategy_id
            HAVING period_count >= ?
               AND avg_expectancy >= ?
               AND (max_expectancy - min_expectancy) <= ?
            ORDER BY avg_expectancy DESC, period_count DESC
            LIMIT ?
        """
        cursor.execute(query, (trade_timing, min_periods, min_avg_expectancy, max_expectancy_variance, limit))
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text="No robust strategies found matching criteria. Try relaxing the filters.")]

        lines = [
            f"Robust Strategies (Consistent Across {min_periods}+ Periods)",
            "=" * 80,
            f"Filters: min_expectancy={min_avg_expectancy}%, max_variance={max_expectancy_variance}%",
            ""
        ]

        for i, row in enumerate(rows, 1):
            buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
            sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []

            variance = row['max_expectancy'] - row['min_expectancy'] if row['max_expectancy'] and row['min_expectancy'] else 0

            lines.append(f"#{i} [{row['period_count']} periods]")
            lines.append(f"  BUY:  {' + '.join(buy_signals)}")
            lines.append(f"  SELL: {' + '.join(sell_signals)}")
            lines.append(f"  ─" * 35)
            lines.append(f"  Avg Expectancy: {format_pct(row['avg_expectancy'])} (range: {format_pct(row['min_expectancy'])} to {format_pct(row['max_expectancy'])})")
            lines.append(f"  Variance: {variance:.2f}% | Avg Win Rate: {format_pct(row['avg_win_rate'])} | Avg Consistency: {format_ratio_as_pct(row['avg_consistency'])}")
            lines.append(f"  Avg Alpha vs B&H: {format_pct(row['avg_alpha'])}")
            lines.append(f"  Periods: {row['period_details']}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_benchmark_comparison(arguments: dict[str, Any]) -> list[TextContent]:
    """Compare strategies against Buy & Hold benchmark."""
    run_id = arguments["run_id"]
    limit = arguments.get("limit", 10)
    trade_timing = arguments.get("trade_timing", "conservative")

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Get run info first
        cursor.execute("""
            SELECT period_name, start_date, end_date, years_duration,
                   buy_hold_median_return, buy_hold_avg_return, buy_hold_min_return, buy_hold_max_return
            FROM search_runs WHERE id = ?
        """, (run_id,))
        run_row = cursor.fetchone()

        if not run_row:
            return [TextContent(type="text", text=f"Run not found: {run_id}")]

        # Get top strategies by alpha
        cursor.execute("""
            SELECT
                ar.strategy_id,
                s.buy_signals,
                s.sell_signals,
                ar.median_expectancy,
                ar.median_annualized_return,
                ar.median_alpha,
                ar.symbols_beating_benchmark,
                ar.symbols_tested,
                ar.median_calmar_ratio,
                ar.consistency_score
            FROM aggregated_results ar
            JOIN strategies s ON ar.strategy_id = s.strategy_id
            WHERE ar.search_run_id = ?
              AND ar.trade_timing = ?
              AND ar.median_alpha IS NOT NULL
            ORDER BY ar.median_alpha DESC
            LIMIT ?
        """, (run_id, trade_timing, limit))
        rows = cursor.fetchall()

        lines = [
            f"Benchmark Comparison: {run_row['period_name'] or run_id}",
            "=" * 80,
            "",
            "Buy & Hold Benchmark:",
            f"  Period: {run_row['start_date']} to {run_row['end_date']} ({format_number(run_row['years_duration'])} years)",
            f"  Median Return: {format_pct(run_row['buy_hold_median_return'])}",
            f"  Average Return: {format_pct(run_row['buy_hold_avg_return'])}",
            f"  Range: {format_pct(run_row['buy_hold_min_return'])} to {format_pct(run_row['buy_hold_max_return'])}",
            "",
            "Top Strategies by Alpha (Outperformance vs B&H):",
            "-" * 80,
        ]

        for i, row in enumerate(rows, 1):
            buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
            sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []
            beat_pct = (row['symbols_beating_benchmark'] / row['symbols_tested'] * 100) if row['symbols_tested'] else 0

            lines.append("")
            lines.append(f"#{i} Alpha: {format_pct(row['median_alpha'])}")
            lines.append(f"  BUY: {' + '.join(buy_signals)}")
            lines.append(f"  SELL: {' + '.join(sell_signals)}")
            lines.append(f"  Strategy CAGR: {format_pct(row['median_annualized_return'])} | Calmar: {format_number(row['median_calmar_ratio'])}")
            lines.append(f"  Beat B&H: {row['symbols_beating_benchmark']}/{row['symbols_tested']} symbols ({beat_pct:.0f}%)")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_find_alpha_generators(arguments: dict[str, Any]) -> list[TextContent]:
    """Find strategies that consistently generate alpha."""
    min_alpha = arguments.get("min_alpha", 5.0)
    min_beat_rate = arguments.get("min_beat_rate", 50)
    trade_timing = arguments.get("trade_timing", "conservative")
    limit = arguments.get("limit", 20)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        query = """
            SELECT
                ar.strategy_id,
                s.buy_signals,
                s.sell_signals,
                ar.median_alpha,
                ar.symbols_beating_benchmark,
                ar.symbols_tested,
                ar.median_expectancy,
                ar.median_annualized_return,
                ar.median_calmar_ratio,
                ar.consistency_score,
                sr.period_name
            FROM aggregated_results ar
            JOIN strategies s ON ar.strategy_id = s.strategy_id
            JOIN search_runs sr ON ar.search_run_id = sr.id
            WHERE ar.trade_timing = ?
              AND ar.median_alpha >= ?
              AND ar.symbols_tested > 0
              AND (ar.symbols_beating_benchmark * 100.0 / ar.symbols_tested) >= ?
            ORDER BY ar.median_alpha DESC
            LIMIT ?
        """
        cursor.execute(query, (trade_timing, min_alpha, min_beat_rate, limit))
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text=f"No strategies found with alpha >= {min_alpha}% and beat rate >= {min_beat_rate}%")]

        lines = [
            f"Alpha Generators (Alpha >= {min_alpha}%, Beat Rate >= {min_beat_rate}%)",
            "=" * 80,
            ""
        ]

        for i, row in enumerate(rows, 1):
            buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
            sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []
            beat_pct = (row['symbols_beating_benchmark'] / row['symbols_tested'] * 100) if row['symbols_tested'] else 0

            lines.append(f"#{i} Alpha: {format_pct(row['median_alpha'])} [{row['period_name'] or 'Custom'}]")
            lines.append(f"  BUY: {' + '.join(buy_signals)}")
            lines.append(f"  SELL: {' + '.join(sell_signals)}")
            lines.append(f"  CAGR: {format_pct(row['median_annualized_return'])} | Calmar: {format_number(row['median_calmar_ratio'])}")
            lines.append(f"  Beat B&H: {row['symbols_beating_benchmark']}/{row['symbols_tested']} ({beat_pct:.0f}%) | Consistency: {format_ratio_as_pct(row['consistency_score'])}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_risk_adjusted_rankings(arguments: dict[str, Any]) -> list[TextContent]:
    """Get strategies ranked by risk-adjusted metrics."""
    run_id = arguments.get("run_id")
    min_calmar = arguments.get("min_calmar", 1.0)
    max_drawdown = arguments.get("max_drawdown", 20)
    trade_timing = arguments.get("trade_timing", "conservative")
    limit = arguments.get("limit", 20)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        query = """
            SELECT
                ar.strategy_id,
                s.buy_signals,
                s.sell_signals,
                ar.median_calmar_ratio,
                ar.median_return_over_maxdd,
                ar.median_max_drawdown_pct,
                ar.median_avg_drawdown_pct,
                ar.median_annualized_return,
                ar.median_expectancy,
                ar.median_win_rate,
                ar.consistency_score,
                sr.period_name
            FROM aggregated_results ar
            JOIN strategies s ON ar.strategy_id = s.strategy_id
            JOIN search_runs sr ON ar.search_run_id = sr.id
            WHERE ar.trade_timing = ?
              AND COALESCE(ar.median_calmar_ratio, 0) >= ?
              AND COALESCE(ar.median_max_drawdown_pct, 100) <= ?
        """
        params = [trade_timing, min_calmar, max_drawdown]

        if run_id:
            query += " AND ar.search_run_id = ?"
            params.append(run_id)

        query += " ORDER BY ar.median_calmar_ratio DESC NULLS LAST LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text=f"No strategies found with Calmar >= {min_calmar} and Max DD <= {max_drawdown}%")]

        lines = [
            f"Risk-Adjusted Rankings (Calmar >= {min_calmar}, Max DD <= {max_drawdown}%)",
            "=" * 80,
            ""
        ]

        for i, row in enumerate(rows, 1):
            buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
            sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []

            lines.append(f"#{i} Calmar: {format_number(row['median_calmar_ratio'])} [{row['period_name'] or 'Custom'}]")
            lines.append(f"  BUY: {' + '.join(buy_signals)}")
            lines.append(f"  SELL: {' + '.join(sell_signals)}")
            lines.append(f"  CAGR: {format_pct(row['median_annualized_return'])} | Max DD: {format_pct(row['median_max_drawdown_pct'])} | Avg DD: {format_pct(row['median_avg_drawdown_pct'])}")
            lines.append(f"  Expectancy: {format_pct(row['median_expectancy'])} | Win Rate: {format_pct(row['median_win_rate'])}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_strategy_symbol_breakdown(arguments: dict[str, Any]) -> list[TextContent]:
    """Get per-symbol performance breakdown."""
    strategy_id = arguments["strategy_id"]
    run_id = arguments.get("run_id")
    trade_timing = arguments.get("trade_timing", "conservative")
    sort_by = arguments.get("sort_by", "expectancy")

    sort_col_map = {
        'expectancy': 'expectancy',
        'alpha': 'alpha_vs_buy_hold',
        'win_rate': 'win_rate',
        'calmar_ratio': 'calmar_ratio'
    }
    sort_col = sort_col_map.get(sort_by, 'expectancy')

    conn = get_connection()
    try:
        cursor = conn.cursor()

        query = f"""
            SELECT
                sr.symbol_code,
                sr.exchange_code,
                sr.total_trades,
                sr.win_rate,
                sr.expectancy,
                sr.net_profit,
                sr.max_drawdown_pct,
                sr.buy_hold_return,
                sr.alpha_vs_buy_hold,
                sr.annualized_return,
                sr.calmar_ratio,
                sr.time_in_market_pct
            FROM symbol_results sr
            WHERE sr.strategy_id = ?
              AND sr.trade_timing = ?
        """
        params = [strategy_id, trade_timing]

        if run_id:
            query += " AND sr.search_run_id = ?"
            params.append(run_id)

        query += f" ORDER BY sr.{sort_col} DESC NULLS LAST"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text=f"No symbol results found for strategy: {strategy_id}")]

        lines = [
            f"Symbol Breakdown: {strategy_id}",
            f"Trade Timing: {trade_timing} | Sorted by: {sort_by}",
            "=" * 90,
            ""
        ]

        # Summary
        expectancies = [r['expectancy'] for r in rows if r['expectancy'] is not None]
        alphas = [r['alpha_vs_buy_hold'] for r in rows if r['alpha_vs_buy_hold'] is not None]
        profitable = len([r for r in rows if r['net_profit'] and r['net_profit'] > 0])
        beat_bh = len([r for r in rows if r['alpha_vs_buy_hold'] and r['alpha_vs_buy_hold'] > 0])

        lines.append(f"Summary: {len(rows)} symbols, {profitable} profitable ({profitable/len(rows)*100:.0f}%), {beat_bh} beat B&H ({beat_bh/len(rows)*100:.0f}%)")
        if expectancies:
            lines.append(f"Expectancy Range: {min(expectancies):.2f}% to {max(expectancies):.2f}%")
        lines.append("")

        lines.append(f"{'Symbol':<10} {'Trades':>7} {'WinRate':>8} {'Expect':>8} {'MaxDD':>8} {'B&H':>8} {'Alpha':>8} {'Calmar':>7}")
        lines.append("-" * 90)

        for row in rows:
            lines.append(
                f"{row['symbol_code']:<10} "
                f"{row['total_trades'] or 0:>7} "
                f"{format_pct(row['win_rate']):>8} "
                f"{format_pct(row['expectancy']):>8} "
                f"{format_pct(row['max_drawdown_pct']):>8} "
                f"{format_pct(row['buy_hold_return']):>8} "
                f"{format_pct(row['alpha_vs_buy_hold']):>8} "
                f"{format_number(row['calmar_ratio']):>7}"
            )

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_signal_performance_summary(arguments: dict[str, Any]) -> list[TextContent]:
    """Get aggregated performance by signal type."""
    signal_type = arguments.get("signal_type", "buy")
    trade_timing = arguments.get("trade_timing", "conservative")

    signal_col = 'buy_signals' if signal_type == 'buy' else 'sell_signals'

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Get all strategies with their signals and performance
        cursor.execute(f"""
            SELECT
                s.{signal_col},
                ar.median_expectancy,
                ar.median_win_rate,
                ar.consistency_score,
                ar.median_alpha
            FROM strategies s
            JOIN aggregated_results ar ON s.strategy_id = ar.strategy_id
            WHERE ar.trade_timing = ?
              AND ar.median_expectancy IS NOT NULL
        """, (trade_timing,))
        rows = cursor.fetchall()

        # Aggregate by signal
        signal_stats = {}
        for row in rows:
            signals = json.loads(row[signal_col]) if row[signal_col] else []
            for signal in signals:
                if signal not in signal_stats:
                    signal_stats[signal] = {
                        'count': 0,
                        'expectancies': [],
                        'win_rates': [],
                        'consistencies': [],
                        'alphas': []
                    }
                signal_stats[signal]['count'] += 1
                if row['median_expectancy'] is not None:
                    signal_stats[signal]['expectancies'].append(row['median_expectancy'])
                if row['median_win_rate'] is not None:
                    signal_stats[signal]['win_rates'].append(row['median_win_rate'])
                if row['consistency_score'] is not None:
                    signal_stats[signal]['consistencies'].append(row['consistency_score'])
                if row['median_alpha'] is not None:
                    signal_stats[signal]['alphas'].append(row['median_alpha'])

        # Calculate averages and sort by average expectancy
        signal_summaries = []
        for signal, stats in signal_stats.items():
            avg_exp = statistics.mean(stats['expectancies']) if stats['expectancies'] else 0
            avg_wr = statistics.mean(stats['win_rates']) if stats['win_rates'] else 0
            avg_cons = statistics.mean(stats['consistencies']) if stats['consistencies'] else 0
            avg_alpha = statistics.mean(stats['alphas']) if stats['alphas'] else 0
            signal_summaries.append({
                'signal': signal,
                'count': stats['count'],
                'avg_expectancy': avg_exp,
                'avg_win_rate': avg_wr,
                'avg_consistency': avg_cons,
                'avg_alpha': avg_alpha
            })

        signal_summaries.sort(key=lambda x: x['avg_expectancy'], reverse=True)

        lines = [
            f"Signal Performance Summary ({signal_type.upper()} signals)",
            "=" * 80,
            "",
            f"{'Signal':<35} {'Count':>6} {'AvgExp':>8} {'AvgWR':>8} {'AvgCons':>8} {'AvgAlpha':>9}",
            "-" * 80,
        ]

        for s in signal_summaries:
            lines.append(
                f"{s['signal']:<35} "
                f"{s['count']:>6} "
                f"{s['avg_expectancy']:>7.2f}% "
                f"{s['avg_win_rate']:>7.1f}% "
                f"{s['avg_consistency']*100:>7.0f}% "
                f"{s['avg_alpha']:>8.2f}%"
            )

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_schema(arguments: dict[str, Any]) -> list[TextContent]:
    """Get database schema information."""
    table_name = arguments.get("table_name")

    conn = get_connection()
    try:
        cursor = conn.cursor()

        if table_name:
            # Get schema for specific table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            if not columns:
                return [TextContent(type="text", text=f"Table not found: {table_name}")]

            lines = [
                f"Schema for table: {table_name}",
                "=" * 70,
                "",
                f"{'Column':<30} {'Type':<15} {'Nullable':>10} {'PK':>5}",
                "-" * 70,
            ]

            for col in columns:
                nullable = "NULL" if not col['notnull'] else "NOT NULL"
                pk = "PK" if col['pk'] else ""
                lines.append(f"{col['name']:<30} {col['type'] or 'ANY':<15} {nullable:>10} {pk:>5}")

            return [TextContent(type="text", text="\n".join(lines))]
        else:
            # Get all tables and their schemas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row['name'] for row in cursor.fetchall()]

            lines = [
                "Database Schema",
                "=" * 80,
                "",
                f"Tables: {len(tables)}",
                "",
            ]

            for tbl in tables:
                cursor.execute(f"PRAGMA table_info({tbl})")
                columns = cursor.fetchall()

                lines.append(f"TABLE: {tbl}")
                lines.append("-" * 40)
                for col in columns:
                    nullable = "" if not col['notnull'] else " NOT NULL"
                    pk = " [PK]" if col['pk'] else ""
                    lines.append(f"  {col['name']}: {col['type'] or 'ANY'}{nullable}{pk}")
                lines.append("")

            # Also show common column mappings for JOINs
            lines.append("=" * 80)
            lines.append("Common JOIN Patterns:")
            lines.append("-" * 40)
            lines.append("  search_runs.id = aggregated_results.search_run_id")
            lines.append("  search_runs.id = symbol_results.search_run_id")
            lines.append("  strategies.strategy_id = aggregated_results.strategy_id")
            lines.append("  strategies.strategy_id = symbol_results.strategy_id")
            lines.append("")
            lines.append("Strategy ID Format:")
            lines.append("  buy_signal1+signal2__sell_signal1+signal2")
            lines.append("  Example: buy_adm_momentum_rising__sell_adm_acceleration_falling")

            return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_list_strategy_ids(arguments: dict[str, Any]) -> list[TextContent]:
    """List strategy IDs with optional pattern filtering."""
    pattern = arguments.get("pattern")
    limit = arguments.get("limit", 50)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        if pattern:
            cursor.execute("""
                SELECT strategy_id, buy_signals, sell_signals
                FROM strategies
                WHERE strategy_id LIKE ?
                ORDER BY strategy_id
                LIMIT ?
            """, (f"%{pattern}%", limit))
        else:
            cursor.execute("""
                SELECT strategy_id, buy_signals, sell_signals
                FROM strategies
                ORDER BY strategy_id
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text=f"No strategy IDs found" + (f" matching '{pattern}'" if pattern else ""))]

        lines = [
            f"Strategy IDs" + (f" matching '{pattern}'" if pattern else ""),
            "=" * 80,
            "",
            "Format: buy_signal1+signal2__sell_signal1+signal2",
            "",
            f"Found {len(rows)} strategies:",
            "-" * 80,
            ""
        ]

        for row in rows:
            buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
            sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []

            lines.append(f"ID: {row['strategy_id']}")
            lines.append(f"    BUY:  {' + '.join(buy_signals)}")
            lines.append(f"    SELL: {' + '.join(sell_signals)}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_run_custom_query(arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a custom SQL query."""
    query = arguments["query"]
    limit = min(arguments.get("limit", 100), 500)

    # Basic safety check - only allow SELECT and PRAGMA (read-only)
    query_upper = query.strip().upper()
    is_pragma = query_upper.startswith("PRAGMA")
    is_select = query_upper.startswith("SELECT")

    if not (is_select or is_pragma):
        return [TextContent(type="text", text="Error: Only SELECT queries and PRAGMA commands are allowed.")]

    # Disallow potentially dangerous operations
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
    for word in forbidden:
        if word in query_upper:
            return [TextContent(type="text", text=f"Error: {word} operation not allowed.")]

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Add LIMIT if not present (only for SELECT, not PRAGMA)
        if is_select and 'LIMIT' not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return [TextContent(type="text", text="Query returned no results.")]

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Format as table
        lines = [
            "Custom Query Results",
            "=" * 80,
            "",
            " | ".join(columns),
            "-" * 80,
        ]

        for row in rows:
            values = [str(v) if v is not None else 'NULL' for v in row]
            lines.append(" | ".join(values))

        lines.append("")
        lines.append(f"({len(rows)} rows)")

        return [TextContent(type="text", text="\n".join(lines))]
    except sqlite3.Error as e:
        return [TextContent(type="text", text=f"SQL Error: {str(e)}")]
    finally:
        conn.close()


async def handle_compare_timing_modes(arguments: dict[str, Any]) -> list[TextContent]:
    """Compare Conservative vs Aggressive timing modes side-by-side."""
    run_id = arguments.get("run_id")
    strategy_id = arguments.get("strategy_id")
    sort_by = arguments.get("sort_by", "expectancy_delta")
    min_trades_per_year = arguments.get("min_trades_per_year", 5)
    limit = arguments.get("limit", 20)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # If specific strategy is requested
        if strategy_id:
            query = """
                SELECT
                    cons.strategy_id,
                    s.buy_signals,
                    s.sell_signals,
                    sr.period_name,
                    -- Conservative metrics
                    cons.median_expectancy as cons_expectancy,
                    cons.median_win_rate as cons_win_rate,
                    cons.median_profit_factor as cons_profit_factor,
                    cons.median_avg_trades_per_year as cons_trades_per_year,
                    cons.median_calmar_ratio as cons_calmar,
                    cons.median_alpha as cons_alpha,
                    cons.consistency_score as cons_consistency,
                    -- Aggressive metrics
                    aggr.median_expectancy as aggr_expectancy,
                    aggr.median_win_rate as aggr_win_rate,
                    aggr.median_profit_factor as aggr_profit_factor,
                    aggr.median_avg_trades_per_year as aggr_trades_per_year,
                    aggr.median_calmar_ratio as aggr_calmar,
                    aggr.median_alpha as aggr_alpha,
                    aggr.consistency_score as aggr_consistency
                FROM aggregated_results cons
                JOIN aggregated_results aggr ON cons.strategy_id = aggr.strategy_id
                    AND cons.search_run_id = aggr.search_run_id
                JOIN strategies s ON cons.strategy_id = s.strategy_id
                JOIN search_runs sr ON cons.search_run_id = sr.id
                WHERE cons.trade_timing = 'conservative'
                  AND aggr.trade_timing = 'aggressive'
                  AND cons.strategy_id = ?
                ORDER BY sr.period_name
            """
            params = [strategy_id]
            if run_id:
                query = query.replace("ORDER BY sr.period_name", "AND cons.search_run_id = ? ORDER BY sr.period_name")
                params.append(run_id)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return [TextContent(type="text", text=f"No timing comparison data found for strategy: {strategy_id}")]

            lines = [
                f"Timing Mode Comparison: {strategy_id}",
                "=" * 100,
                "",
            ]

            for row in rows:
                buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
                sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []

                exp_delta = (row['aggr_expectancy'] or 0) - (row['cons_expectancy'] or 0)
                wr_delta = (row['aggr_win_rate'] or 0) - (row['cons_win_rate'] or 0)
                calmar_delta = (row['aggr_calmar'] or 0) - (row['cons_calmar'] or 0)

                lines.append(f"Period: {row['period_name'] or 'Custom'}")
                lines.append(f"  BUY:  {' + '.join(buy_signals)}")
                lines.append(f"  SELL: {' + '.join(sell_signals)}")
                lines.append("")
                lines.append(f"  {'Metric':<20} {'CONS':>12} {'AGGR':>12} {'Delta':>12}")
                lines.append(f"  {'-'*56}")
                lines.append(f"  {'Expectancy':<20} {format_pct(row['cons_expectancy']):>12} {format_pct(row['aggr_expectancy']):>12} {exp_delta:>+11.2f}%")
                lines.append(f"  {'Win Rate':<20} {format_pct(row['cons_win_rate']):>12} {format_pct(row['aggr_win_rate']):>12} {wr_delta:>+11.2f}%")
                lines.append(f"  {'Profit Factor':<20} {format_number(row['cons_profit_factor']):>12} {format_number(row['aggr_profit_factor']):>12}")
                lines.append(f"  {'Calmar Ratio':<20} {format_number(row['cons_calmar']):>12} {format_number(row['aggr_calmar']):>12} {calmar_delta:>+11.2f}")
                lines.append(f"  {'Alpha vs B&H':<20} {format_pct(row['cons_alpha']):>12} {format_pct(row['aggr_alpha']):>12}")
                lines.append(f"  {'Trades/Year':<20} {format_number(row['cons_trades_per_year']):>12} {format_number(row['aggr_trades_per_year']):>12}")
                lines.append(f"  {'Consistency':<20} {format_ratio_as_pct(row['cons_consistency']):>12} {format_ratio_as_pct(row['aggr_consistency']):>12}")
                lines.append("")

            return [TextContent(type="text", text="\n".join(lines))]

        # Otherwise, find strategies with biggest timing impact
        else:
            run_filter = "AND cons.search_run_id = ?" if run_id else ""

            # Determine sort expression
            sort_expressions = {
                'expectancy_delta': 'ABS(aggr.median_expectancy - cons.median_expectancy) DESC',
                'cons_expectancy': 'cons.median_expectancy DESC',
                'aggr_expectancy': 'aggr.median_expectancy DESC',
                'win_rate_delta': 'ABS(aggr.median_win_rate - cons.median_win_rate) DESC',
            }
            sort_expr = sort_expressions.get(sort_by, sort_expressions['expectancy_delta'])

            query = f"""
                SELECT
                    cons.strategy_id,
                    s.buy_signals,
                    s.sell_signals,
                    sr.period_name,
                    cons.median_expectancy as cons_expectancy,
                    cons.median_win_rate as cons_win_rate,
                    cons.median_profit_factor as cons_profit_factor,
                    cons.median_avg_trades_per_year as cons_trades_per_year,
                    cons.median_calmar_ratio as cons_calmar,
                    cons.consistency_score as cons_consistency,
                    aggr.median_expectancy as aggr_expectancy,
                    aggr.median_win_rate as aggr_win_rate,
                    aggr.median_profit_factor as aggr_profit_factor,
                    aggr.median_avg_trades_per_year as aggr_trades_per_year,
                    aggr.median_calmar_ratio as aggr_calmar,
                    aggr.consistency_score as aggr_consistency,
                    (aggr.median_expectancy - cons.median_expectancy) as expectancy_delta,
                    (aggr.median_win_rate - cons.median_win_rate) as win_rate_delta
                FROM aggregated_results cons
                JOIN aggregated_results aggr ON cons.strategy_id = aggr.strategy_id
                    AND cons.search_run_id = aggr.search_run_id
                JOIN strategies s ON cons.strategy_id = s.strategy_id
                JOIN search_runs sr ON cons.search_run_id = sr.id
                WHERE cons.trade_timing = 'conservative'
                  AND aggr.trade_timing = 'aggressive'
                  AND (cons.median_avg_trades_per_year >= ? OR cons.median_avg_trades_per_year IS NULL)
                  AND (aggr.median_avg_trades_per_year >= ? OR aggr.median_avg_trades_per_year IS NULL)
                  {run_filter}
                ORDER BY {sort_expr}
                LIMIT ?
            """

            params = [min_trades_per_year, min_trades_per_year]
            if run_id:
                params.append(run_id)
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return [TextContent(type="text", text="No timing comparison data found. Ensure both timing modes are in the database.")]

            lines = [
                f"Timing Mode Comparison (sorted by {sort_by})",
                "=" * 100,
                f"Min trades/year: {min_trades_per_year}",
                "",
                f"{'#':<3} {'Period':<25} {'CONS Exp':>10} {'AGGR Exp':>10} {'Delta':>10} {'CONS WR':>9} {'AGGR WR':>9}",
                "-" * 100,
            ]

            for i, row in enumerate(rows, 1):
                buy_signals = json.loads(row['buy_signals']) if row['buy_signals'] else []
                sell_signals = json.loads(row['sell_signals']) if row['sell_signals'] else []

                exp_delta = row['expectancy_delta'] or 0
                delta_str = f"{exp_delta:>+9.2f}%" if exp_delta != 0 else "     0.00%"

                lines.append("")
                lines.append(f"#{i:<2} [{row['period_name'] or 'Custom'}]")
                lines.append(f"    BUY:  {' + '.join(buy_signals)}")
                lines.append(f"    SELL: {' + '.join(sell_signals)}")
                lines.append(f"    CONS: Exp {format_pct(row['cons_expectancy']):>8} | WR {format_pct(row['cons_win_rate']):>6} | PF {format_number(row['cons_profit_factor']):>5} | Calmar {format_number(row['cons_calmar']):>5}")
                lines.append(f"    AGGR: Exp {format_pct(row['aggr_expectancy']):>8} | WR {format_pct(row['aggr_win_rate']):>6} | PF {format_number(row['aggr_profit_factor']):>5} | Calmar {format_number(row['aggr_calmar']):>5}")
                lines.append(f"    Delta: Expectancy {delta_str} | Win Rate {(row['win_rate_delta'] or 0):>+.1f}%")

            return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_capital_deployment_analysis(arguments: dict[str, Any]) -> list[TextContent]:
    """Analyze portfolio-level capital deployment for a strategy."""
    strategy_id = arguments["strategy_id"]
    compare_strategy_id = arguments.get("compare_strategy_id")
    trade_timing = arguments.get("trade_timing", "conservative")
    run_id = arguments.get("run_id")

    conn = get_connection()
    try:
        cursor = conn.cursor()

        def get_strategy_deployment(strat_id: str) -> list[dict]:
            """Get deployment stats for a single strategy across periods."""
            query = """
                SELECT
                    sr.period_name,
                    sr.start_date,
                    sr.end_date,
                    sr.years_duration,
                    ar.symbols_tested,
                    ar.median_time_in_market_pct,
                    ar.median_avg_trades_per_year,
                    ar.median_expectancy,
                    ar.median_annualized_return,
                    ar.median_alpha,
                    ar.consistency_score
                FROM aggregated_results ar
                JOIN search_runs sr ON ar.search_run_id = sr.id
                WHERE ar.strategy_id = ?
                  AND ar.trade_timing = ?
            """
            params = [strat_id, trade_timing]

            if run_id:
                query += " AND ar.search_run_id = ?"
                params.append(run_id)

            query += " ORDER BY sr.start_date"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                symbols = row['symbols_tested'] or 0
                time_in_market = row['median_time_in_market_pct'] or 0

                # Estimated average simultaneous positions
                # If 100 symbols and 25% time-in-market, expect ~25 positions on average
                est_positions = symbols * time_in_market / 100 if symbols > 0 else 0

                results.append({
                    'period': row['period_name'] or 'Custom',
                    'start_date': row['start_date'],
                    'end_date': row['end_date'],
                    'years': row['years_duration'],
                    'symbols_tested': symbols,
                    'time_in_market_pct': time_in_market,
                    'est_avg_positions': est_positions,
                    'trades_per_year': row['median_avg_trades_per_year'],
                    'expectancy': row['median_expectancy'],
                    'cagr': row['median_annualized_return'],
                    'alpha': row['median_alpha'],
                    'consistency': row['consistency_score']
                })

            return results

        # Get strategy info
        cursor.execute("SELECT buy_signals, sell_signals FROM strategies WHERE strategy_id = ?", (strategy_id,))
        strat_row = cursor.fetchone()
        if not strat_row:
            return [TextContent(type="text", text=f"Strategy not found: {strategy_id}\n\nUse list_strategy_ids tool to discover valid strategy IDs.")]

        buy_signals = json.loads(strat_row['buy_signals']) if strat_row['buy_signals'] else []
        sell_signals = json.loads(strat_row['sell_signals']) if strat_row['sell_signals'] else []

        primary_data = get_strategy_deployment(strategy_id)

        if not primary_data:
            return [TextContent(type="text", text=f"No deployment data found for strategy: {strategy_id} with timing={trade_timing}")]

        lines = [
            "Capital Deployment Analysis",
            "=" * 90,
            "",
            f"Primary Strategy: {strategy_id}",
            f"  BUY:  {' + '.join(buy_signals)}",
            f"  SELL: {' + '.join(sell_signals)}",
            f"  Trade Timing: {trade_timing}",
            "",
        ]

        # Calculate summary statistics
        avg_time_in_market = statistics.mean([d['time_in_market_pct'] for d in primary_data])
        avg_est_positions = statistics.mean([d['est_avg_positions'] for d in primary_data])
        total_symbols = primary_data[0]['symbols_tested']  # Assume consistent across periods

        lines.append("Summary Across All Periods:")
        lines.append(f"  Average Time-in-Market: {avg_time_in_market:.1f}%")
        lines.append(f"  Estimated Avg Positions: {avg_est_positions:.1f} (of {total_symbols} symbols)")
        lines.append(f"  Capital Utilization: ~{avg_est_positions/total_symbols*100:.0f}% of portfolio capacity")
        lines.append("")

        # If comparison strategy is provided
        compare_data = None
        if compare_strategy_id:
            cursor.execute("SELECT buy_signals, sell_signals FROM strategies WHERE strategy_id = ?", (compare_strategy_id,))
            comp_row = cursor.fetchone()
            if comp_row:
                compare_data = get_strategy_deployment(compare_strategy_id)
                comp_buy = json.loads(comp_row['buy_signals']) if comp_row['buy_signals'] else []
                comp_sell = json.loads(comp_row['sell_signals']) if comp_row['sell_signals'] else []

                lines.append(f"Comparison Strategy: {compare_strategy_id}")
                lines.append(f"  BUY:  {' + '.join(comp_buy)}")
                lines.append(f"  SELL: {' + '.join(comp_sell)}")
                lines.append("")

                if compare_data:
                    comp_avg_time = statistics.mean([d['time_in_market_pct'] for d in compare_data])
                    comp_avg_positions = statistics.mean([d['est_avg_positions'] for d in compare_data])

                    lines.append("Comparison Summary:")
                    lines.append(f"  Average Time-in-Market: {comp_avg_time:.1f}%")
                    lines.append(f"  Estimated Avg Positions: {comp_avg_positions:.1f}")
                    lines.append("")

                    # Deployment advantage
                    deployment_ratio = avg_est_positions / comp_avg_positions if comp_avg_positions > 0 else float('inf')
                    if deployment_ratio > 1:
                        lines.append(f"  >> Primary deploys {deployment_ratio:.1f}x MORE capital than Comparison")
                    else:
                        lines.append(f"  >> Comparison deploys {1/deployment_ratio:.1f}x MORE capital than Primary")
                    lines.append("")

        # Period-by-period breakdown
        lines.append("-" * 90)
        lines.append("Period-by-Period Breakdown:")
        lines.append("")

        if compare_data:
            # Side-by-side comparison table
            lines.append(f"{'Period':<25} {'Primary TiM':>12} {'Primary Pos':>12} {'Comp TiM':>12} {'Comp Pos':>12} {'Ratio':>8}")
            lines.append("-" * 90)

            # Match periods by name
            comp_by_period = {d['period']: d for d in compare_data}

            for p in primary_data:
                comp_p = comp_by_period.get(p['period'])
                if comp_p:
                    ratio = p['est_avg_positions'] / comp_p['est_avg_positions'] if comp_p['est_avg_positions'] > 0 else float('inf')
                    ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "N/A"
                    lines.append(
                        f"{p['period']:<25} "
                        f"{p['time_in_market_pct']:>11.1f}% "
                        f"{p['est_avg_positions']:>12.1f} "
                        f"{comp_p['time_in_market_pct']:>11.1f}% "
                        f"{comp_p['est_avg_positions']:>12.1f} "
                        f"{ratio_str:>8}"
                    )
                else:
                    lines.append(
                        f"{p['period']:<25} "
                        f"{p['time_in_market_pct']:>11.1f}% "
                        f"{p['est_avg_positions']:>12.1f} "
                        f"{'N/A':>12} "
                        f"{'N/A':>12} "
                        f"{'N/A':>8}"
                    )

            lines.append("")
            lines.append("(Ratio = Primary positions / Comparison positions)")
        else:
            # Single strategy detailed breakdown
            lines.append(f"{'Period':<25} {'TiM %':>10} {'Est Pos':>10} {'Trades/Yr':>10} {'Expect':>10} {'CAGR':>10} {'Alpha':>10}")
            lines.append("-" * 90)

            for p in primary_data:
                lines.append(
                    f"{p['period']:<25} "
                    f"{p['time_in_market_pct']:>9.1f}% "
                    f"{p['est_avg_positions']:>10.1f} "
                    f"{format_number(p['trades_per_year']):>10} "
                    f"{format_pct(p['expectancy']):>10} "
                    f"{format_pct(p['cagr']):>10} "
                    f"{format_pct(p['alpha']):>10}"
                )

        lines.append("")
        lines.append("Interpretation:")
        lines.append("  - Time-in-Market (TiM): % of trading days with an active position (per symbol)")
        lines.append("  - Est Pos: Estimated simultaneous positions = symbols × TiM%")
        lines.append("  - Higher TiM = more capital deployed = more exposure to market moves")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


async def handle_get_daily_position_counts(arguments: dict[str, Any]) -> list[TextContent]:
    """Get daily position counts from trade-level data for time-series graphing."""
    strategy_id = arguments["strategy_id"]
    run_id = arguments["run_id"]
    trade_timing = arguments.get("trade_timing", "conservative")
    compare_strategy_id = arguments.get("compare_strategy_id")
    output_format = arguments.get("output_format", "summary")

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Check if trade_results table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='trade_results'
        """)
        if not cursor.fetchone():
            return [TextContent(type="text", text=(
                "Trade-level data not available.\n\n"
                "The trade_results table doesn't exist yet. This feature requires:\n"
                "1. Strategy search runs performed after Dec 2025\n"
                "2. Re-running existing searches to populate trade data\n\n"
                "Use get_capital_deployment_analysis for estimated position counts from aggregated data."
            ))]

        def get_position_counts(strat_id: str) -> dict:
            """Get position counts for a strategy."""
            # Get all trades for this strategy/run/timing
            cursor.execute("""
                SELECT entry_date, exit_date, symbol_code
                FROM trade_results
                WHERE search_run_id = ?
                  AND strategy_id = ?
                  AND trade_timing = ?
                ORDER BY entry_date
            """, (run_id, strat_id, trade_timing))
            trades = cursor.fetchall()

            if not trades:
                return {'trades': [], 'daily_counts': {}, 'stats': {}}

            # Build daily position counts
            from datetime import datetime, timedelta

            # Get date range from trades
            all_entry_dates = [t['entry_date'][:10] for t in trades]
            all_exit_dates = [t['exit_date'][:10] for t in trades]
            min_date = min(all_entry_dates)
            max_date = max(all_exit_dates)

            # Generate all trading days in range
            start_dt = datetime.strptime(min_date, '%Y-%m-%d')
            end_dt = datetime.strptime(max_date, '%Y-%m-%d')

            daily_counts = {}
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime('%Y-%m-%d')
                # Count positions open on this date
                count = sum(
                    1 for t in trades
                    if t['entry_date'][:10] <= date_str <= t['exit_date'][:10]
                )
                daily_counts[date_str] = count
                current_dt += timedelta(days=1)

            # Calculate statistics
            counts = list(daily_counts.values())
            non_zero_counts = [c for c in counts if c > 0]

            stats = {
                'total_days': len(counts),
                'days_with_positions': len(non_zero_counts),
                'days_at_zero': counts.count(0),
                'mean_positions': statistics.mean(counts) if counts else 0,
                'median_positions': statistics.median(counts) if counts else 0,
                'max_positions': max(counts) if counts else 0,
                'min_positions': min(counts) if counts else 0,
                'pct_days_at_zero': (counts.count(0) / len(counts) * 100) if counts else 0,
            }
            if non_zero_counts:
                stats['mean_when_invested'] = statistics.mean(non_zero_counts)

            return {
                'trades': trades,
                'daily_counts': daily_counts,
                'stats': stats
            }

        # Get primary strategy data
        primary = get_position_counts(strategy_id)

        if not primary['trades']:
            return [TextContent(type="text", text=(
                f"No trade data found for strategy: {strategy_id}\n"
                f"Run ID: {run_id}, Timing: {trade_timing}\n\n"
                "This could mean:\n"
                "1. The strategy had no trades in this period\n"
                "2. Trade-level data wasn't stored for this run (pre-Dec 2025)\n"
                "3. Wrong strategy_id or run_id"
            ))]

        # Get comparison strategy if requested
        compare = None
        if compare_strategy_id:
            compare = get_position_counts(compare_strategy_id)

        # Build output
        lines = [
            "Daily Position Count Analysis",
            "=" * 80,
            "",
            f"Strategy: {strategy_id}",
            f"Run ID: {run_id}",
            f"Trade Timing: {trade_timing}",
            "",
            "Position Count Statistics:",
            f"  Total Trading Days: {primary['stats']['total_days']}",
            f"  Days with Positions: {primary['stats']['days_with_positions']}",
            f"  Days at Zero: {primary['stats']['days_at_zero']} ({primary['stats']['pct_days_at_zero']:.1f}%)",
            f"  Mean Positions/Day: {primary['stats']['mean_positions']:.1f}",
            f"  Median Positions/Day: {primary['stats']['median_positions']:.0f}",
            f"  Max Concurrent Positions: {primary['stats']['max_positions']}",
        ]

        if 'mean_when_invested' in primary['stats']:
            lines.append(f"  Mean When Invested: {primary['stats']['mean_when_invested']:.1f}")

        # Add comparison if available
        if compare and compare['trades']:
            lines.append("")
            lines.append(f"Comparison Strategy: {compare_strategy_id}")
            lines.append(f"  Days with Positions: {compare['stats']['days_with_positions']}")
            lines.append(f"  Days at Zero: {compare['stats']['days_at_zero']} ({compare['stats']['pct_days_at_zero']:.1f}%)")
            lines.append(f"  Mean Positions/Day: {compare['stats']['mean_positions']:.1f}")
            lines.append(f"  Max Concurrent: {compare['stats']['max_positions']}")

            # Ratio comparison
            if compare['stats']['mean_positions'] > 0:
                ratio = primary['stats']['mean_positions'] / compare['stats']['mean_positions']
                lines.append(f"  >> Primary deploys {ratio:.1f}x {'MORE' if ratio > 1 else 'LESS'} capital")

        # Add time-series data if requested
        if output_format == 'daily':
            lines.append("")
            lines.append("-" * 80)
            lines.append("Daily Position Counts (last 60 days):")
            lines.append("")

            if compare and compare['trades']:
                lines.append(f"{'Date':<12} {'Primary':>10} {'Compare':>10} {'Diff':>10}")
                lines.append("-" * 50)
            else:
                lines.append(f"{'Date':<12} {'Positions':>10}")
                lines.append("-" * 25)

            # Show last 60 days
            sorted_dates = sorted(primary['daily_counts'].keys(), reverse=True)[:60]
            for date in reversed(sorted_dates):
                p_count = primary['daily_counts'].get(date, 0)
                if compare and compare['trades']:
                    c_count = compare['daily_counts'].get(date, 0)
                    diff = p_count - c_count
                    lines.append(f"{date:<12} {p_count:>10} {c_count:>10} {diff:>+10}")
                else:
                    lines.append(f"{date:<12} {p_count:>10}")

        elif output_format == 'weekly':
            lines.append("")
            lines.append("-" * 80)
            lines.append("Weekly Average Position Counts:")
            lines.append("")

            # Group by week
            from collections import defaultdict
            weekly = defaultdict(list)
            for date, count in primary['daily_counts'].items():
                dt = datetime.strptime(date, '%Y-%m-%d')
                week_start = (dt - timedelta(days=dt.weekday())).strftime('%Y-%m-%d')
                weekly[week_start].append(count)

            if compare and compare['trades']:
                compare_weekly = defaultdict(list)
                for date, count in compare['daily_counts'].items():
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    week_start = (dt - timedelta(days=dt.weekday())).strftime('%Y-%m-%d')
                    compare_weekly[week_start].append(count)

                lines.append(f"{'Week Start':<12} {'Primary Avg':>12} {'Compare Avg':>12}")
                lines.append("-" * 40)
                for week in sorted(weekly.keys()):
                    p_avg = statistics.mean(weekly[week])
                    c_avg = statistics.mean(compare_weekly.get(week, [0])) if compare_weekly else 0
                    lines.append(f"{week:<12} {p_avg:>12.1f} {c_avg:>12.1f}")
            else:
                lines.append(f"{'Week Start':<12} {'Avg Positions':>15}")
                lines.append("-" * 30)
                for week in sorted(weekly.keys()):
                    avg = statistics.mean(weekly[week])
                    lines.append(f"{week:<12} {avg:>15.1f}")

        lines.append("")
        lines.append("Note: Position count = number of symbols with open positions on each day")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


# Resources
@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="strategy://status",
            name="Database Status",
            description="Current strategy database statistics and status",
            mimeType="text/plain"
        ),
        Resource(
            uri="strategy://runs",
            name="Search Runs",
            description="List of all search runs in the database",
            mimeType="text/plain"
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    if uri == "strategy://status":
        result = await handle_get_database_status({})
        return result[0].text
    elif uri == "strategy://runs":
        result = await handle_list_search_runs({"limit": 50})
        return result[0].text
    else:
        raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
