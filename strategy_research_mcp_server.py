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
    """Format a percentage value for display."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


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
                        "description": "Metric to sort by: median_expectancy, median_win_rate, median_profit_factor, median_calmar_ratio, median_alpha, consistency_score",
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
            name="run_custom_query",
            description="Execute a custom SQL query against the strategy database. For advanced users who need specific analysis not covered by other tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute (read-only)"
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
        elif name == "run_custom_query":
            return await handle_run_custom_query(arguments)
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
        'median_annualized_return', 'symbols_beating_benchmark'
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
                f"Consistency: {format_pct(row['consistency_score'], 0) if row['consistency_score'] else 'N/A'}"
            )
            if row['median_annualized_return'] is not None:
                beat_pct = (row['symbols_beating_benchmark'] / row['symbols_tested'] * 100) if row['symbols_tested'] else 0
                lines.append(
                    f"  CAGR: {format_pct(row['median_annualized_return'])} | "
                    f"Calmar: {format_number(row['median_calmar_ratio'])} | "
                    f"Alpha: {format_pct(row['median_alpha'])}"
                )
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
            return [TextContent(type="text", text=f"Strategy not found: {strategy_id}")]

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
                lines.append(f"  Consistency: {format_pct(row['consistency_score'], 0)}")
                if row['median_alpha'] is not None:
                    lines.append(f"  Alpha vs B&H: {format_pct(row['median_alpha'])} (B&H baseline: {format_pct(row['buy_hold_median_return'])})")

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
            lines.append(f"  Expectancy: {format_pct(row['median_expectancy'])} | Win Rate: {format_pct(row['median_win_rate'])} | Consistency: {format_pct(row['consistency_score'], 0)}")
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
            return [TextContent(type="text", text=f"No results found for strategy: {strategy_id}")]

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
            lines.append(f"  Alpha: {format_pct(row['median_alpha'])} | Beat B&H: {row['symbols_beating_benchmark'] or 0}/{row['symbols_tested']} ({beat_pct:.0f}%)")
            lines.append(f"  Consistency: {format_pct(row['consistency_score'], 0)} ({row['symbols_profitable']}/{row['symbols_tested']} profitable)")

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
            lines.append(f"  Variance: {variance:.2f}% | Avg Win Rate: {format_pct(row['avg_win_rate'])} | Avg Consistency: {format_pct(row['avg_consistency'], 0)}")
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
            lines.append(f"  Beat B&H: {row['symbols_beating_benchmark']}/{row['symbols_tested']} ({beat_pct:.0f}%) | Consistency: {format_pct(row['consistency_score'], 0)}")
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


async def handle_run_custom_query(arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a custom SQL query."""
    query = arguments["query"]
    limit = min(arguments.get("limit", 100), 500)

    # Basic safety check - only allow SELECT
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return [TextContent(type="text", text="Error: Only SELECT queries are allowed.")]

    # Disallow potentially dangerous operations
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
    for word in forbidden:
        if word in query_upper:
            return [TextContent(type="text", text=f"Error: {word} operation not allowed.")]

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Add LIMIT if not present
        if 'LIMIT' not in query_upper:
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
