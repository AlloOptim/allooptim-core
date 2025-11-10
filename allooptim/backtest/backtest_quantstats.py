"""QuantStats integration for professional portfolio analytics and reporting.

This module provides enhanced visualization and reporting capabilities using
the QuantStats library for institutional-grade performance analysis.

Key features:
- HTML tearsheet generation with interactive charts
- Advanced risk metrics (VaR, CVaR, Sortino, Calmar)
- Benchmark-relative performance analysis
- Rolling statistics visualization
- Monthly returns heatmaps
- Automated report generation
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import quantstats as qs

    QUANTSTATS_AVAILABLE = True
except ImportError:
    logger.info("QuantStats not available. Install with: pip install quantstats")
    QUANTSTATS_AVAILABLE = False

MAX_NAN_VALUES = 5
MIN_DATA_LENGTH = 2


def prepare_returns_for_quantstats(results: dict, optimizer_name: str) -> Optional[pd.Series]:
    """Extract and validate returns series for QuantStats analysis.

    Args:
        results: Backtest results dictionary
        optimizer_name: Name of optimizer to extract returns for

    Returns:
        Daily returns as pd.Series with datetime index, or None if unavailable
    """
    if optimizer_name not in results:
        logger.warning(f"Optimizer {optimizer_name} not found in results")
        return None

    returns = results[optimizer_name].get("returns")

    if returns is None or returns.empty:
        logger.warning(f"No returns data for {optimizer_name}")
        return None

    # Validate datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        logger.warning(f"Returns index is not DatetimeIndex for {optimizer_name}")
        return None

    # Remove any NaN and inf values
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns_clean) < MIN_DATA_LENGTH:
        logger.warning(f"Insufficient data points ({len(returns_clean)}) for {optimizer_name} after cleaning")
        return None

    # Ensure index is unique and sorted
    returns_clean = returns_clean[~returns_clean.index.duplicated(keep="first")]
    returns_clean = returns_clean.sort_index()

    # Remove timezone info if present
    if hasattr(returns_clean.index, "tz") and returns_clean.index.tz is not None:
        returns_clean.index = returns_clean.index.tz_localize(None)

    return returns_clean


def generate_tearsheet(
    results: dict, optimizer_name: str, benchmark: str = "SPY", output_path: Optional[Path] = None, mode: str = "full"
) -> bool:
    """Generate QuantStats HTML tearsheet for a single optimizer.

    Args:
        results: Backtest results dictionary
        optimizer_name: Name of optimizer to analyze
        benchmark: Benchmark ticker or returns series (default: "SPY")
        output_path: Path to save HTML file (None = auto-generate)
        mode: "basic" or "full" tearsheet (default: "full")

    Returns:
        True if successful, False otherwise
    """
    if not QUANTSTATS_AVAILABLE:
        logger.warning("QuantStats not available. Cannot generate tearsheet.")
        return False

    # Prepare returns
    returns = prepare_returns_for_quantstats(results, optimizer_name)
    if returns is None:
        return False

    # Determine output path
    if output_path is None:
        results_dir = Path("backtest_results") / "quantstats_reports"
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_name = optimizer_name.replace(" ", "_").replace("/", "_")
        output_path = results_dir / f"{safe_name}_tearsheet.html"

    try:
        # Generate tearsheet
        logger.info(f"Generating {mode} tearsheet for {optimizer_name}")

        # Extend pandas with QuantStats methods
        qs.extend_pandas()

        # Get benchmark returns if needed
        benchmark_arg = None
        if benchmark in results:
            benchmark_returns = prepare_returns_for_quantstats(results, benchmark)
            if benchmark_returns is not None:
                benchmark_arg = benchmark_returns

        if benchmark_arg is None:
            logger.warning(f"No benchmark available for {optimizer_name}, generating without benchmark")

        # Generate report - try with benchmark first, then without if it fails
        success = False

        if benchmark_arg is not None:
            try:
                if mode == "basic":
                    qs.reports.basic(
                        returns,
                        benchmark=benchmark_arg,
                        output=str(output_path),
                        title=f"{optimizer_name} Performance Analysis",
                    )
                else:  # full
                    qs.reports.html(
                        returns,
                        benchmark=benchmark_arg,
                        output=str(output_path),
                        title=f"{optimizer_name} Performance Analysis",
                    )
                logger.info(f"Tearsheet saved to {output_path} (with benchmark)")
                success = True
            except Exception as e:
                logger.warning(
                    f"Failed to generate tearsheet with benchmark for {optimizer_name}: {e}. Trying without benchmark."
                )

        if not success:
            # Generate without benchmark
            if mode == "basic":
                qs.reports.basic(returns, output=str(output_path), title=f"{optimizer_name} Performance Analysis")
            else:  # full
                qs.reports.html(returns, output=str(output_path), title=f"{optimizer_name} Performance Analysis")
            logger.info(f"Tearsheet saved to {output_path} (without benchmark)")
        return True

    except Exception as e:
        logger.error(f"Failed to generate tearsheet for {optimizer_name}: {e}")
        return False


def generate_comparative_tearsheets(
    results: dict, benchmark: str = "SPY", output_dir: Optional[Path] = None, top_n: int = 5
) -> Dict[str, bool]:
    """Generate tearsheets for top N performing optimizers.

    Args:
        results: Backtest results dictionary
        benchmark: Benchmark strategy name or ticker
        output_dir: Directory to save reports
        top_n: Number of top performers to analyze

    Returns:
        Dict mapping optimizer names to generation success status
    """
    if not QUANTSTATS_AVAILABLE:
        logger.warning("QuantStats not available.")
        return {}

    # Rank by Sharpe ratio
    ranked_optimizers = sorted(results.items(), key=lambda x: x[1]["metrics"].get("sharpe_ratio", -999), reverse=True)

    # Generate for top N
    status = {}
    for optimizer_name, _ in ranked_optimizers[:top_n]:
        if optimizer_name == benchmark:
            continue  # Skip benchmark itself

        output_path = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = optimizer_name.replace(" ", "_").replace("/", "_")
            output_path = output_dir / f"{safe_name}_tearsheet.html"

        success = generate_tearsheet(
            results,
            optimizer_name,
            benchmark=benchmark,  # Pass benchmark name, not returns
            output_path=output_path,
            mode="full",
        )
        status[optimizer_name] = success

    return status


def calculate_quantstats_metrics(results: dict, optimizer_name: str, benchmark: str = "SPY") -> Optional[Dict]:
    """Calculate QuantStats metrics without generating full tearsheet.

    Args:
        results: Backtest results dictionary
        optimizer_name: Name of optimizer
        benchmark: Benchmark ticker or returns

    Returns:
        Dictionary of QuantStats metrics, or None if unavailable
    """
    if not QUANTSTATS_AVAILABLE:
        return None

    returns = prepare_returns_for_quantstats(results, optimizer_name)
    if returns is None:
        return None

    try:
        # Get benchmark returns
        benchmark_returns = None
        if benchmark in results:
            benchmark_returns = prepare_returns_for_quantstats(results, benchmark)

        # Calculate metrics
        metrics = {
            # Risk-adjusted returns
            "sharpe": qs.stats.sharpe(returns),
            "sortino": qs.stats.sortino(returns),
            "calmar": qs.stats.calmar(returns),
            # Risk metrics
            "max_drawdown": qs.stats.max_drawdown(returns),
            "var_95": qs.stats.var(returns, confidence=0.95),
            "cvar_95": qs.stats.cvar(returns, confidence=0.95),
            "volatility": qs.stats.volatility(returns),
            # Return metrics
            "cagr": qs.stats.cagr(returns),
            "total_return": qs.stats.comp(returns),
            "best_day": qs.stats.best(returns),
            "worst_day": qs.stats.worst(returns),
            # Win/Loss
            "win_rate": qs.stats.win_rate(returns),
            "profit_factor": qs.stats.profit_factor(returns),
            "payoff_ratio": qs.stats.payoff_ratio(returns),
            # Advanced
            "ulcer_index": qs.stats.ulcer_index(returns),
            "tail_ratio": qs.stats.tail_ratio(returns),
            "common_sense_ratio": qs.stats.common_sense_ratio(returns),
        }

        # Benchmark-relative metrics (if available)
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            try:
                metrics.update(
                    {
                        "alpha": qs.stats.greeks(returns, benchmark_returns).get("alpha", None),
                        "beta": qs.stats.greeks(returns, benchmark_returns).get("beta", None),
                        "information_ratio": qs.stats.information_ratio(returns, benchmark_returns),
                        "r_squared": qs.stats.r_squared(returns, benchmark_returns),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to calculate benchmark-relative metrics for {optimizer_name}: {e}")

        return metrics

    except Exception as e:
        logger.error(f"Failed to calculate QuantStats metrics for {optimizer_name}: {e}")
        return None


def create_quantstats_reports(
    results: dict, output_dir: Path, generate_individual: bool = True, generate_top_n: int = 5, benchmark: str = "SPY"
) -> None:
    """Create QuantStats reports as part of backtesting pipeline.

    This function integrates with existing backtest_engine.py workflow.

    Args:
        results: Backtest results from BacktestEngine
        output_dir: Directory to save reports
        generate_individual: Generate tearsheet for each optimizer
        generate_top_n: Generate comparative analysis for top N performers
        benchmark: Benchmark ticker or strategy name
    """
    if not QUANTSTATS_AVAILABLE:
        logger.info("QuantStats not installed. Skipping QuantStats reports.")
        logger.info("Install with: poetry install --with visualizations")
        return

    # Check if we have sufficient data for meaningful QuantStats analysis
    sample_optimizer = next(iter(results.keys()))
    if sample_optimizer in results and "returns" in results[sample_optimizer]:
        sample_returns = results[sample_optimizer]["returns"]
        if sample_returns is not None and len(sample_returns.dropna()) < MAX_NAN_VALUES:
            logger.info("Insufficient data points for QuantStats analysis. Skipping reports.")
            return

    logger.info("Generating QuantStats reports...")  # Generate individual tearsheets
    if generate_individual:
        for optimizer_name in results:
            if optimizer_name == benchmark:
                continue  # Skip benchmark itself

            generate_tearsheet(
                results,
                optimizer_name,
                benchmark=benchmark,
                output_path=output_dir / f"{optimizer_name.replace(' ', '_')}_tearsheet.html",
                mode="full",
            )

    # Generate comparative analysis
    if generate_top_n > 0:
        logger.info(f"Generating comparative analysis for top {generate_top_n} optimizers")
        generate_comparative_tearsheets(results, benchmark=benchmark, output_dir=output_dir, top_n=generate_top_n)

    # Generate A2A vs Benchmark comparison
    if "A2AEnsemble" in results:
        logger.info("Generating A2A vs Benchmark comparison tearsheet")
        generate_tearsheet(
            results,
            "A2AEnsemble",
            benchmark=benchmark,
            output_path=output_dir / "A2A_vs_Benchmark_tearsheet.html",
            mode="full",
        )

    logger.info(f"QuantStats reports saved to {output_dir}")
