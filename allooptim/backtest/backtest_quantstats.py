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
import yfinance as yf


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


def _prepare_returns_for_quantstats(results: dict, optimizer_name: str) -> Optional[pd.Series]:
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

    if returns is None:
        logger.warning(f"Returns value is None for {optimizer_name}")
        return None
    
    if isinstance(returns, float) or isinstance(returns, int):
        logger.warning(f"Returns is a scalar ({returns}) for {optimizer_name}, not a series")
        return None

    if returns.empty:
        logger.warning(f"Returns series is empty for {optimizer_name}")
        return None

    # Validate datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        logger.warning(f"Returns index is not DatetimeIndex for {optimizer_name}: {type(returns.index)}")
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


def _fetch_benchmark_returns(benchmark: str, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None) -> Optional[pd.Series]:
    """Fetch benchmark returns from yfinance.

    Args:
        benchmark: Benchmark ticker (e.g., "SPY")
        start_date: Start date for benchmark data
        end_date: End date for benchmark data

    Returns:
        Daily returns as pd.Series with datetime index, or None if unavailable
    """
    try:
        logger.debug(f"Fetching benchmark data for {benchmark} from {start_date} to {end_date}")

        # Download price data
        data = yf.download(benchmark, start=start_date, end=end_date, progress=False)

        if data is None or data.empty:
            logger.warning(f"No data downloaded for benchmark {benchmark}")
            return None

        logger.debug(f"Downloaded {len(data)} rows of data for {benchmark}")
        logger.debug(f"Columns available: {data.columns.tolist()}")

        # Calculate returns from adjusted close prices
        # Handle both Series (single ticker) and DataFrame (multiple tickers)
        if isinstance(data, pd.Series):
            prices = data
        elif "Adj Close" in data.columns:
            prices = data["Adj Close"]
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]  # Take first column if multiple
        elif "Close" in data.columns:
            prices = data["Close"]
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]  # Take first column if multiple
        else:
            logger.warning(f"No price column found in downloaded data for {benchmark}")
            logger.warning(f"Available columns: {data.columns.tolist()}")
            return None

        # Calculate returns
        returns = prices.pct_change().dropna()

        if len(returns) < MIN_DATA_LENGTH:
            logger.warning(f"Insufficient data points for benchmark {benchmark}: {len(returns)}")
            return None

        # Remove timezone info if present
        if hasattr(returns.index, "tz") and returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)

        logger.debug(
            f"Successfully fetched {len(returns)} days of benchmark returns for {benchmark}"
        )
        return returns

    except Exception as e:
        logger.warning(f"Failed to fetch benchmark data for {benchmark}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def _generate_tearsheet(
    results: dict, optimizer_name: str, benchmark: str, output_path: Optional[Path] = None, mode: str = "full"
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
    returns = _prepare_returns_for_quantstats(results, optimizer_name)
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
        logger.debug(f"Generating {mode} tearsheet for {optimizer_name}")

        # Extend pandas with QuantStats methods
        qs.extend_pandas()

        # Get benchmark returns if needed
        benchmark_arg = None
        if benchmark in results:
            # Benchmark is in results (another optimizer or strategy)
            benchmark_returns = _prepare_returns_for_quantstats(results, benchmark)
            if benchmark_returns is not None:
                benchmark_arg = benchmark_returns
                logger.debug(f"Using benchmark from results: {benchmark}")
        else:
            # Try to fetch benchmark from yfinance
            logger.info(f"Benchmark '{benchmark}' not in results, fetching from yfinance")
            # Infer date range from returns index
            if returns.index is not None and len(returns.index) > 0:
                start_date = returns.index[0]
                end_date = returns.index[-1]
                logger.debug(f"Fetching {benchmark} from {start_date} to {end_date}")
                benchmark_arg = _fetch_benchmark_returns(benchmark, start_date=start_date, end_date=end_date)
            else:
                logger.warning(f"Cannot infer date range from returns index")
                benchmark_arg = _fetch_benchmark_returns(benchmark)

            if benchmark_arg is None:
                logger.warning(f"Failed to fetch benchmark {benchmark}, will generate tearsheet without it")

        if benchmark_arg is None:
            logger.warning(f"No benchmark available for {optimizer_name}, generating without benchmark")

        # Generate report - try with benchmark first, then without if it fails
        success = False

        if benchmark_arg is not None:
            try:
                logger.debug(
                    f"Passing benchmark to QuantStats: type={type(benchmark_arg)}, len={len(benchmark_arg) if hasattr(benchmark_arg, '__len__') else 'N/A'}"
                )

                # Align benchmark returns with portfolio returns
                # Ensure they have the same dates
                common_dates = returns.index.intersection(benchmark_arg.index)
                if len(common_dates) < MIN_DATA_LENGTH:
                    logger.warning(f"Not enough common dates between portfolio ({len(returns)}) and benchmark ({len(benchmark_arg)})")
                    logger.warning(f"Common dates: {len(common_dates)}")
                    benchmark_arg = None
                else:
                    returns_aligned = returns.loc[common_dates]
                    benchmark_aligned = benchmark_arg.loc[common_dates]
                    logger.debug(
                        f"Aligned returns and benchmark to {len(common_dates)} common dates"
                    )

                    if mode == "basic":
                        qs.reports.basic(
                            returns_aligned,
                            benchmark=benchmark_aligned,
                            output=str(output_path),
                            title=f"{optimizer_name} Performance Analysis",
                        )
                    else:  # full
                        qs.reports.html(
                            returns_aligned,
                            benchmark=benchmark_aligned,
                            output=str(output_path),
                            title=f"{optimizer_name} Performance Analysis",
                        )
                    logger.debug(f"Tearsheet saved to {output_path} (with benchmark)")
                    success = True
            except Exception as e:
                logger.warning(
                    f"Failed to generate tearsheet with benchmark for {optimizer_name}: {e}. Trying without benchmark."
                )
                import traceback
                logger.debug(traceback.format_exc())
                # Clear benchmark_arg to trigger fallback
                benchmark_arg = None

        if not success:
            # Generate without benchmark
            if mode == "basic":
                qs.reports.basic(returns, output=str(output_path), title=f"{optimizer_name} Performance Analysis")
            else:  # full
                qs.reports.html(returns, output=str(output_path), title=f"{optimizer_name} Performance Analysis")
            logger.debug(f"Tearsheet saved to {output_path} (without benchmark)")
        return True

    except Exception as e:
        logger.error(f"Failed to generate tearsheet for {optimizer_name}: {e}")
        return False


def _generate_comparative_tearsheets(
    results: dict, benchmark: str, output_dir: Optional[Path] = None, top_n: int = 5
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

        success = _generate_tearsheet(
            results,
            optimizer_name,
            benchmark=benchmark,  # Pass benchmark name, not returns
            output_path=output_path,
            mode="full",
        )
        status[optimizer_name] = success

    return status

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

    logger.debug("Generating QuantStats reports...")  # Generate individual tearsheets
    if generate_individual:
        for optimizer_name in results:
            if optimizer_name == benchmark:
                continue  # Skip benchmark itself

            _generate_tearsheet(
                results,
                optimizer_name,
                benchmark=benchmark,
                output_path=output_dir / f"{optimizer_name.replace(' ', '_')}_tearsheet.html",
                mode="full",
            )

    # Generate comparative analysis
    if generate_top_n > 0:
        logger.debug(
            f"Generating comparative analysis for top {generate_top_n} optimizers"
        )
        _generate_comparative_tearsheets(results, benchmark=benchmark, output_dir=output_dir, top_n=generate_top_n)

    # Generate A2A vs Benchmark comparison
    if "A2AEnsemble" in results:
        logger.debug("Generating A2A vs Benchmark comparison tearsheet")
        _generate_tearsheet(
            results,
            "A2AEnsemble",
            benchmark=benchmark,
            output_path=output_dir / "A2A_vs_Benchmark_tearsheet.html",
            mode="full",
        )

    logger.debug(f"QuantStats reports saved to {output_dir}")
