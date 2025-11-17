"""Portfolio backtesting engine for evaluating optimization strategies.

This module provides comprehensive backtesting capabilities for portfolio
optimization algorithms. It simulates historical performance by running
optimizers on rolling windows of historical data, computing portfolio weights,
and tracking returns over time.

Key features:
- Rolling window backtesting with configurable lookback periods
- Multiple rebalancing frequencies (daily, weekly, monthly)
- Performance metrics calculation (Sharpe ratio, max drawdown, etc.)
- Risk-adjusted return analysis
- Transaction cost modeling
- Benchmark comparisons
- Comprehensive result aggregation and reporting
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from allooptim.allocation_to_allocators.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.a2a_result import A2AResult
from allooptim.allocation_to_allocators.orchestrator_factory import (
    OrchestratorType,
    create_orchestrator,
)
from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allooptim.backtest.backtest_config import BacktestConfig
from allooptim.backtest.data_loader import DataLoader
from allooptim.backtest.performance_metrics import PerformanceMetrics
from allooptim.covariance_transformer.transformer_list import get_transformer_by_names
from allooptim.optimizer.allocation_metric import MIN_OBSERVATIONS, LMoments, estimate_linear_moments
from allooptim.optimizer.wikipedia.wiki_database import download_data

logger = logging.getLogger(__name__)


# Constants for backtest engine
MIN_LOOKBACK_OBSERVATIONS = 5
TRAINING_WINDOW_DAYS = 730  # 2 years
MIN_VALID_ASSETS = 2
COVARIANCE_NAN_DIAGONAL_FILL = 0.01  # 1% variance for NaN diagonal elements
COVARIANCE_NAN_OFF_DIAGONAL_FILL = 0.0  # Zero correlation for NaN off-diagonal elements
MIN_WEIGHT_THRESHOLD = 0.0001  # Filter very small weights
INITIAL_PORTFOLIO_VALUE = 100000  # Start with $100k


class BacktestEngine:
    """Main backtesting engine with efficient optimizer execution."""

    def __init__(
        self,
        config_backtest: Optional[BacktestConfig] = None,
        a2a_config: Optional[A2AConfig] = None,
        orchestrator_type: Optional[OrchestratorType] = None,
        **orchestrator_kwargs,
    ) -> None:
        """Initialize the backtest engine.

        Args:
            config_backtest: Configuration for backtest parameters including date ranges,
                optimizers, and result storage settings.
            a2a_config: Configuration for allocation-to-allocators orchestration parameters.
            orchestrator_type: Type of orchestrator to use (equal_weight, optimized, etc.).
                If None, uses the type specified in config_backtest.
            **orchestrator_kwargs: Additional keyword arguments passed to orchestrator creation.
        """
        self.config_backtest = config_backtest or BacktestConfig()

        self.data_loader = DataLoader(
            benchmark=self.config_backtest.benchmark,
            symbols=self.config_backtest.symbols,
            interval=self.config_backtest.data_interval,
        )

        # Create orchestrator using factory
        if a2a_config is None:
            a2a_config = A2AConfig()

        # Use orchestrator_type parameter if provided, otherwise use config
        final_orchestrator_type = orchestrator_type or self.config_backtest.orchestration_type

        self.orchestrator = create_orchestrator(
            orchestrator_type=final_orchestrator_type,
            optimizer_configs=self.config_backtest.get_optimizer_configs_dict(),
            transformer_names=self.config_backtest.transformer_names,
            config=a2a_config,
            **orchestrator_kwargs,
        )

        self.results = {}

        self.transformers = get_transformer_by_names(self.config_backtest.transformer_names)

        # Create results directory (optional for notebook environments)
        if self.config_backtest.store_results:
            self.config_backtest.results_dir.mkdir(exist_ok=True, parents=True)

    def run_backtest(self) -> dict:
        """Run the comprehensive backtest.

        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("Starting comprehensive backtest")

        # Get date range based on debug mode
        start_data_date, end_data_date = self.config_backtest.get_data_date_range()

        # Check if Wikipedia optimizer is being used and download data if needed
        wikipedia_optimizer_names = ["WikipediaOptimizer"]
        has_wikipedia_optimizer = any(
            opt_name in wikipedia_optimizer_names for opt_name in self.config_backtest.optimizer_names
        )

        if has_wikipedia_optimizer:
            logger.info("Wikipedia optimizer detected, downloading Wikipedia data...")
            try:
                download_data(start_data_date, end_data_date, self.data_loader.stock_universe)
                logger.info("Wikipedia data download completed")
            except Exception as e:
                logger.warning(f"Failed to download Wikipedia data: {e}")

        # Load price data
        price_data = self.data_loader.load_price_data(start_data_date, end_data_date)

        # Generate rebalancing dates
        rebalance_dates = self._get_rebalance_dates(price_data.index)

        logger.info(f"Running backtest with {len(rebalance_dates)} rebalancing dates")

        # Initialize tracking
        allocation_results = []  # Store AllocationResult per timestep
        benchmark_weights_history = []  # SPY benchmark

        # Run backtest for each rebalancing date
        for i, rebalance_date in enumerate(rebalance_dates):
            logger.info(f"Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")

            # Get lookback window for estimation
            lookback_start = rebalance_date - timedelta(days=self.config_backtest.lookback_days)
            lookback_data = price_data[(price_data.index >= lookback_start) & (price_data.index <= rebalance_date)]

            if len(lookback_data) < MIN_LOOKBACK_OBSERVATIONS:  # Need minimum data (very reduced for 3-month backtest)
                logger.warning(f"Insufficient data for {rebalance_date}, skipping")
                continue

            # Clean data for estimation - remove assets with insufficient data
            min_observations = max(5, len(lookback_data) // 4)  # Need at least 25% of observations
            valid_assets = []

            for asset in lookback_data.columns:
                asset_data = lookback_data[asset].dropna()
                if len(asset_data) >= min_observations:
                    valid_assets.append(asset)

            if len(valid_assets) < MIN_VALID_ASSETS:
                logger.warning(f"Insufficient valid assets ({len(valid_assets)}) for {rebalance_date}, skipping")
                continue

            # Use only valid assets for estimation - ensure consistency across all data
            clean_data = lookback_data[valid_assets]

            # Single orchestrator call
            data_provider = _PriceDataProvider(clean_data)
            allocation_result = self.orchestrator.allocate(
                data_provider=data_provider,
                time_today=rebalance_date,
                all_stocks=self.data_loader.stock_universe,
            )
            allocation_results.append(allocation_result)

            # SPY benchmark
            benchmark_weights = pd.Series(0.0, index=clean_data.columns)
            if self.config_backtest.benchmark in benchmark_weights.index:
                benchmark_weights[self.config_backtest.benchmark] = 1.0
            benchmark_weights_history.append(benchmark_weights)

        # Calculate portfolio performance
        self.results = self._calculate_portfolio_performance(
            price_data, allocation_results, benchmark_weights_history, rebalance_dates
        )

        # Save A2A allocations to CSV
        self._save_a2a_allocations(allocation_results, rebalance_dates)

        logger.info("Backtest completed")
        return self.results

    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> list[datetime]:
        """Get rebalancing dates every N trading days."""
        rebalance_dates = []
        for i in range(self.config_backtest.lookback_days, len(date_index), self.config_backtest.rebalance_frequency):
            rebalance_dates.append(date_index[i])

        return rebalance_dates

    def _save_a2a_allocations(self, allocation_results: list[A2AResult], rebalance_dates: list[datetime]):
        """Save A2A final allocations to CSV file.

        Args:
            allocation_results: List of A2AResult objects
            rebalance_dates: list of rebalancing timestamps
        """
        if not allocation_results:
            logger.warning("No A2A results to save")
            return

        # Check if results directory exists (only save if it was created)
        if not self.config_backtest.results_dir.exists():
            logger.info("Results directory not created, skipping A2A allocation save")
            return

        try:
            # Create DataFrame with timestamps as index and asset symbols as columns
            final_allocations = []
            valid_indices = []

            for i, result in enumerate(allocation_results):
                if result.final_allocation is not None and not result.final_allocation.empty:
                    final_allocations.append(result.final_allocation)
                    valid_indices.append(rebalance_dates[i])
                else:
                    logger.warning(f"Skipping empty final allocation at index {i}")

            if not final_allocations:
                logger.warning("No valid A2A results to save")
                return

            a2a_allocations = pd.DataFrame(final_allocations, index=valid_indices)

            # Ensure index is datetime format with readable formatting
            a2a_allocations.index.name = "timestamp"

            # Save to CSV in the results directory
            csv_path = self.config_backtest.results_dir / "a2a_allocations.csv"
            a2a_allocations.to_csv(csv_path, float_format="%.6f")

            logger.info(f"✅ Saved A2A allocations to {csv_path}")
            logger.info(f"   Shape: {a2a_allocations.shape[0]} timestamps × {a2a_allocations.shape[1]} assets")

            # Also save a summary with non-zero positions only for readability
            summary_path = self.config_backtest.results_dir / "a2a_allocations_summary.csv"

            # For each timestamp, keep only non-zero weights
            summary_rows = []
            for timestamp in a2a_allocations.index:
                weights = a2a_allocations.loc[timestamp]
                non_zero_weights = weights[weights > MIN_WEIGHT_THRESHOLD]  # Filter very small weights

                if len(non_zero_weights) > 0:
                    summary_rows.append(
                        {
                            "timestamp": timestamp,
                            "num_positions": len(non_zero_weights),
                            "top_5_assets": ", ".join(non_zero_weights.nlargest(5).index.tolist()),
                            "top_5_weights": ", ".join([f"{w:.4f}" for w in non_zero_weights.nlargest(5).values]),
                            "total_weight": non_zero_weights.sum(),
                        }
                    )

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"✅ Saved A2A allocation summary to {summary_path}")

        except Exception as e:
            logger.error(f"Failed to save A2A allocations: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _calculate_portfolio_performance(
        self,
        price_data: pd.DataFrame,
        allocation_results: list[A2AResult],
        benchmark_weights_history: list[pd.Series],
        rebalance_dates: list[datetime],
    ) -> dict:
        """Calculate comprehensive performance metrics for all portfolios."""
        # Extract per-optimizer weights_history from optimizer_allocations
        optimizer_names = set()
        for result in allocation_results:
            for opt_alloc in result.optimizer_allocations:
                optimizer_names.add(opt_alloc.optimizer_name)

        # Build weights_history dict: {optimizer_name: list[pd.Series]}
        weights_history = {name: [] for name in optimizer_names}
        for result in allocation_results:
            for opt_alloc in result.optimizer_allocations:
                weights_history[opt_alloc.optimizer_name].append(opt_alloc.weights)

        # Calculate performance for each optimizer
        results = {}
        for optimizer_name in optimizer_names:
            if not weights_history[optimizer_name]:
                continue

            logger.info(f"Calculating performance for {optimizer_name}")

            # Reconstruct portfolio values
            portfolio_values = self._simulate_portfolio(price_data, weights_history[optimizer_name], rebalance_dates)

            if portfolio_values.empty:
                logger.warning(f"Empty portfolio values for {optimizer_name}")
                continue

            # Calculate returns
            returns = PerformanceMetrics.calculate_returns(portfolio_values)

            # Calculate all metrics
            metrics = {
                "sharpe_ratio": PerformanceMetrics.sharpe_ratio(returns),
                "max_drawdown": PerformanceMetrics.max_drawdown(portfolio_values),
                "time_underwater": PerformanceMetrics.time_underwater(portfolio_values),
                "cagr": PerformanceMetrics.cagr(portfolio_values),
                "risk_adjusted_return": PerformanceMetrics.risk_adjusted_return(returns),
                "annual_return": returns.mean() * 252,
                "annual_volatility": returns.std() * np.sqrt(252),
                "total_return": (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1,
            }

            # Returns statistics
            returns_stats = {
                "returns_mean": returns.mean() * 252,
                "returns_std": returns.std() * np.sqrt(252),
                "returns_min": returns.min() * 252,
                "returns_max": returns.max() * 252,
                "returns_q25": returns.quantile(0.25) * 252,
                "returns_q50": returns.quantile(0.50) * 252,
                "returns_q75": returns.quantile(0.75) * 252,
                "returns_skew": returns.skew(),
                "returns_kurtosis": returns.kurtosis(),
            }

            # Portfolio turnover statistics
            weights_df = pd.DataFrame(weights_history[optimizer_name], index=rebalance_dates)
            turnover = PerformanceMetrics.portfolio_turnover(weights_df)

            turnover_stats = {}
            if not turnover.empty:
                turnover_stats = {
                    "turnover_mean": turnover.mean(),
                    "turnover_std": turnover.std(),
                    "turnover_min": turnover.min(),
                    "turnover_max": turnover.max(),
                    "turnover_q25": turnover.quantile(0.25),
                    "turnover_q50": turnover.quantile(0.50),
                    "turnover_q75": turnover.quantile(0.75),
                }

            # Portfolio change rate statistics
            change_rate = PerformanceMetrics.portfolio_changerate(weights_df)

            change_rate_stats = {}
            if not change_rate.empty:
                change_rate_stats = {
                    "change_rate_mean": change_rate.mean(),
                    "change_rate_std": change_rate.std(),
                    "change_rate_min": change_rate.min(),
                    "change_rate_max": change_rate.max(),
                    "change_rate_q25": change_rate.quantile(0.25),
                    "change_rate_q50": change_rate.quantile(0.50),
                    "change_rate_q75": change_rate.quantile(0.75),
                }

            # Invested assets statistics
            invested_assets_stats = {}
            for rel_threshold in [5, 10, 50, 100]:
                invested_assets = PerformanceMetrics.portfolio_invested_assets(weights_df, rel_threshold / 100)

                if not invested_assets.empty:
                    invested_assets_stats.update(
                        {
                            f"invested_{rel_threshold}_abs_mean": invested_assets.mean(),
                            f"invested_{rel_threshold}_abs_std": invested_assets.std(),
                            f"invested_{rel_threshold}_abs_min": invested_assets.min(),
                            f"invested_{rel_threshold}_abs_max": invested_assets.max(),
                            f"invested_{rel_threshold}_pct_mean": (invested_assets / weights_df.shape[1]).mean(),
                            f"invested_{rel_threshold}_pct_std": (invested_assets / weights_df.shape[1]).std(),
                            f"invested_{rel_threshold}_pct_min": (invested_assets / weights_df.shape[1]).min(),
                            f"invested_{rel_threshold}_pct_max": (invested_assets / weights_df.shape[1]).max(),
                        }
                    )

            # Top N invested assets statistics
            invested_top_n_stats = {}
            for top_n in [5, 10, 50]:
                invested_assets = PerformanceMetrics.portfolio_invested_top_n(weights_df, top_n)

                if not invested_assets.empty:
                    invested_top_n_stats.update(
                        {
                            f"invested_top_{top_n}_mean": invested_assets.mean(),
                            f"invested_top_{top_n}_std": invested_assets.std(),
                            f"invested_top_{top_n}_min": invested_assets.min(),
                            f"invested_top_{top_n}_max": invested_assets.max(),
                        }
                    )

            # Computation and memory statistics
            # Extract computation times from A2AResult objects
            computation_times = []
            for result in allocation_results:
                if result.runtime_seconds > 0:
                    # Since runtime_seconds is for all optimizers combined, divide by number of optimizers
                    per_optimizer_time = result.runtime_seconds / len(result.optimizer_allocations)
                    computation_times.append(per_optimizer_time)
            
            if computation_times:
                computation_stats = {
                    "avg_computation_time": np.mean(computation_times),
                    "max_computation_time": np.max(computation_times),
                    "min_computation_time": np.min(computation_times),
                    "avg_memory_usage_mb": 0,  # Memory usage not tracked
                    "max_memory_usage_mb": 0,
                }
            else:
                computation_stats = {
                    "avg_computation_time": 0,
                    "max_computation_time": 0,
                    "min_computation_time": 0,
                    "avg_memory_usage_mb": 0,
                    "max_memory_usage_mb": 0,
                }

            # Combine all metrics
            all_metrics = {
                **metrics,
                **returns_stats,
                **invested_assets_stats,
                **invested_top_n_stats,
                **turnover_stats,
                **change_rate_stats,
                **computation_stats,
            }

            results[optimizer_name] = {
                "metrics": all_metrics,
                "portfolio_values": portfolio_values,
                "returns": returns,
                "weights_history": weights_df,
                "turnover": turnover,
            }

        # Add SPY benchmark results
        if benchmark_weights_history:
            logger.info("Calculating performance for SPY benchmark")

            spy_portfolio_values = self._simulate_portfolio(price_data, benchmark_weights_history, rebalance_dates)

            if not spy_portfolio_values.empty:
                spy_returns = PerformanceMetrics.calculate_returns(spy_portfolio_values)

                spy_metrics = {
                    "sharpe_ratio": PerformanceMetrics.sharpe_ratio(spy_returns),
                    "max_drawdown": PerformanceMetrics.max_drawdown(spy_portfolio_values),
                    "time_underwater": PerformanceMetrics.time_underwater(spy_portfolio_values),
                    "cagr": PerformanceMetrics.cagr(spy_portfolio_values),
                    "risk_adjusted_return": PerformanceMetrics.risk_adjusted_return(spy_returns),
                    "annual_return": spy_returns.mean() * 252,
                    "annual_volatility": spy_returns.std() * np.sqrt(252),
                    "total_return": (spy_portfolio_values.iloc[-1] / spy_portfolio_values.iloc[0]) - 1,
                }

                spy_returns_stats = {
                    "returns_mean": spy_returns.mean() * 252,
                    "returns_std": spy_returns.std() * np.sqrt(252),
                    "returns_min": spy_returns.min() * 252,
                    "returns_max": spy_returns.max() * 252,
                    "returns_q25": spy_returns.quantile(0.25) * 252,
                    "returns_q50": spy_returns.quantile(0.50) * 252,
                    "returns_q75": spy_returns.quantile(0.75) * 252,
                    "returns_skew": spy_returns.skew(),
                    "returns_kurtosis": spy_returns.kurtosis(),
                }

                spy_weights_df = pd.DataFrame(benchmark_weights_history, index=rebalance_dates)
                spy_turnover = PerformanceMetrics.portfolio_turnover(spy_weights_df)

                spy_turnover_stats = {}
                if not spy_turnover.empty:
                    spy_turnover_stats = {
                        "turnover_mean": spy_turnover.mean(),
                        "turnover_std": spy_turnover.std(),
                        "turnover_min": spy_turnover.min(),
                        "turnover_max": spy_turnover.max(),
                        "turnover_q25": spy_turnover.quantile(0.25),
                        "turnover_q50": spy_turnover.quantile(0.50),
                        "turnover_q75": spy_turnover.quantile(0.75),
                    }

                spy_change_rate = PerformanceMetrics.portfolio_changerate(spy_weights_df)

                spy_change_rate_stats = {}
                if not spy_change_rate.empty:
                    spy_change_rate_stats = {
                        "change_rate_mean": spy_change_rate.mean(),
                        "change_rate_std": spy_change_rate.std(),
                        "change_rate_min": spy_change_rate.min(),
                        "change_rate_max": spy_change_rate.max(),
                        "change_rate_q25": spy_change_rate.quantile(0.25),
                        "change_rate_q50": spy_change_rate.quantile(0.50),
                        "change_rate_q75": spy_change_rate.quantile(0.75),
                    }

                spy_all_metrics = {
                    **spy_metrics,
                    **spy_returns_stats,
                    **spy_turnover_stats,
                    **spy_change_rate_stats,
                }

                results[self.config_backtest.benchmark] = {
                    "metrics": spy_all_metrics,
                    "portfolio_values": spy_portfolio_values,
                    "returns": spy_returns,
                    "weights_history": spy_weights_df,
                    "turnover": spy_turnover,
                }

        # Calculate A2A Ensemble using final_allocation from each A2AResult
        if allocation_results:
            logger.info("Calculating A2A ensemble performance")

            # Use final_allocation from each A2AResult, filtering out empty ones
            a2a_weights_history = []
            valid_rebalance_dates = []

            for i, result in enumerate(allocation_results):
                if result.final_allocation is not None and not result.final_allocation.empty:
                    a2a_weights_history.append(result.final_allocation)
                    valid_rebalance_dates.append(rebalance_dates[i])

            if a2a_weights_history:
                # Simulate portfolio with A2A final allocations
                a2a_portfolio_values = self._simulate_portfolio(price_data, a2a_weights_history, valid_rebalance_dates)

                if not a2a_portfolio_values.empty:
                    a2a_returns = PerformanceMetrics.calculate_returns(a2a_portfolio_values)

                    a2a_metrics = {
                        "sharpe_ratio": PerformanceMetrics.sharpe_ratio(a2a_returns),
                        "max_drawdown": PerformanceMetrics.max_drawdown(a2a_portfolio_values),
                        "time_underwater": PerformanceMetrics.time_underwater(a2a_portfolio_values),
                        "cagr": PerformanceMetrics.cagr(a2a_portfolio_values),
                        "risk_adjusted_return": PerformanceMetrics.risk_adjusted_return(a2a_returns),
                        "annual_return": a2a_returns.mean() * 252,
                        "annual_volatility": a2a_returns.std() * np.sqrt(252),
                        "total_return": (a2a_portfolio_values.iloc[-1] / a2a_portfolio_values.iloc[0]) - 1,
                    }

                    a2a_returns_stats = {
                        "returns_mean": a2a_returns.mean() * 252,
                        "returns_std": a2a_returns.std() * np.sqrt(252),
                        "returns_min": a2a_returns.min() * 252,
                        "returns_max": a2a_returns.max() * 252,
                        "returns_q25": a2a_returns.quantile(0.25) * 252,
                        "returns_q50": a2a_returns.quantile(0.50) * 252,
                        "returns_q75": a2a_returns.quantile(0.75) * 252,
                        "returns_skew": a2a_returns.skew(),
                        "returns_kurtosis": a2a_returns.kurtosis(),
                    }

                    a2a_weights_df = pd.DataFrame(a2a_weights_history, index=valid_rebalance_dates)
                    a2a_turnover = PerformanceMetrics.portfolio_turnover(a2a_weights_df)

                    a2a_turnover_stats = {}
                    if not a2a_turnover.empty:
                        a2a_turnover_stats = {
                            "turnover_mean": a2a_turnover.mean(),
                            "turnover_std": a2a_turnover.std(),
                            "turnover_min": a2a_turnover.min(),
                            "turnover_max": a2a_turnover.max(),
                            "turnover_q25": a2a_turnover.quantile(0.25),
                            "turnover_q50": a2a_turnover.quantile(0.50),
                            "turnover_q75": a2a_turnover.quantile(0.75),
                        }

                    a2a_change_rate = PerformanceMetrics.portfolio_changerate(a2a_weights_df)

                    a2a_change_rate_stats = {}
                    if not a2a_change_rate.empty:
                        a2a_change_rate_stats = {
                            "change_rate_mean": a2a_change_rate.mean(),
                            "change_rate_std": a2a_change_rate.std(),
                            "change_rate_min": a2a_change_rate.min(),
                            "change_rate_max": a2a_change_rate.max(),
                            "change_rate_q25": a2a_change_rate.quantile(0.25),
                            "change_rate_q50": a2a_change_rate.quantile(0.50),
                            "change_rate_q75": a2a_change_rate.quantile(0.75),
                        }

                    # Invested assets statistics for A2A
                    a2a_invested_assets_stats = {}
                    for rel_threshold in [5, 10, 50, 100]:
                        invested_assets = PerformanceMetrics.portfolio_invested_assets(
                            a2a_weights_df, rel_threshold / 100
                        )

                        if not invested_assets.empty:
                            a2a_invested_assets_stats.update(
                                {
                                    f"invested_{rel_threshold}_abs_mean": invested_assets.mean(),
                                    f"invested_{rel_threshold}_abs_std": invested_assets.std(),
                                    f"invested_{rel_threshold}_abs_min": invested_assets.min(),
                                    f"invested_{rel_threshold}_abs_max": invested_assets.max(),
                                    f"invested_{rel_threshold}_pct_mean": (
                                        invested_assets / a2a_weights_df.shape[1]
                                    ).mean(),
                                    f"invested_{rel_threshold}_pct_std": (
                                        invested_assets / a2a_weights_df.shape[1]
                                    ).std(),
                                    f"invested_{rel_threshold}_pct_min": (
                                        invested_assets / a2a_weights_df.shape[1]
                                    ).min(),
                                    f"invested_{rel_threshold}_pct_max": (
                                        invested_assets / a2a_weights_df.shape[1]
                                    ).max(),
                                }
                            )

                    # Top N invested assets statistics for A2A
                    a2a_invested_top_n_stats = {}
                    for top_n in [5, 10, 50]:
                        invested_assets = PerformanceMetrics.portfolio_invested_top_n(a2a_weights_df, top_n)

                        if not invested_assets.empty:
                            a2a_invested_top_n_stats.update(
                                {
                                    f"invested_top_{top_n}_mean": invested_assets.mean(),
                                    f"invested_top_{top_n}_std": invested_assets.std(),
                                    f"invested_top_{top_n}_min": invested_assets.min(),
                                    f"invested_top_{top_n}_max": invested_assets.max(),
                                }
                            )

                    a2a_all_metrics = {
                        **a2a_metrics,
                        **a2a_returns_stats,
                        **a2a_invested_assets_stats,
                        **a2a_invested_top_n_stats,
                        **a2a_turnover_stats,
                        **a2a_change_rate_stats,
                    }

                    results["A2AEnsemble"] = {
                        "metrics": a2a_all_metrics,
                        "portfolio_values": a2a_portfolio_values,
                        "returns": a2a_returns,
                        "weights_history": a2a_weights_df,
                        "turnover": a2a_turnover,
                    }

        return results

    def _simulate_portfolio(
        self, price_data: pd.DataFrame, weights_history: list[pd.Series], rebalance_dates: list[datetime]
    ) -> pd.Series:
        """Simulate portfolio performance with perfect execution."""
        if not weights_history or not rebalance_dates:
            return pd.Series([], dtype=float)

        # Initialize portfolio value
        portfolio_values = []
        current_weights = None
        portfolio_value = INITIAL_PORTFOLIO_VALUE  # Start with $100k

        for date in price_data.index:
            # Check if we need to rebalance
            if date in rebalance_dates:
                rebalance_idx = rebalance_dates.index(date)
                if rebalance_idx < len(weights_history):
                    current_weights = weights_history[rebalance_idx]

            if current_weights is not None:
                # Calculate portfolio value
                available_assets = current_weights.index.intersection(price_data.columns)

                if len(available_assets) > 0:
                    asset_prices = price_data.loc[date, available_assets]
                    asset_weights = current_weights[available_assets]

                    if len(portfolio_values) == 0:
                        # First day, use starting weights
                        portfolio_values.append(portfolio_value)
                    else:
                        # Calculate return
                        prev_prices = price_data.loc[
                            price_data.index[price_data.index.get_loc(date) - 1], available_assets
                        ]
                        returns = (asset_prices / prev_prices - 1).fillna(0)
                        portfolio_return = (asset_weights * returns).sum()
                        portfolio_value *= 1 + portfolio_return
                        portfolio_values.append(portfolio_value)
                else:
                    portfolio_values.append(portfolio_value)
            else:
                portfolio_values.append(portfolio_value)

        return pd.Series(portfolio_values, index=price_data.index)


class _PriceDataProvider(AbstractObservationSimulator):
    """Simple data provider that wraps price DataFrame for orchestrator compatibility."""

    def __init__(self, price_data: pd.DataFrame):
        """Initialize the price data provider.

        Args:
            price_data: Historical price data with datetime index and asset columns.
        """
        self.price_data = price_data
        # Calculate basic statistics (simplified) - return as pandas objects
        returns = price_data.pct_change().dropna()
        self._mu = returns.mean()
        self._cov = returns.cov()

        # Compute L-moments from returns data
        if len(returns) >= MIN_OBSERVATIONS:
            self._l_moments = estimate_linear_moments(returns.values)
        else:
            # Fallback: create zero L-moments with correct structure
            n_assets = len(price_data.columns)
            self._l_moments = LMoments(
                lt_comoment_1=np.zeros((n_assets, n_assets)),
                lt_comoment_2=np.zeros((n_assets, n_assets)),
                lt_comoment_3=np.zeros((n_assets, n_assets)),
                lt_comoment_4=np.zeros((n_assets, n_assets)),
            )

    @property
    def mu(self):
        """Get expected returns as pandas Series."""
        return self._mu

    @property
    def cov(self):
        """Get covariance matrix as pandas DataFrame."""
        return self._cov

    @property
    def historical_prices(self):
        """Get historical price data as pandas DataFrame."""
        return self.price_data

    @property
    def n_observations(self):
        """Get number of observations in the price data."""
        return len(self.price_data)

    def get_sample(self):
        """Get sample market parameters (same as ground truth for backtest)."""
        return self.get_ground_truth()

    def get_ground_truth(self):
        """Get ground truth market parameters from historical data."""
        time_end = self.price_data.index[-1]  # Last timestamp in the data
        return self._mu, self._cov, self.price_data, time_end, self._l_moments

    @property
    def name(self) -> str:
        """Get the data provider name identifier."""
        return "BacktestDataProvider"
