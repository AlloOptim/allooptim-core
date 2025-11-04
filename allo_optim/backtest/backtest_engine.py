import logging
import tracemalloc
from datetime import datetime, timedelta
from time import time

import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from allo_optim.allocation_to_allocators.orchestrator_adapter import (
    AllocationOrchestratorAdapter,
)
from allo_optim.backtest.backtest_config import BacktestConfig, config
from allo_optim.backtest.data_loader import DataLoader
from allo_optim.backtest.performance_metrics import PerformanceMetrics
from allo_optim.covariance_transformer.transformer_list import get_transformer_by_names
from allo_optim.optimizer.allocation_metric import estimate_linear_moments
from allo_optim.optimizer.ensemble_optimizers import A2AEnsembleOptimizer, SPY500Benchmark
from allo_optim.optimizer.optimizer_interface import (
    AbstractEnsembleOptimizer,
    AbstractOptimizer,
)
from allo_optim.optimizer.optimizer_list import get_optimizer_by_names
from allo_optim.optimizer.wikipedia.sql_database import download_data

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

    def __init__(self, config_instance: BacktestConfig = None):
        if config_instance is None:
            config_instance = config
        self.config = config_instance

        self.data_loader = DataLoader()
        self.individual_optimizers, self.ensemble_optimizers = self._setup_optimizers()
        # Combine for backward compatibility where needed
        self.optimizers = self.individual_optimizers + self.ensemble_optimizers
        self.results = {}

        self.transformers = get_transformer_by_names(self.config.transformer_names)

        # Create results directory
        self.config.results_dir.mkdir(exist_ok=True, parents=True)

    def _setup_optimizers(self) -> tuple[list[AbstractOptimizer], list[AbstractEnsembleOptimizer]]:
        """Setup individual and ensemble optimizers separately for efficient execution."""

        # Check if using orchestrator mode
        if self.config.use_orchestrator:
            # Use AllocationOrchestratorAdapter
            orchestrator = AllocationOrchestratorAdapter(
                optimizer_names=self.config.optimizer_names,
                transformer_names=self.config.transformer_names,
                orchestration_type=self.config.orchestration_type,
            )

            individual_optimizers = [orchestrator]
            ensemble_optimizers = []  # Don't double-count with A2AEnsembleOptimizer

            logger.info(
                f"Using AllocationOrchestrator in {self.config.orchestration_type.value} mode"
            )
            logger.info(f"Orchestrator wraps: {self.config.optimizer_names}")
        else:
            # Traditional mode: Individual optimizers
            optimizer_names = self.config.optimizer_names
            individual_optimizers = get_optimizer_by_names(optimizer_names)

            # Ensemble optimizers (run after individual optimizers to use df_allocations)
            ensemble_optimizers = [A2AEnsembleOptimizer(), SPY500Benchmark()]

            logger.info(
                f"Setup {len(individual_optimizers)} individual optimizers: {[opt.name for opt in individual_optimizers]}"
            )
            logger.info(
                f"Setup {len(ensemble_optimizers)} ensemble optimizers: {[opt.name for opt in ensemble_optimizers]}"
            )

        return individual_optimizers, ensemble_optimizers

    def run_backtest(self) -> dict:
        """
        Run the comprehensive backtest.

        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("Starting comprehensive backtest")

        # Get date range based on debug mode
        start_data_date, end_data_date = self.config.get_data_date_range()

        # Check if Wikipedia optimizer is being used and download data if needed
        wikipedia_optimizer_names = ["WikipediaOptimizer"]
        has_wikipedia_optimizer = any(opt.name in wikipedia_optimizer_names for opt in self.individual_optimizers)

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
        {opt.name: [] for opt in self.optimizers}
        weights_history = {opt.name: [] for opt in self.optimizers}
        computation_times = {opt.name: [] for opt in self.optimizers}
        memory_usage = {opt.name: [] for opt in self.optimizers}

        # Run backtest for each rebalancing date
        for i, rebalance_date in enumerate(rebalance_dates):
            logger.info(f"Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")

            # Get lookback window for estimation
            lookback_start = rebalance_date - timedelta(days=self.config.lookback_days)
            lookback_data = price_data[(price_data.index >= lookback_start) & (price_data.index <= rebalance_date)]

            if len(lookback_data) < MIN_LOOKBACK_OBSERVATIONS:  # Need minimum data (very reduced for 3-month backtest)
                logger.warning(f"Insufficient data for {rebalance_date}, skipping")
                continue

            # Get 2-year training window for optimizer.fit()
            training_start = rebalance_date - timedelta(days=TRAINING_WINDOW_DAYS)  # 2 years
            training_data = price_data[(price_data.index >= training_start) & (price_data.index <= rebalance_date)]

            # If not enough training data, use all available data
            if len(training_data) < self.config.lookback_days:
                training_data = lookback_data

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

            # Also filter training data to match valid assets
            clean_training_data = (
                training_data[valid_assets] if len(training_data.columns) > len(valid_assets) else training_data
            )

            # Follow the correct call sequence as specified by user:
            # 1. mu = estimate_returns(prices_of_lookback_period)
            mu = mean_historical_return(clean_data, log_returns=self.config.log_returns)

            # 2. cov = estimate_covariance(prices_of_lookback_period)
            cov = sample_cov(clean_data, log_returns=self.config.log_returns)

            # Handle any remaining NaN values in covariance matrix
            if cov.isna().any().any():
                logger.warning(
                    f"NaN values detected in covariance matrix for {rebalance_date}, filling with robust estimates"
                )
                # Fill NaN with small diagonal values and zero off-diagonal
                for row_idx in range(len(cov)):
                    for col_idx in range(len(cov.columns)):
                        if pd.isna(cov.iloc[row_idx, col_idx]):
                            if row_idx == col_idx:  # Diagonal - use small positive variance
                                cov.iloc[row_idx, col_idx] = COVARIANCE_NAN_DIAGONAL_FILL  # 1% variance
                            else:  # Off-diagonal - use zero correlation
                                cov.iloc[row_idx, col_idx] = COVARIANCE_NAN_OFF_DIAGONAL_FILL
            # 3. cov = Transformer(cov)
            for transformer in self.transformers:
                cov = transformer.transform(cov, n_observations=len(clean_data))

            # 5. l_moments = estimate_l_moments(df_prices_lookback_period)
            l_moments = estimate_linear_moments(clean_data)

            # Step 1: Run individual optimizers first and collect allocations
            df_allocations_data = {}

            for optimizer in self.individual_optimizers:
                logger.info(f"Running optimizer: {optimizer.name}")

                # Track memory and time
                tracemalloc.start()
                start_time = time()

                try:
                    # 4. optimizer.fit(df_train_prices) - use 2-year training data with valid assets only
                    optimizer.fit(clean_training_data)

                    # 6. allocation_weights = optimizer.allocate(mu, cov, time, l_moments)
                    weights = optimizer.allocate(
                        ds_mu=mu,
                        df_cov=cov,
                        df_prices=clean_data,
                        time=rebalance_date,
                        l_moments=l_moments,
                    )

                    # Handle failed allocation
                    if weights is None or weights.sum() == 0:
                        if self.config.use_equal_weights_fallback:
                            weights = pd.Series(np.ones(len(mu)) / len(mu), index=mu.index)
                        else:
                            weights = pd.Series(np.zeros(len(mu)), index=mu.index)

                    # Normalize weights
                    if weights.sum() > 0:
                        weights = weights / weights.sum()

                    # Store in both tracking and allocations DataFrame
                    weights_history[optimizer.name].append(weights)
                    df_allocations_data[optimizer.name] = weights

                    # Track performance metrics
                    end_time = time()
                    computation_times[optimizer.name].append(end_time - start_time)

                    current, peak = tracemalloc.get_traced_memory()
                    memory_usage[optimizer.name].append(current / 1024 / 1024)  # MB
                    tracemalloc.stop()

                except Exception as e:
                    optimizer.reset()

                    logger.error(f"Error in {optimizer.name} at {rebalance_date}: {e}")
                    if self.config.rerun_allocator_exceptions:
                        raise e

                    # Use fallback weights
                    if self.config.use_equal_weights_fallback:
                        fallback_weights = pd.Series(np.ones(len(mu)) / len(mu), index=mu.index)
                    else:
                        fallback_weights = pd.Series(np.zeros(len(mu)), index=mu.index)

                    # Store fallback in both tracking and allocations DataFrame
                    weights_history[optimizer.name].append(fallback_weights)
                    df_allocations_data[optimizer.name] = fallback_weights

                    # Record time and memory for failed optimizers too
                    end_time = time()
                    computation_times[optimizer.name].append(end_time - start_time)

                    current, peak = tracemalloc.get_traced_memory()
                    memory_usage[optimizer.name].append(peak / 1024 / 1024)  # Convert to MB
                    tracemalloc.stop()

            # Step 2: Create df_allocations DataFrame from individual optimizer results
            if df_allocations_data:
                df_allocations = pd.DataFrame(df_allocations_data).T  # Transpose so optimizers are rows
                logger.info(
                    f"Created allocation matrix with {len(df_allocations)} optimizers and "
                    f"{len(df_allocations.columns)} assets"
                )
            else:
                # No individual optimizers succeeded
                df_allocations = None
                logger.warning("No individual optimizers succeeded")

            # Step 3: Run ensemble optimizers with df_allocations
            for optimizer in self.ensemble_optimizers:
                # Track memory and time
                tracemalloc.start()
                start_time = time()

                try:
                    # Ensemble optimizers don't need fitting typically
                    optimizer.fit(clean_training_data)

                    # Pass df_allocations to ensemble optimizers for efficient computation
                    weights = optimizer.allocate(
                        ds_mu=mu,
                        df_cov=cov,
                        df_prices=clean_data,
                        df_allocations=df_allocations,
                        time=rebalance_date,
                        l_moments=l_moments,
                    )

                    # Handle failed allocation
                    if weights is None or weights.sum() == 0:
                        if self.config.use_equal_weights_fallback:
                            weights = pd.Series(np.ones(len(mu)) / len(mu), index=mu.index)
                        else:
                            weights = pd.Series(np.zeros(len(mu)), index=mu.index)

                    # Normalize weights
                    if weights.sum() > 0:
                        weights = weights / weights.sum()

                    weights_history[optimizer.name].append(weights)

                except Exception as e:
                    logger.error(f"Error in ensemble optimizer {optimizer.name} at {rebalance_date}: {e}")

                    # Use fallback weights for ensemble optimizers too
                    if self.config.use_equal_weights_fallback:
                        fallback_weights = pd.Series(np.ones(len(mu)) / len(mu), index=mu.index)
                    else:
                        fallback_weights = pd.Series(np.zeros(len(mu)), index=mu.index)

                    weights_history[optimizer.name].append(fallback_weights)

                # Record time and memory for ensemble optimizers
                end_time = time()
                computation_times[optimizer.name].append(end_time - start_time)

                current, peak = tracemalloc.get_traced_memory()
                memory_usage[optimizer.name].append(peak / 1024 / 1024)  # Convert to MB
                tracemalloc.stop()

        # Calculate portfolio performance
        self.results = self._calculate_portfolio_performance(
            price_data, weights_history, rebalance_dates, computation_times, memory_usage
        )

        # Save A2A allocations to CSV
        self._save_a2a_allocations(weights_history, rebalance_dates)

        logger.info("Backtest completed")
        return self.results

    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> list[datetime]:
        """Get rebalancing dates every N trading days."""

        rebalance_dates = []
        for i in range(self.config.lookback_days, len(date_index), self.config.rebalance_frequency):
            rebalance_dates.append(date_index[i])

        return rebalance_dates

    def _save_a2a_allocations(self, weights_history: dict[str, list[pd.Series]], rebalance_dates: list[datetime]):
        """
        Save A2A ensemble allocations to CSV file.

        Args:
            weights_history: Dictionary of optimizer name -> list of weight series
            rebalance_dates: list of rebalancing timestamps
        """
        # Check if A2A_Ensemble exists in weights_history
        a2a_key = None
        for key in weights_history:
            if "A2A" in key or "a2a" in key.lower():
                a2a_key = key
                break

        if a2a_key is None or not weights_history[a2a_key]:
            logger.warning("A2A allocations not found in weights_history, skipping CSV export")
            return

        try:
            # Create DataFrame with timestamps as index and asset symbols as columns
            a2a_allocations = pd.DataFrame(weights_history[a2a_key], index=rebalance_dates)

            # Ensure index is datetime format with readable formatting
            a2a_allocations.index.name = "timestamp"

            # Save to CSV in the results directory
            csv_path = self.config.results_dir / "a2a_allocations.csv"
            a2a_allocations.to_csv(csv_path, float_format="%.6f")

            logger.info(f"✅ Saved A2A allocations to {csv_path}")
            logger.info(f"   Shape: {a2a_allocations.shape[0]} timestamps × {a2a_allocations.shape[1]} assets")

            # Also save a summary with non-zero positions only for readability
            summary_path = self.config.results_dir / "a2a_allocations_summary.csv"

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
        weights_history: dict[str, list[pd.Series]],
        rebalance_dates: list[datetime],
        computation_times: dict[str, list[float]],
        memory_usage: dict[str, list[float]],
    ) -> dict:
        """Calculate comprehensive performance metrics for all portfolios."""

        results = {}

        for optimizer_name in weights_history:
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

            # Portfolio turnover statistics
            weights_df = pd.DataFrame(weights_history[optimizer_name], index=rebalance_dates)
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

            weights_df = pd.DataFrame(weights_history[optimizer_name], index=rebalance_dates)
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

            weights_df = pd.DataFrame(weights_history[optimizer_name], index=rebalance_dates)
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
            computation_stats = {
                "avg_computation_time": np.mean(computation_times[optimizer_name])
                if computation_times[optimizer_name]
                else 0,
                "max_computation_time": np.max(computation_times[optimizer_name])
                if computation_times[optimizer_name]
                else 0,
                "min_computation_time": np.min(computation_times[optimizer_name])
                if computation_times[optimizer_name]
                else 0,
                "avg_memory_usage_mb": np.mean(memory_usage[optimizer_name]) if memory_usage[optimizer_name] else 0,
                "max_memory_usage_mb": np.max(memory_usage[optimizer_name]) if memory_usage[optimizer_name] else 0,
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
