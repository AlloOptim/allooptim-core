"""
Enhanced Allocation Workflow

Clean orchestration of the complete allocation-to-allocators process.
Integrates MCOS simulation with allocation optimization.
"""

import logging
from datetime import datetime
from timeit import default_timer as timer
from typing import List, Optional

import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from allo_optim.allocation_to_allocators.allocation_optimizer import (
    optimize_allocator_weights,
)
from allo_optim.allocation_to_allocators.optimizer_simulator import (
    simulate_optimizers_with_allocation_statistics,
)
from allo_optim.config.allocation_dataclasses import (
    A2AStatistics,
    AllocationResult,
    NoStatistics,
)
from allo_optim.config.stock_dataclasses import StockUniverse
from allo_optim.covariance_transformer.covariance_transformer import (
    OracleCovarianceTransformer,
)
from allo_optim.optimizer.allocation_metric import (
    estimate_linear_moments,
    validate_no_nan,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer
from allo_optim.optimizer.optimizer_list import (
    get_all_optimizers,
    get_optimizer_by_names,
)
from allo_optim.optimizer.wikipedia.allocate_wikipedia import allocate_wikipedia

logger = logging.getLogger(__name__)


class AllocationOrchestrator:
    def __init__(self):
        self._optimizer_names: Optional[list[str]] = None
        self._optimizer_list: Optional[list[AbstractOptimizer]] = get_all_optimizers()

    def _check_weights_bounds(self, weights: np.ndarray, name: str) -> np.ndarray:
        weights_tolterance = 0.01

        if np.sum(weights) < -weights_tolterance or np.sum(weights) > 1.0 + weights_tolterance:
            raise ValueError(f"{name} must sum to 1.0, got {np.sum(weights)}")

        if any(weights < -weights_tolterance) or any(weights > 1.0 + weights_tolterance):
            raise ValueError(f"{name} contains negative or leveraged values")

        if any(weights < 0.0) or any(weights > 1.0):
            logger.warning(f"{name} contains invalid weights, clipping to [0.0, 1.0]")
            weights = np.clip(weights, 0.0, 1.0)

        if np.sum(weights) < 0.0:
            logger.warning(f"{name} sums to {np.sum(weights)} < 0.0, adjusting to 0.0")
            weights = np.zeros_like(weights)

        if np.sum(weights) > 1.0 + weights_tolterance or np.sum(weights) < 1.0 - weights_tolterance:
            logger.warning(f"{name} sums to {np.sum(weights)}, normalizing to 1.0")
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                # If all weights are zero, set equal weights
                weights = np.ones_like(weights) / len(weights)

        return weights

    def _compute_final_asset_weights(
        self,
        df_prices: pd.DataFrame,
        allocator_weights: np.ndarray,
    ) -> tuple:
        """
        Compute final asset weights and performance statistics.

        Args:
            df_assets: Asset price DataFrame
            allocator_weights: Optimal weights for each allocator

        Returns:
            Tuple of (asset_weights_dict, statistics_dict)
        """

        # Input validation
        if self._optimizer_list is None:
            raise ValueError("optimizer_list is not initialized")

        if len(self._optimizer_list) != len(allocator_weights):
            raise ValueError(f"Optimizer count {len(self._optimizer_list)} != weight count {len(allocator_weights)}")

        validate_no_nan(allocator_weights, "allocator weights")

        time = df_prices.index.max()

        # Compute asset statistics
        l_moments = estimate_linear_moments(df_prices)

        # Keep as pandas Series/DataFrame for new optimizer interface
        mu = mean_historical_return(df_prices)  # Returns pandas Series
        cov = sample_cov(df_prices)  # Returns pandas DataFrame

        transformer = OracleCovarianceTransformer()
        cov = transformer.transform(cov, n_observations=len(df_prices))  # Returns pandas DataFrame

        validate_no_nan(mu.values, "historical returns")
        validate_no_nan(cov.values, "sample covariance")

        # Initialize tracking
        asset_weights = np.zeros(len(mu))
        statistic_returns = {}
        statistic_volatilities = {}
        statistic_runtime = {"A2A": np.nan}

        # Compute weighted asset allocation
        for k, optimizer in enumerate(self._optimizer_list):
            start = timer()

            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(df_prices)

                weights = optimizer.allocate(mu, cov, df_prices, None, time, l_moments)  # Returns pandas Series
                validate_no_nan(weights.values, f"{optimizer.name} weights")

                # Validate allocation
                weights_array = self._check_weights_bounds(weights.values, f"{optimizer.name} weights")
                # Convert back to pandas Series to maintain asset names
                weights = pd.Series(weights_array, index=weights.index)

            except Exception as error:
                optimizer.reset()

                raise RuntimeError(f"Allocation failed for {optimizer.name}: {str(error)}")

            end = timer()

            # Compute optimizer statistics using pandas operations
            expected_return = (weights * mu).sum()  # Element-wise multiplication and sum
            expected_volatility = np.sqrt((weights * (cov @ weights)).sum())  # Matrix multiplication with pandas

            validate_no_nan(np.array([expected_return]), f"{optimizer.name} expected return")
            validate_no_nan(np.array([expected_volatility]), f"{optimizer.name} expected volatility")

            # Accumulate weighted asset allocation
            asset_weights += allocator_weights[k] * weights.values

            # Store statistics
            statistic_returns[optimizer.name] = expected_return
            statistic_volatilities[optimizer.name] = expected_volatility
            statistic_runtime[optimizer.name] = end - start

        # Final validation of asset weights
        validate_no_nan(asset_weights, "final asset weights")

        asset_weights = self._check_weights_bounds(asset_weights, "final asset weights")

        # Compute A2A portfolio statistics
        expected_return_a2a = np.dot(asset_weights, mu)
        expected_volatility_a2a = np.sqrt(np.dot(asset_weights, np.dot(cov, asset_weights)))

        validate_no_nan(np.array([expected_return_a2a]), "A2A expected return")
        validate_no_nan(np.array([expected_volatility_a2a]), "A2A expected volatility")

        statistic_returns["A2A"] = expected_return_a2a
        statistic_volatilities["A2A"] = expected_volatility_a2a

        # Build result dictionaries
        statistic_algo_weights = {
            optimizer.name: weight for optimizer, weight in zip(self._optimizer_list, allocator_weights)
        }

        asset_weights_dict = {col: weight for col, weight in zip(df_prices.columns, asset_weights)}

        statistics_dict = {
            "returns": statistic_returns,
            "volatilities": statistic_volatilities,
            "runtime": statistic_runtime,
            "algo_weights": statistic_algo_weights,
            "asset_weights": asset_weights_dict,
        }

        return asset_weights_dict, statistics_dict

    def run_optimized_allocation_to_allocators(
        self,
        df_prices: pd.DataFrame,
        n_data_observations: int = 50,
        n_particle_swarm_iterations: int = 1000,
        n_particles: int = 1000,
    ) -> AllocationResult:
        """
        Enhanced allocation-to-allocators workflow.

        Always uses:
        - Enhanced MCOS with allocation statistics
        - Full allocation covariance matrices
        - Particle swarm optimization
        - Analytical expected return estimation

        Args:
            df_assets: Asset price DataFrame
            n_data_observations: Number of MCOS simulation iterations
            n_options_simulations: Number of PSO optimization iterations
            n_particles: Number of PSO particles

        Returns:
            A2AResult with optimized asset weights and statistics

        Raises:
            ValueError: If inputs contain NaN or validation fails
        """

        # Input validation
        if df_prices.isna().any().any():
            raise ValueError("df_assets contains NaN values")

        if self._optimizer_list is None or len(self._optimizer_list) == 0:
            raise ValueError("optimizer_list cannot be empty")

        if n_data_observations < 1:
            raise ValueError(f"n_data_observations must be >= 1, got {n_data_observations}")

        if n_particle_swarm_iterations < 1:
            raise ValueError(f"n_options_simulations must be >= 1, got {n_particle_swarm_iterations}")

        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")

        try:
            # Step 1: Run enhanced MCOS simulation
            mcos_result = simulate_optimizers_with_allocation_statistics(
                df_assets=df_prices,
                optimizer_list=self._optimizer_list,
                n_data_observations=n_data_observations,
            )

            # Validate MCOS result
            if mcos_result.expected_return_means is None:
                raise ValueError("MCOS simulation failed to generate expected return means")

            if mcos_result.expected_returns_covariance is None:
                raise ValueError("MCOS simulation failed to generate expected returns covariance")

            if len(mcos_result.expected_return_means) != len(self._optimizer_list):
                raise ValueError(
                    f"MCOS generated {len(mcos_result.expected_return_means)} optimizer means "
                    f"but expected {len(self._optimizer_list)}"
                )

            # Step 2: Optimize allocator weights
            allocator_weights = optimize_allocator_weights(
                mcos_result=mcos_result,
                n_particle_swarm_iterations=n_particle_swarm_iterations,
                n_particles=n_particles,
            )

            # Step 3: Compute final asset weights and statistics
            asset_weights_dict, statistics_dict = self._compute_final_asset_weights(
                df_prices=df_prices,
                allocator_weights=allocator_weights,
            )

            # Build result
            statistics = A2AStatistics(
                asset_returns=statistics_dict["returns"],
                asset_volatilities=statistics_dict["volatilities"],
                algo_runtime=statistics_dict["runtime"],
                algo_weights=statistics_dict["algo_weights"],
            )

            result = AllocationResult(asset_weights=asset_weights_dict, success=True, statistics=statistics)

            return result

        except Exception as e:
            raise RuntimeError(f"Enhanced allocation workflow failed: {str(e)}") from e

    def run_equal_allocation_to_allocators(
        self,
        df_prices: pd.DataFrame,
        allocator_weights: Optional[np.ndarray] = None,
    ) -> AllocationResult:
        try:
            if df_prices.isna().any().any():
                raise ValueError("df_assets contains NaN values")

            if self._optimizer_list is None or len(self._optimizer_list) == 0:
                raise ValueError("optimizer_list cannot be empty")

            if allocator_weights is None:
                allocator_weights = 1.0 / len(self._optimizer_list) * np.ones(len(self._optimizer_list))

            if len(self._optimizer_list) != len(allocator_weights):
                raise ValueError(
                    f"Optimizer count {len(self._optimizer_list)} != weight count {len(allocator_weights)}"
                )

            if not np.isclose(np.sum(allocator_weights), 1.0):
                raise ValueError(f"Allocator weights must sum to 1.0, got {np.sum(allocator_weights)}")

            asset_weights_dict, statistics_dict = self._compute_final_asset_weights(
                df_prices=df_prices,
                allocator_weights=allocator_weights,
            )

            statistics = A2AStatistics(
                asset_returns=statistics_dict["returns"],
                asset_volatilities=statistics_dict["volatilities"],
                algo_runtime=statistics_dict["runtime"],
                algo_weights=statistics_dict["algo_weights"],
            )

            result = AllocationResult(asset_weights=asset_weights_dict, success=True, statistics=statistics)

            return result

        except Exception as e:
            raise RuntimeError(f"Enhanced allocation workflow failed: {str(e)}") from e

    def run_wikipedia_and_equal_allocation(
        self,
        all_stocks: List[StockUniverse],
        optimizer_names: List[str],
        time_today: datetime,
        df_prices: pd.DataFrame,
        use_wikipedia: bool,
        use_sql_database: bool,
        n_historical_days: int,
    ) -> AllocationResult:
        """
        Gateway function that combines Wikipedia-based stock pre-selection with portfolio optimization.

        Steps:
        1. Use allocate_wikipedia for pre-selection of stocks with significant Wikipedia correlation
        2. Filter df_prices to only include pre-selected stocks
        3. Run portfolio optimization on the filtered dataset
        4. Return the optimized portfolio allocation

        Args:
            all_stocks: List of all available stocks
            time_today: Current date for analysis
            df_prices: DataFrame with historical price data for all stocks
            use_sql_database: Whether to use SQL database for data loading
            n_historical_days: Number of historical days to use for Wikipedia analysis

        Returns:
            AllocationResult from portfolio optimization with optimized weights
        """
        start_time = timer()

        try:
            if self._optimizer_names is None or self._optimizer_names != optimizer_names:
                self._optimizer_names = optimizer_names
                self._optimizer_list = get_optimizer_by_names(optimizer_names)

            if use_wikipedia:
                # Step 1: Get pre-selection from Wikipedia allocation
                wikipedia_result = allocate_wikipedia(
                    all_stocks=all_stocks,
                    time_today=time_today,
                    n_historical_days=n_historical_days,
                    use_sql_database=use_sql_database,
                )

                if not wikipedia_result.success:
                    # Return failed result if wikipedia allocation failed
                    end_time = timer()
                    return AllocationResult(
                        asset_weights=wikipedia_result.asset_weights,
                        success=False,
                        statistics=wikipedia_result.statistics,
                        computation_time=end_time - start_time,
                        error_message="Wikipedia allocation failed",
                    )

                # Step 2: Extract pre-selected stocks (those with non-zero weights)
                preselected_stocks = [
                    symbol for symbol, weight in wikipedia_result.asset_weights.items() if weight > 0.0
                ]

                logger.info(
                    f"Pre-selected {len(preselected_stocks)} / {len(all_stocks)} stocks from Wikipedia allocation"
                )

                if not preselected_stocks:
                    # Return failed result if no stocks were pre-selected
                    end_time = timer()
                    return AllocationResult(
                        asset_weights={},
                        success=False,
                        statistics=NoStatistics(),
                        computation_time=end_time - start_time,
                        error_message="No stocks pre-selected from Wikipedia",
                    )

            else:
                preselected_stocks = df_prices.columns.tolist()

            # Step 3: Filter df_prices to only include pre-selected stocks
            available_columns = [col for col in preselected_stocks if col in df_prices.columns]
            if not available_columns:
                # Return failed result if none of the pre-selected stocks are in df_prices
                end_time = timer()
                return AllocationResult(
                    asset_weights={},
                    success=False,
                    statistics=NoStatistics(),
                    computation_time=end_time - start_time,
                    error_message="No pre-selected stocks available in price data",
                )

            df_prices_filtered = df_prices[available_columns]

            # Step 4: Run portfolio optimization on filtered data
            optimization_result = self.run_equal_allocation_to_allocators(
                df_prices=df_prices_filtered,
            )

            end_time = timer()
            return AllocationResult(
                asset_weights=optimization_result.asset_weights,
                success=optimization_result.success,
                statistics=optimization_result.statistics,
                computation_time=end_time - start_time,
            )

        except Exception as e:
            end_time = timer()
            logger.error(f"Allocation gateway failed: {str(e)}")
            return AllocationResult(
                asset_weights={},
                success=False,
                statistics=NoStatistics(),
                computation_time=end_time - start_time,
                error_message=str(e),
            )


if __name__ == "__main__":
    # Generate proper test data: historical returns for moment estimation
    np.random.seed(42)
    n_observations = 100
    returns_data = np.random.multivariate_normal(
        mean=[0.1, 0.12, 0.08, 0.09, 0.11],
        cov=[
            [0.04, 0.01, 0.005, 0.002, 0.003],
            [0.01, 0.05, 0.002, 0.001, 0.004],
            [0.005, 0.002, 0.03, 0.001, 0.002],
            [0.002, 0.001, 0.001, 0.02, 0.003],
            [0.003, 0.004, 0.002, 0.003, 0.05],
        ],
        size=n_observations,
    )

    # Calculate expected returns and covariance from historical data
    mu = np.mean(returns_data, axis=0)  # This should be 1D array of expected returns
    cov = np.cov(returns_data, rowvar=False)

    print(f"Expected returns: {mu}")
    print(f"Expected returns shape: {mu.shape}")
    print(f"Covariance matrix shape: {cov.shape}")

    for optimizer in OPTIMIZER_LIST:
        optimizer.fit(returns_data)
        weights = optimizer.allocate(mu, cov, None, None)
        print(f"Optimal weights for {optimizer.name}: {weights}")
