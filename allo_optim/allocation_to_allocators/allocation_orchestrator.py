"""
Enhanced Allocation Workflow

Clean orchestration of the complete allocation-to-allocators process.
Integrates MCOS simulation with allocation optimization.
"""

import logging
from datetime import datetime
from enum import Enum
from timeit import default_timer as timer
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
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
from allo_optim.covariance_transformer.transformer_list import (
    get_transformer_by_names,
)
from allo_optim.optimizer.allocation_metric import (
    estimate_linear_moments,
    validate_no_nan,
)
from allo_optim.optimizer.optimizer_list import (
    get_optimizer_by_names,
)
from allo_optim.optimizer.wikipedia.allocate_wikipedia import allocate_wikipedia

logger = logging.getLogger(__name__)


class OrchestrationType(str, Enum):
    EQUAL = "equal"
    OPTIMIZED = "optimized"
    WIKIPEDIA_PIPELINE = "wikipedia_pipeline"


class AllocationOrchestratorConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    weights_tolterance: float = Field(0.01, ge=0.0, le=0.1)

    orchestration_type: OrchestrationType = OrchestrationType.EQUAL

    allocator_weights: Optional[np.ndarray] = None

    n_data_observations: int = Field(50, ge=1)
    n_particle_swarm_iterations: int = Field(1000, ge=1)
    n_particles: int = Field(1000, ge=1)

    use_sql_database: bool = False
    n_historical_days: int = Field(60, ge=1)
    
    # Timezone for timestamp localization (e.g., "UTC", "US/Eastern")
    timezone: str = Field("UTC")

    @field_validator("orchestration_type")
    @classmethod
    def validate_weights_tolerance(cls, v) -> OrchestrationType:
        return OrchestrationType(v)


class AllocationOrchestrator:
    def __init__(
        self,
        optimizer_names: list[str] = None,
        transformer_names: list[str] = None,
        config: Optional[AllocationOrchestratorConfig] = None,
    ) -> None:
        self._optimizers = get_optimizer_by_names(optimizer_names)
        self._transformers = get_transformer_by_names(transformer_names)

        self.config = config or AllocationOrchestratorConfig()

    def fit_optimizers(self, df_prices: pd.DataFrame) -> None:
        """
        Fit all optimizers with the provided price data.
        """
        for optimizer in self._optimizers:
            optimizer.fit(df_prices)

    def run_allocation(
        self,
        all_stocks: List[StockUniverse],
        time_today: datetime,
        df_prices: pd.DataFrame,
    ) -> AllocationResult:
        match self.config.orchestration_type:
            case OrchestrationType.EQUAL:
                return self._run_equal_allocation_to_allocators(
                    df_prices=df_prices,
                    time_today=time_today,
                )

            case OrchestrationType.OPTIMIZED:
                return self._run_optimized_allocation_to_allocators(
                    df_prices=df_prices,
                    time_today=time_today,
                )

            case OrchestrationType.WIKIPEDIA_PIPELINE:
                return self._run_wikipedia_and_equal_allocation(
                    all_stocks=all_stocks,
                    time_today=time_today,
                    df_prices=df_prices,
                )

    def _check_weights_bounds(self, weights: np.ndarray, name: str) -> np.ndarray:
        if np.sum(weights) < -self.config.weights_tolterance or np.sum(weights) > 1.0 + self.config.weights_tolterance:
            raise ValueError(f"{name} must sum to 1.0, got {np.sum(weights)}")

        if any(weights < -self.config.weights_tolterance) or any(weights > 1.0 + self.config.weights_tolterance):
            raise ValueError(f"{name} contains negative or leveraged values")

        if any(weights < 0.0) or any(weights > 1.0):
            logger.warning(f"{name} contains invalid weights, clipping to [0.0, 1.0]")
            weights = np.clip(weights, 0.0, 1.0)

        if np.sum(weights) < 0.0:
            logger.warning(f"{name} sums to {np.sum(weights)} < 0.0, adjusting to 0.0")
            weights = np.zeros_like(weights)

        if (
            np.sum(weights) > 1.0 + self.config.weights_tolterance
            or np.sum(weights) < 1.0 - self.config.weights_tolterance
        ):
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
        time_today: datetime,
        allocator_weights: np.ndarray,
    ) -> tuple:
        """
        Compute final asset weights and performance statistics.

        Args:
            df_assets: Asset price DataFrame
            allocator_weights: Optimal weights for each allocator

        Returns:
            Tuple of (asset_weights_dict, statistics_dict, df_allocation)
        """

        if len(self._optimizers) != len(allocator_weights):
            raise ValueError(f"Optimizer count {len(self._optimizers)} != weight count {len(allocator_weights)}")

        validate_no_nan(allocator_weights, "allocator weights")

        # Compute asset statistics
        l_moments = estimate_linear_moments(df_prices)

        # Keep as pandas Series/DataFrame for new optimizer interface
        mu = mean_historical_return(df_prices)  # Returns pandas Series
        cov = sample_cov(df_prices)  # Returns pandas DataFrame

        for transformer in self._transformers:
            cov = transformer.transform(cov, n_observations=len(df_prices))  # Returns pandas DataFrame

        validate_no_nan(mu.values, "historical returns")
        validate_no_nan(cov.values, "sample covariance")

        # Initialize tracking
        asset_weights = np.zeros(len(mu))
        statistic_returns = {}
        statistic_volatilities = {}
        statistic_runtime = {"A2A": np.nan}
        
        # Track individual optimizer allocations for df_allocation
        optimizer_allocations = {}

        # Compute weighted asset allocation
        for k, optimizer in enumerate(self._optimizers):
            start = timer()

            try:
                logger.info(f"Computing allocation for {optimizer.name}...")

                optimizer.fit(df_prices)

                weights = optimizer.allocate(mu, cov, df_prices, time_today, l_moments)  # Returns pandas Series
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
            
            # Store optimizer allocation for df_allocation DataFrame
            optimizer_allocations[optimizer.name] = weights

        # Build df_allocation DataFrame (rows=optimizers, cols=assets)
        df_allocation = pd.DataFrame(optimizer_allocations).T if optimizer_allocations else None

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
            optimizer.name: weight for optimizer, weight in zip(self._optimizers, allocator_weights)
        }

        asset_weights_dict = {col: weight for col, weight in zip(df_prices.columns, asset_weights)}

        statistics_dict = {
            "returns": statistic_returns,
            "volatilities": statistic_volatilities,
            "runtime": statistic_runtime,
            "algo_weights": statistic_algo_weights,
            "asset_weights": asset_weights_dict,
        }

        return asset_weights_dict, statistics_dict, df_allocation

    def _run_optimized_allocation_to_allocators(
        self,
        df_prices: pd.DataFrame,
        time_today: datetime,
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

        if len(self._optimizers) == 0:
            raise ValueError("optimizer_list cannot be empty")

        try:
            # Step 1: Run enhanced MCOS simulation
            mcos_result = simulate_optimizers_with_allocation_statistics(
                df_assets=df_prices,
                optimizer_list=self._optimizers,
                n_data_observations=self.config.n_data_observations,
            )

            # Validate MCOS result
            if mcos_result.expected_return_means is None:
                raise ValueError("MCOS simulation failed to generate expected return means")

            if mcos_result.expected_returns_covariance is None:
                raise ValueError("MCOS simulation failed to generate expected returns covariance")

            if len(mcos_result.expected_return_means) != len(self._optimizers):
                raise ValueError(
                    f"MCOS generated {len(mcos_result.expected_return_means)} optimizer means "
                    f"but expected {len(self._optimizers)}"
                )

            # Step 2: Optimize allocator weights
            allocator_weights = optimize_allocator_weights(
                mcos_result=mcos_result,
                n_particle_swarm_iterations=self.config.n_particle_swarm_iterations,
                n_particles=self.config.n_particles,
            )

            # Step 3: Compute final asset weights and statistics
            asset_weights_dict, statistics_dict, df_allocation = self._compute_final_asset_weights(
                df_prices=df_prices,
                time_today=time_today,
                allocator_weights=allocator_weights,
            )

            # Build result
            statistics = A2AStatistics(
                asset_returns=statistics_dict["returns"],
                asset_volatilities=statistics_dict["volatilities"],
                algo_runtime=statistics_dict["runtime"],
                algo_weights=statistics_dict["algo_weights"],
            )

            result = AllocationResult(
                asset_weights=asset_weights_dict, 
                success=True, 
                statistics=statistics,
                df_allocation=df_allocation
            )

            return result

        except Exception as e:
            raise RuntimeError(f"Enhanced allocation workflow failed: {str(e)}") from e

    def _run_equal_allocation_to_allocators(
        self,
        df_prices: pd.DataFrame,
        time_today: datetime,
    ) -> AllocationResult:
        try:
            if df_prices.isna().any().any():
                raise ValueError("df_assets contains NaN values")

            if self._optimizers is None or len(self._optimizers) == 0:
                raise ValueError("optimizer_list cannot be empty")

            if self.config.allocator_weights is None:
                allocator_weights = 1.0 / len(self._optimizers) * np.ones(len(self._optimizers))
            else:
                allocator_weights = self.config.allocator_weights

            if len(self._optimizers) != len(allocator_weights):
                raise ValueError(f"Optimizer count {len(self._optimizers)} != weight count {len(allocator_weights)}")

            if not np.isclose(np.sum(allocator_weights), 1.0):
                raise ValueError(f"Allocator weights must sum to 1.0, got {np.sum(allocator_weights)}")

            asset_weights_dict, statistics_dict, df_allocation = self._compute_final_asset_weights(
                df_prices=df_prices,
                time_today=time_today,
                allocator_weights=allocator_weights,
            )

            statistics = A2AStatistics(
                asset_returns=statistics_dict["returns"],
                asset_volatilities=statistics_dict["volatilities"],
                algo_runtime=statistics_dict["runtime"],
                algo_weights=statistics_dict["algo_weights"],
            )

            result = AllocationResult(
                asset_weights=asset_weights_dict, 
                success=True, 
                statistics=statistics,
                df_allocation=df_allocation
            )

            return result

        except Exception as e:
            raise RuntimeError(f"Enhanced allocation workflow failed: {str(e)}") from e

    def _run_wikipedia_and_equal_allocation(
        self,
        all_stocks: List[StockUniverse],
        time_today: datetime,
        df_prices: pd.DataFrame,
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
            # Localize time_today if it's timezone-naive
            if time_today.tzinfo is None:
                time_today = time_today.tz_localize(self.config.timezone)
            
            # Step 1: Get pre-selection from Wikipedia allocation
            wikipedia_result = allocate_wikipedia(
                all_stocks=all_stocks,
                time_today=time_today,
                n_historical_days=self.config.n_historical_days,
                use_sql_database=self.config.use_sql_database,
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
            preselected_stocks = [symbol for symbol, weight in wikipedia_result.asset_weights.items() if weight > 0.0]

            logger.info(f"Pre-selected {len(preselected_stocks)} / {len(all_stocks)} stocks from Wikipedia allocation")

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
            optimization_result = self._run_equal_allocation_to_allocators(
                df_prices=df_prices_filtered,
                time_today=time_today,
            )

            # Step 5: Pad weights with zeros for non-selected assets
            # Create full weight dict with all original assets
            all_assets = df_prices.columns.tolist()
            asset_weights_padded = {asset: 0.0 for asset in all_assets}
            
            # Update with actual weights from optimization
            if optimization_result.asset_weights is not None:
                asset_weights_padded.update(optimization_result.asset_weights)

            # Pad df_allocation with zeros for non-selected assets
            df_allocation_padded = None
            if optimization_result.df_allocation is not None:
                df_allocation_padded = optimization_result.df_allocation.copy()
                # Add missing assets with zero weights
                for asset in all_assets:
                    if asset not in df_allocation_padded.columns:
                        df_allocation_padded[asset] = 0.0
                # Reorder to match original asset order
                df_allocation_padded = df_allocation_padded[all_assets]

            end_time = timer()
            return AllocationResult(
                asset_weights=asset_weights_padded,
                success=optimization_result.success,
                statistics=optimization_result.statistics,
                computation_time=end_time - start_time,
                df_allocation=df_allocation_padded,
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
