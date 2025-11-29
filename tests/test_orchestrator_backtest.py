"""Integration test for AllocationOrchestrator in backtest context.

This test verifies that the orchestrator works correctly when integrated
into the backtesting framework and that df_allocation is properly populated.
"""

import warnings
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from allooptim.allocation_to_allocators.a2a_manager_config import A2AManagerConfig
from allooptim.allocation_to_allocators.orchestrator_factory import OrchestratorType
from allooptim.backtest.backtest_engine import BacktestEngine
from allooptim.config.a2a_config import A2AConfig
from allooptim.config.allocation_dataclasses import (
    AllocationResult,
    WikipediaStatistics,
)
from allooptim.config.backtest_config import BacktestConfig


@pytest.mark.parametrize(
    "orchestrator_type",
    OrchestratorType,
)
def test_orchestrator_in_backtest(orchestrator_type, fast_a2a_config):
    """Test orchestrator integration with backtest engine."""
    # Suppress numpy and pandas warnings that can occur during statistical calculations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)

        # Create a minimal backtest configuration using orchestrator
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            rebalance_frequency=5,
        )

        # Create A2A config with fast test parameters
        if orchestrator_type == OrchestratorType.CUSTOM_WEIGHT:
            a2a_config = A2AConfig(
                n_simulations=10,  # Fast for testing
                n_pso_iterations=5,
                n_particles=10,
                custom_a2a_weights={"NaiveOptimizer": 0.5, "MomentumOptimizer": 0.5},  # Custom weights for each optimizer
            )
        else:
            a2a_config = A2AConfig(
                n_simulations=10,  # Fast for testing
                n_pso_iterations=5,
                n_particles=10,
            )

        a2a_manager_config = A2AManagerConfig(
            lookback_days=5,  # Minimal lookback
            optimizer_configs=["NaiveOptimizer", "MomentumOptimizer"],
            transformer_names=["OracleCovarianceTransformer"],
            orchestration_type=orchestrator_type,
            benchmark="SPY",
            a2a_config=a2a_config,
        )    # Create and run backtest with specified orchestrator type
    engine = BacktestEngine(
        config_backtest=backtest_config,
        a2a_manager_config=a2a_manager_config,
    )

    # Verify orchestrator was set up correctly
    assert engine.a2a_manager.orchestrator is not None
    assert engine.a2a_manager.orchestrator.name is not None

    # Create synthetic price data
    dates = pd.date_range("2022-12-20", "2023-01-15", freq="D")  # Cover the backtest period
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]  # Include SPY for benchmark
    np.random.seed(123)  # For reproducible results
    # Generate more stable price data to avoid NaN/inf issues
    price_changes = np.random.normal(0.001, 0.02, (len(dates), len(symbols)))  # Small daily returns
    price_data = 100 * np.exp(np.cumsum(price_changes, axis=0))  # Geometric Brownian motion
    synthetic_prices = pd.DataFrame(price_data, index=dates, columns=symbols)

    # Ensure no NaN or inf values in the synthetic data
    synthetic_prices = synthetic_prices.replace([np.inf, -np.inf], np.nan).dropna()
    assert not synthetic_prices.isna().any().any(), "Synthetic prices contain NaN values"
    assert not np.isinf(synthetic_prices.values).any(), "Synthetic prices contain infinite values"

    # Mock the data loader and Wikipedia allocation for speed
    with patch.object(engine.a2a_manager.data_loader, "load_price_data", return_value=synthetic_prices):
        if orchestrator_type == OrchestratorType.WIKIPEDIA_PIPELINE:
            # Mock allocate_wikipedia to return quick result for testing
            mock_wiki_result = AllocationResult(
                asset_weights={"AAPL": 0.5, "MSFT": 0.5, "GOOGL": 0.0, "AMZN": 0.0, "TSLA": 0.0},
                success=True,
                statistics=WikipediaStatistics(
                    end_date="2023-01-01",
                    r_squared=0.5,
                    p_value=0.01,
                    std_err=0.1,
                    slope=0.2,
                    intercept=0.0,
                    all_symbols=symbols,
                    valid_data_symbols=symbols,
                    significant_positive_stocks=["AAPL", "MSFT"],
                    top_n_symbols=["AAPL", "MSFT"],
                ),
            )
            with patch(
                "allooptim.allocation_to_allocators.wikipedia_pipeline_orchestrator.allocate_wikipedia",
                return_value=mock_wiki_result,
            ):
                # Run the backtest
                results = engine.run_backtest()
        else:
            # Run the backtest
            results = engine.run_backtest()

    # Verify results were generated
    assert results is not None
    assert len(results) > 0

    # Check that we have the expected result keys
    expected_keys = ["NaiveOptimizer", "MomentumOptimizer", "SPY"]
    for key in expected_keys:
        assert key in results, f"Missing expected result key: {key}"
        assert "metrics" in results[key], f"Missing metrics for {key}"
        assert "portfolio_values" in results[key], f"Missing portfolio_values for {key}"

    print(f"✓ Backtest with {orchestrator_type} orchestration completed successfully")
    print(f"  Generated {len(results)} result entries: {list(results.keys())}")
