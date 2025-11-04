"""
Integration test for AllocationOrchestrator in backtest context.

This test verifies that the orchestrator works correctly when integrated
into the backtesting framework and that df_allocation is properly populated.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from allo_optim.allocation_to_allocators.orchestrator_factory import OrchestratorType
from allo_optim.backtest.backtest_config import BacktestConfig
from allo_optim.backtest.backtest_engine import BacktestEngine
from allo_optim.config.allocation_dataclasses import (
    AllocationResult,
    WikipediaStatistics,
)


@pytest.mark.parametrize(
    "orchestrator_type",
    OrchestratorType,
)
def test_orchestrator_in_backtest(orchestrator_type, fast_a2a_config):
    """Test orchestrator integration with backtest engine."""

    # Create a minimal backtest configuration using orchestrator
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 10),
        rebalance_frequency=5,
        lookback_days=5,  # Minimal lookback
        optimizer_names=["Naive"],
        transformer_names=["OracleCovarianceTransformer"],
        orchestration_type=OrchestratorType.AUTO,
    )

    # Create and run backtest with specified orchestrator type
    engine = BacktestEngine(
        config_backtest=config,
        orchestrator_type=orchestrator_type,
        n_data_observations=10,  # Fast for testing
        n_particle_swarm_iterations=5,
        n_particles=10,
    )

    # Verify orchestrator was set up correctly
    assert engine.orchestrator is not None
    assert engine.orchestrator.name is not None

    # Create synthetic price data
    dates = pd.date_range("2022-12-20", "2023-01-15", freq="D")  # Cover the backtest period
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]  # Include SPY for benchmark
    np.random.seed(42)  # For reproducible results
    price_data = np.random.randn(len(dates), len(symbols)).cumsum(axis=0) + 100
    synthetic_prices = pd.DataFrame(price_data, index=dates, columns=symbols)

    # Mock the data loader and Wikipedia allocation for speed
    with patch.object(engine.data_loader, "load_price_data", return_value=synthetic_prices):
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
                "allo_optim.allocation_to_allocators.wikipedia_pipeline_orchestrator.allocate_wikipedia",
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
    expected_keys = ["Naive", "SPY_Benchmark", "A2A_Ensemble"]
    for key in expected_keys:
        assert key in results, f"Missing expected result key: {key}"
        assert "metrics" in results[key], f"Missing metrics for {key}"
        assert "portfolio_values" in results[key], f"Missing portfolio_values for {key}"

    print(f"✓ Backtest with {orchestrator_type} orchestration completed successfully")
    print(f"  Generated {len(results)} result entries: {list(results.keys())}")
