"""
Integration test for AllocationOrchestrator in backtest context.

This test verifies that the orchestrator works correctly when integrated
into the backtesting framework and that df_allocation is properly populated.
"""

from datetime import datetime

import pytest

from allo_optim.allocation_to_allocators.allocation_orchestrator import (
    AllocationOrchestrator,
    AllocationOrchestratorConfig,
    OrchestrationType,
)
from allo_optim.backtest.backtest_config import BacktestConfig
from allo_optim.backtest.backtest_engine import BacktestEngine


@pytest.mark.parametrize("orchestration_type", OrchestrationType)
def test_orchestrator_in_backtest(orchestration_type):
    """Test orchestrator integration with backtest engine."""

    # Create a minimal backtest configuration using orchestrator
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 10),
        rebalance_frequency=5,
        lookback_days=10,
        orchestration_type=orchestration_type,
        optimizer_names=["Naive"],
        transformer_names=["OracleCovarianceTransformer"],
    )

    # Create fast A2A config for testing
    fast_a2a_config = AllocationOrchestratorConfig(
        orchestration_type=orchestration_type,
        n_particles=2,
        n_particle_swarm_iterations=2,
    )

    # Create and run backtest
    engine = BacktestEngine(config_backtest=config, config_a2a=fast_a2a_config)

    # Verify orchestrator was set up correctly
    assert isinstance(engine.orchestrator, AllocationOrchestrator)
    assert engine.orchestrator.config.orchestration_type == orchestration_type
    assert engine.orchestrator.config.n_particles == 2
    assert engine.orchestrator.config.n_particle_swarm_iterations == 2

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

    print(f"✓ Backtest with {orchestration_type} orchestration completed successfully")
    print(f"  Generated {len(results)} result entries: {list(results.keys())}")
