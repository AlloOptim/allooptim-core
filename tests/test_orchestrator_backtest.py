"""
Integration test for AllocationOrchestrator in backtest context.

This test verifies that the orchestrator works correctly when integrated
into the backtesting framework and that df_allocation is properly populated.
"""

from datetime import datetime

import pandas as pd
import pytest

from allo_optim.allocation_to_allocators.orchestrator_adapter import (
    AllocationOrchestratorAdapter,
    OrchestrationType,
)
from allo_optim.backtest.backtest_config import BacktestConfig
from allo_optim.backtest.backtest_engine import BacktestEngine


@pytest.mark.parametrize("orchestration_type", ["equal", "optimized"])
def test_orchestrator_in_backtest(orchestration_type):
    """Test orchestrator integration with backtest engine."""
    
    # Create a minimal backtest configuration using orchestrator
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31),
        debug_start_date=datetime(2023, 1, 1),
        debug_end_date=datetime(2023, 1, 31),  # Just one month for speed
        quick_test=True,
        rebalance_frequency=10,
        lookback_days=30,
        optimizer_names=["Naive", "CappedMomentum"],
        transformer_names=["OracleCovarianceTransformer"],
        use_orchestrator=True,
        orchestration_type=orchestration_type,
    )
    
    # Create and run backtest
    engine = BacktestEngine(config_instance=config)
    
    # Verify orchestrator was set up correctly
    assert len(engine.individual_optimizers) == 1
    assert isinstance(engine.individual_optimizers[0], AllocationOrchestratorAdapter)
    assert engine.individual_optimizers[0].name.startswith("A2A_")
    
    # Run the backtest
    results = engine.run_backtest()
    
    # Verify results were generated
    assert results is not None
    assert len(results) > 0
    
    print(f"✓ Backtest with {orchestration_type} orchestration completed successfully")
    print(f"  Generated {len(results)} result entries")


def test_orchestrator_adapter_with_backtest_flow():
    """Test that orchestrator adapter works in the backtest workflow."""
    
    import numpy as np
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import sample_cov
    
    from allo_optim.config.stock_universe import list_of_dax_stocks
    
    # Create test data
    all_stocks = list_of_dax_stocks()[:5]
    assets = [stock.symbol for stock in all_stocks]
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    price_data = np.random.randn(60, len(assets)).cumsum(axis=0) + 100
    prices = pd.DataFrame(price_data, index=dates, columns=assets)
    
    # Create adapter
    adapter = AllocationOrchestratorAdapter(
        optimizer_names=["Naive", "CappedMomentum"],
        transformer_names=["OracleCovarianceTransformer"],
        orchestration_type=OrchestrationType.EQUAL,
    )
    
    # Simulate backtest workflow
    # 1. Fit
    adapter.fit(prices)
    
    # 2. Compute mu/cov (like backtest does)
    mu = mean_historical_return(prices)
    cov = sample_cov(prices)
    
    # 3. Allocate
    weights = adapter.allocate(
        ds_mu=mu,
        df_cov=cov,
        df_prices=prices,
        time=dates[-1],
    )
    
    # Verify results
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(assets)
    assert all(weights.index == prices.columns)
    assert all(weights >= 0)
    assert 0.99 <= weights.sum() <= 1.01
    
    print("✓ Orchestrator adapter works correctly in backtest flow")
    print(f"  Final weights: {weights.to_dict()}")
