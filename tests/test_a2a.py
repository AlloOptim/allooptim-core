import numpy as np
import pandas as pd
import pytest

from allo_optim.allocation_to_allocators.allocation_orechtrator import AllocationOrchestrator, OrchestrationType, AllocationOrchestratorConfig, AllocationResult
from allo_optim.config.stock_universe import list_of_dax_stocks


@pytest.mark.parametrize("orchestration_type", OrchestrationType)
def test_a2a(orchestration_type):
    """Test that all optimizers in OPTIMIZER_LIST work correctly."""

    # Create sample price data for optimizers that need it
    all_stocks = list_of_dax_stocks()[:5]
    assets = [stock.symbol for stock in all_stocks]
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    price_data = np.random.randn(50, 3).cumsum(axis=0) + 100
    prices = pd.DataFrame(price_data, index=dates, columns=assets)

    config = AllocationOrchestratorConfig(
        orchestration_type=orchestration_type,
    )

    orchestrator = AllocationOrchestrator(
        optimizer_names = ["NaiveOptimizer", "CappedMomentum"],
        transformer_names = ["OracleCovarianceTransformer"],
        config=config,
        )

    result = orchestrator.run_allocation(
        all_stocks=all_stocks,
        time_today=prices.index[-1],
        df_prices=prices,
    )

    assert isinstance(result, AllocationResult)
    assert len(result.asset_weights) == len(assets)
    assert all(0 <= w <= 1.0 for w in result.asset_weights.values())
    assert 0.0 <= sum(result.asset_weights.values()) <= 1.0 + config.weights_tolerance