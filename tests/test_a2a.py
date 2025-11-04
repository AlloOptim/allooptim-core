import numpy as np
import pandas as pd
import pytest

from allo_optim.allocation_to_allocators.allocation_orchestrator import (
    AllocationOrchestrator,
    AllocationResult,
    OrchestrationType,
)
from allo_optim.config.stock_universe import list_of_dax_stocks


@pytest.mark.parametrize("optimizer_names", [["Naive"], ["Naive", "CappedMomentum"]])
@pytest.mark.parametrize("orchestration_type", OrchestrationType)
def test_a2a(orchestration_type, optimizer_names, fast_a2a_config):
    """Test that all A2A allocators work correctly."""

    # Create sample price data for optimizers that need it
    all_stocks = list_of_dax_stocks()[:5]
    assets = [stock.symbol for stock in all_stocks]
    dates = pd.date_range("2025-09-05", periods=50, freq="D")
    price_data = np.random.randn(50, len(assets)).cumsum(axis=0) + 100
    prices = pd.DataFrame(price_data, index=dates, columns=assets)

    fast_a2a_config.orchestration_type = orchestration_type

    # For Wikipedia pipeline, disable database to avoid needing test database
    if orchestration_type == OrchestrationType.WIKIPEDIA_PIPELINE:
        fast_a2a_config.use_wiki_database = False

    orchestrator = AllocationOrchestrator(
        optimizer_names=optimizer_names,
        transformer_names=["OracleCovarianceTransformer"],
        config=fast_a2a_config,
    )

    result = orchestrator.run_allocation(
        all_stocks=all_stocks,
        time_today=prices.index[-1],
        df_prices=prices,
    )

    assert isinstance(result, AllocationResult)
    assert len(result.asset_weights) == len(assets)
    assert all(0 <= w <= 1.0 for w in result.asset_weights.values())
    assert 0.0 <= sum(result.asset_weights.values()) <= 1.0 + fast_a2a_config.weights_tolterance

    # Verify df_allocation is populated correctly
    assert result.df_allocation is not None
    assert isinstance(result.df_allocation, pd.DataFrame)
    assert result.df_allocation.shape[0] == len(optimizer_names)  # rows = optimizers
    assert result.df_allocation.shape[1] == len(assets)  # cols = assets
    assert list(result.df_allocation.columns) == assets  # same asset order
