import numpy as np
import pandas as pd
import pytest

from allooptim.allocation_to_allocators.a2a_result import A2AResult
from allooptim.allocation_to_allocators.data_provider_factory import (
    get_data_provider_factory,
)
from allooptim.allocation_to_allocators.orchestrator_factory import (
    OrchestratorType,
    create_orchestrator,
)
from allooptim.config.stock_universe import list_of_dax_stocks


@pytest.mark.parametrize("optimizer_names", [["NaiveOptimizer"], ["NaiveOptimizer", "MomentumOptimizer"]])
@pytest.mark.parametrize("orchestrator_type", OrchestratorType)
def test_a2a(orchestrator_type, optimizer_names, fast_a2a_config):
    """Test that all A2A allocators work correctly."""

    # Create sample price data for optimizers that need it
    all_stocks = list_of_dax_stocks()[:5]
    assets = [stock.symbol for stock in all_stocks]
    dates = pd.date_range("2025-09-05", periods=50, freq="D")
    price_data = np.random.randn(50, len(assets)).cumsum(axis=0) + 100
    prices = pd.DataFrame(price_data, index=dates, columns=assets)

    # For Wikipedia pipeline, add specific kwargs
    kwargs = {}
    if orchestrator_type == OrchestratorType.WIKIPEDIA_PIPELINE:
        kwargs = {
            "n_historical_days": 30,
            "use_wiki_database": False,
        }

    orchestrator = create_orchestrator(
        orchestrator_type=orchestrator_type,
        optimizer_names=optimizer_names,
        transformer_names=["OracleCovarianceTransformer"],
        config=fast_a2a_config,
        **kwargs,
    )

    # Create data provider for backtest context
    data_provider_factory = get_data_provider_factory("backtest")
    data_provider = data_provider_factory.create_data_provider(prices)

    result = orchestrator.allocate(
        data_provider=data_provider,
        time_today=prices.index[-1],
        all_stocks=all_stocks,
    )

    assert isinstance(result, A2AResult)
    assert len(result.final_allocation) == len(assets)
    assert all(0 <= w <= 1.0 for w in result.final_allocation.values)
    assert abs(sum(result.final_allocation.values) - 1.0) < 1e-6  # Should sum to 1.0

    # Verify df_allocation is populated correctly
    df_allocation = result.to_dataframe()
    assert df_allocation is not None
    assert isinstance(df_allocation, pd.DataFrame)
    assert df_allocation.shape[0] == len(assets)  # rows = assets
    assert df_allocation.shape[1] == len(optimizer_names)  # cols = optimizers
    assert list(df_allocation.index) == assets  # same asset order
