"""Test the AllocationOrchestratorAdapter"""
import numpy as np
import pandas as pd

from allo_optim.allocation_to_allocators.orchestrator_adapter import (
    AllocationOrchestratorAdapter,
    OrchestrationType,
)
from allo_optim.config.stock_universe import list_of_dax_stocks


def test_orchestrator_adapter():
    """Test that adapter works with all orchestration types."""
    
    # Create sample data
    all_stocks = list_of_dax_stocks()[:5]
    assets = [stock.symbol for stock in all_stocks]
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    price_data = np.random.randn(50, len(assets)).cumsum(axis=0) + 100
    prices = pd.DataFrame(price_data, index=dates, columns=assets)
    
    for orch_type in [OrchestrationType.EQUAL, OrchestrationType.OPTIMIZED]:
        adapter = AllocationOrchestratorAdapter(
            optimizer_names=["Naive", "CappedMomentum"],
            transformer_names=["OracleCovarianceTransformer"],
            orchestration_type=orch_type,
        )
        
        # Test fit
        adapter.fit(prices)
        
        # Test allocate
        mu = prices.pct_change().mean()
        cov = prices.pct_change().cov()
        
        weights = adapter.allocate(
            ds_mu=mu,
            df_cov=cov,
            df_prices=prices,
            time=prices.index[-1],
        )
        
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(assets)
        assert all(weights >= 0)
        assert 0.99 <= weights.sum() <= 1.01  # Allow small numerical error
        
        print(f"✓ {adapter.name} passed")
    
    print("All adapter tests passed!")


if __name__ == "__main__":
    test_orchestrator_adapter()
