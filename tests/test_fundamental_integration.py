"""Integration tests for fundamental data providers with optimizers."""

import os
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd

from allooptim.allocation_to_allocators.orchestrator_factory import OrchestratorType, create_orchestrator
from allooptim.config.a2a_config import A2AConfig
from allooptim.data.fundamental_data import FundamentalData
from allooptim.data.provider_factory import FundamentalDataProviderFactory
from allooptim.optimizer.fundamental.fundamental_optimizer import BalancedFundamentalOptimizer
from allooptim.optimizer.optimizer_config import OptimizerConfig
from allooptim.optimizer.optimizer_factory import get_optimizer_by_config


class TestFundamentalIntegration:
    """Integration tests for fundamental providers with optimizers."""

    def test_optimizer_with_injected_provider(self):
        """Test optimizer with mock provider."""
        mock_provider = Mock()
        mock_provider.get_fundamental_data.return_value = [
            FundamentalData(ticker="AAPL", market_cap=3e12, roe=0.3, pb_ratio=5.0)
        ]

        optimizer = BalancedFundamentalOptimizer(data_provider=mock_provider)

        mu = pd.Series([0.1], index=["AAPL"])
        cov = pd.DataFrame([[0.04]], index=["AAPL"], columns=["AAPL"])

        weights = optimizer.allocate(mu, cov, time=datetime.now())

        assert len(weights) == 1
        assert abs(weights.sum() - 1.0) < 0.01
        mock_provider.get_fundamental_data.assert_called_once()

    def test_optimizer_factory_with_provider(self):
        """Test optimizer factory passes provider correctly."""
        mock_provider = Mock()
        mock_provider.get_fundamental_data.return_value = [
            FundamentalData(ticker="AAPL", market_cap=3e12, roe=0.3, pb_ratio=5.0)
        ]

        config = OptimizerConfig(name="BalancedFundamentalOptimizer", display_name="Test Fundamental")

        optimizers = get_optimizer_by_config([config], fundamental_data_provider=mock_provider)

        assert len(optimizers) == 1
        optimizer = optimizers[0]
        assert isinstance(optimizer, BalancedFundamentalOptimizer)
        assert optimizer.data_provider is mock_provider

    def test_optimizer_factory_without_provider(self):
        """Test optimizer factory creates default provider."""
        config = OptimizerConfig(name="BalancedFundamentalOptimizer", display_name="Test Fundamental")

        optimizers = get_optimizer_by_config([config])

        assert len(optimizers) == 1
        optimizer = optimizers[0]
        assert isinstance(optimizer, BalancedFundamentalOptimizer)
        assert optimizer.data_provider is not None
        # Should be UnifiedFundamentalProvider
        from allooptim.data.provider_factory import UnifiedFundamentalProvider

        assert isinstance(optimizer.data_provider, UnifiedFundamentalProvider)

    def test_orchestrator_with_provider(self):
        """Test orchestrator factory passes provider."""
        mock_provider = Mock()
        mock_provider.get_fundamental_data.return_value = [
            FundamentalData(ticker="AAPL", market_cap=3e12, roe=0.3, pb_ratio=5.0),
            FundamentalData(ticker="MSFT", market_cap=2e12, roe=0.25, pb_ratio=4.0),
        ]


        orchestrator = create_orchestrator(
            orchestrator_type=OrchestratorType.OPTIMIZED,
            optimizer_configs=[
                OptimizerConfig(name="BalancedFundamentalOptimizer"),
                OptimizerConfig(name="MomentumOptimizer"),
            ],
            transformer_names=[],  # Empty list instead of None
            a2a_config=A2AConfig(allocation_constraints={"max_active_assets": 2}),
            fundamental_data_provider=mock_provider,
        )

        # Verify fundamental optimizer has provider
        fundamental_opts = [o for o in orchestrator.optimizers if hasattr(o, "data_provider")]
        assert len(fundamental_opts) == 1
        assert fundamental_opts[0].data_provider is mock_provider

    def test_provider_factory_integration(self):
        """Test full integration with factory-created provider."""
        # This test uses the actual factory (with mocking to avoid network calls)
        with patch.dict(os.environ, {}, clear=True), \
             patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
            mock_yahoo.return_value = Mock()
            mock_yahoo.return_value.get_fundamental_data.return_value = [
                FundamentalData(ticker="AAPL", market_cap=3e12, roe=0.3, pb_ratio=5.0)
            ]
            mock_yahoo.return_value.supports_historical_data.return_value = False

            provider = FundamentalDataProviderFactory.create_provider()

            # Create optimizer with this provider
            optimizer = BalancedFundamentalOptimizer(data_provider=provider)

            mu = pd.Series([0.1], index=["AAPL"])
            cov = pd.DataFrame([[0.04]], index=["AAPL"], columns=["AAPL"])

            weights = optimizer.allocate(mu, cov, time=datetime.now())

            assert len(weights) == 1
            assert abs(weights.sum() - 1.0) < 0.01

    def test_fundamental_data_caching(self):
        """Test that caching works in integration."""
        from allooptim.data.fundamental_providers import YahooFinanceProvider
        from allooptim.data.provider_factory import UnifiedFundamentalProvider

        mock_provider = Mock(spec=YahooFinanceProvider)
        mock_provider.get_fundamental_data.return_value = [FundamentalData(ticker="AAPL", market_cap=3e12)]
        mock_provider.supports_historical_data.return_value = True

        provider = UnifiedFundamentalProvider([mock_provider], enable_caching=True)

        date = datetime(2023, 1, 1)

        # First call should hit provider
        result1 = provider.get_fundamental_data(["AAPL"], date)
        assert len(result1) == 1
        assert mock_provider.get_fundamental_data.call_count == 1

        # Second call should use cache
        result2 = provider.get_fundamental_data(["AAPL"], date)
        assert len(result2) == 1
        assert mock_provider.get_fundamental_data.call_count == 1  # Still 1

        # Results should be identical
        assert result1[0].ticker == result2[0].ticker
        assert result1[0].market_cap == result2[0].market_cap

    def test_multiple_tickers_integration(self):
        """Test integration with multiple tickers."""
        mock_provider = Mock()
        mock_provider.get_fundamental_data.return_value = [
            FundamentalData(ticker="AAPL", market_cap=3e12, roe=0.3, pb_ratio=5.0),
            FundamentalData(ticker="MSFT", market_cap=2e12, roe=0.25, pb_ratio=4.0),
            FundamentalData(ticker="GOOGL", market_cap=1.5e12, roe=0.2, pb_ratio=3.0),
        ]

        optimizer = BalancedFundamentalOptimizer(data_provider=mock_provider)

        tickers = ["AAPL", "MSFT", "GOOGL"]
        mu = pd.Series([0.1, 0.08, 0.12], index=tickers)
        cov = pd.DataFrame(
            [[0.04, 0.02, 0.01], [0.02, 0.03, 0.015], [0.01, 0.015, 0.05]], index=tickers, columns=tickers
        )

        weights = optimizer.allocate(mu, cov, time=datetime.now())

        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 0.01
        assert all(weights >= 0)  # Should be long-only

        # Should have called provider once
        mock_provider.get_fundamental_data.assert_called_once()
        args, kwargs = mock_provider.get_fundamental_data.call_args
        assert args[0] == tickers  # First arg is tickers
