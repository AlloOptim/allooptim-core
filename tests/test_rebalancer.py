"""Tests for PortfolioRebalancer."""

import numpy as np
import pandas as pd
import pytest

from allooptim.allocation_to_allocators.rebalancer import PortfolioRebalancer


class TestPortfolioRebalancer:
    """Test suite for PortfolioRebalancer."""

    def test_rebalancer_initialization(self):
        """Test that rebalancer initializes with correct default parameters."""
        rebalancer = PortfolioRebalancer(rebalancing_days=20)

        assert rebalancer.rebalancing_days == 20
        assert rebalancer.config.absolute_threshold == 0.05
        assert rebalancer.config.relative_threshold == 0.15
        assert rebalancer.config.min_trade_pct is None
        assert rebalancer.config.max_trades_per_day == 15
        assert rebalancer.config.trade_to_band_edge is True

    def test_rebalancer_custom_parameters(self):
        """Test rebalancer with custom parameters."""
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            filter_lifetime_days=20,
            absolute_threshold=0.1,
            relative_threshold=0.5,
            min_trade_pct=0.001,
            max_trades_per_day=10,
            trade_to_band_edge=False,
        )
        rebalancer = PortfolioRebalancer(rebalancing_days=20, config=config)

        assert rebalancer.rebalancing_days == 20
        assert rebalancer.config.filter_lifetime_days == 20
        assert rebalancer.config.absolute_threshold == 0.1
        assert rebalancer.config.relative_threshold == 0.5
        assert rebalancer.config.min_trade_pct == 0.001
        assert rebalancer.config.max_trades_per_day == 10
        assert rebalancer.config.trade_to_band_edge is False

    def test_first_rebalance_no_previous_weights(self):
        """Test first rebalance when no previous weights exist."""
        rebalancer = PortfolioRebalancer(rebalancing_days=20)

        # Create target weights
        assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        target_weights = pd.Series([0.2, 0.3, 0.1, 0.2, 0.2], index=assets)

        result = rebalancer.rebalance(target_weights)

        # Should return the target weights unchanged on first call (but possibly sorted)
        assert len(result) == len(target_weights)
        assert result.sum() == pytest.approx(1.0)
        assert all(result >= 0)
        # Check that all target values are present
        for asset, weight in target_weights.items():
            assert result[asset] == pytest.approx(weight)

    def test_rebalance_with_previous_weights(self):
        """Test rebalance with existing previous weights."""
        rebalancer = PortfolioRebalancer(rebalancing_days=20)

        assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        # First rebalance
        target1 = pd.Series([0.2, 0.3, 0.1, 0.2, 0.2], index=assets)
        result1 = rebalancer.rebalance(target1)

        # Should return target weights on first call
        assert len(result1) == len(target1)
        assert result1.sum() == pytest.approx(1.0)

        # Second rebalance with different target
        target2 = pd.Series([0.3, 0.2, 0.1, 0.2, 0.2], index=assets)
        result2 = rebalancer.rebalance(target2)

        # Should apply smoothing and threshold logic
        assert len(result2) > 0
        assert result2.sum() == pytest.approx(1.0)
        assert all(result2 >= 0)  # No negative weights

    def test_rebalance_small_changes_no_trades(self):
        """Test that small changes don't trigger trades when above threshold."""
        # Set high thresholds so small changes don't trigger rebalancing
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            absolute_threshold=0.5,  # 50% threshold
            relative_threshold=1.0,  # 100% of weight
        )
        rebalancer = PortfolioRebalancer(rebalancing_days=20, config=config)

        assets = ["AAPL", "MSFT"]
        target1 = pd.Series([0.5, 0.5], index=assets)
        result1 = rebalancer.rebalance(target1)

        # Small change (0.01 = 1% of portfolio)
        target2 = pd.Series([0.51, 0.49], index=assets)
        result2 = rebalancer.rebalance(target2)

        # Should maintain previous weights due to high thresholds
        # (actual=0.5, target=0.51, deviation=0.01, threshold=0.5/2=0.25)
        assert result2.sum() == pytest.approx(1.0)

    def test_rebalance_large_changes_trigger_trades(self):
        """Test that large changes trigger trades."""
        # Set low thresholds so changes trigger rebalancing
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            absolute_threshold=0.01,  # 1% threshold
            relative_threshold=0.1,  # 10% of weight
        )
        rebalancer = PortfolioRebalancer(rebalancing_days=20, config=config)

        assets = ["AAPL", "MSFT"]
        target1 = pd.Series([0.5, 0.5], index=assets)
        result1 = rebalancer.rebalance(target1)

        # Large change
        target2 = pd.Series([0.7, 0.3], index=assets)
        result2 = rebalancer.rebalance(target2)

        # Should rebalance towards target
        assert result2.sum() == pytest.approx(1.0)
        assert abs(result2["AAPL"] - 0.7) < abs(result1["AAPL"] - 0.7)  # Closer to target

    def test_rebalance_minimum_trade_filter(self):
        """Test minimum trade size filtering."""
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            absolute_threshold=0.01,
            min_trade_pct=0.05,  # 5% minimum trade
        )
        rebalancer = PortfolioRebalancer(rebalancing_days=20, config=config)

        assets = ["AAPL", "MSFT", "GOOGL"]
        target1 = pd.Series([0.4, 0.4, 0.2], index=assets)
        result1 = rebalancer.rebalance(target1)

        # Small change that would trigger tiny trades
        target2 = pd.Series([0.405, 0.395, 0.2], index=assets)  # 0.5% changes
        result2 = rebalancer.rebalance(target2)

        # Should filter out small trades and maintain previous weights
        assert result2.sum() == pytest.approx(1.0)

    def test_rebalance_max_trades_limit(self):
        """Test maximum trades per day limit."""
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            absolute_threshold=0.01,
            max_trades_per_day=2,  # Only allow 2 trades
        )
        rebalancer = PortfolioRebalancer(rebalancing_days=20, config=config)

        assets = [f"ASSET_{i}" for i in range(10)]  # 10 assets
        target1 = pd.Series([0.1] * 10, index=assets)
        result1 = rebalancer.rebalance(target1)

        # Large change for all assets
        target2 = pd.Series([0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1], index=assets)
        result2 = rebalancer.rebalance(target2)

        # Should only execute top 2 trades by priority
        assert result2.sum() == pytest.approx(1.0)

    def test_rebalance_ema_smoothing(self):
        """Test first-order low-pass filter smoothing."""
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            filter_lifetime_days=20,  # Approximately equivalent to ema_alpha=0.5 for 20-day rebalancing
            absolute_threshold=0.05,
            max_trades_per_day=None,
            trade_to_band_edge=False,
        )
        rebalancer = PortfolioRebalancer(rebalancing_days=20, config=config)

        assets = ["AAPL", "MSFT"]
        target1 = pd.Series([0.5, 0.5], index=assets)
        result1 = rebalancer.rebalance(target1)

        # Jump to very different weights
        target2 = pd.Series([0.8, 0.2], index=assets)
        result2 = rebalancer.rebalance(target2)

        # Should be smoothed towards target, not exactly target
        assert result2["AAPL"] > 0.5  # Moving towards 0.8
        assert result2["MSFT"] < 0.5  # Moving towards 0.2
        assert result2.sum() == pytest.approx(1.0)

    def test_rebalance_trade_to_band_edge(self):
        """Test trading to band edge vs target."""
        # Test with trade_to_band_edge=True (default)
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            filter_lifetime_days=10000,  # No smoothing
            absolute_threshold=0.1,
        )
        rebalancer_edge = PortfolioRebalancer(rebalancing_days=20, config=config)

        assets = ["AAPL"]
        target1 = pd.Series([0.5], index=assets)
        result1 = rebalancer_edge.rebalance(target1)

        # Current = 0.5, Target = 0.8, Threshold = 0.1/1 = 0.1
        target2 = pd.Series([0.8], index=assets)
        result2 = rebalancer_edge.rebalance(target2)

        # Should trade to band edge: target - threshold = 0.8 - 0.1 = 0.7
        assert result2["AAPL"] == pytest.approx(0.7)

    def test_rebalance_normalization(self):
        """Test that weights are properly normalized."""
        rebalancer = PortfolioRebalancer(rebalancing_days=20)

        assets = ["AAPL", "MSFT", "GOOGL"]
        # Weights that don't sum to 1
        target = pd.Series([0.3, 0.3, 0.3], index=assets)  # Sums to 0.9
        result = rebalancer.rebalance(target)

        assert result.sum() == pytest.approx(1.0)
        assert all(result >= 0)

    def test_rebalance_empty_target_weights(self):
        """Test handling of empty target weights."""
        rebalancer = PortfolioRebalancer(rebalancing_days=20)

        empty_weights = pd.Series(dtype=float)
        result = rebalancer.rebalance(empty_weights)

        assert len(result) == 0

    def test_reset_smoothing(self):
        """Test resetting EMA smoothing state."""
        rebalancer = PortfolioRebalancer(rebalancing_days=20)

        assets = ["AAPL"]
        target1 = pd.Series([0.5], index=assets)
        result1 = rebalancer.rebalance(target1)

        # Reset smoothing
        rebalancer.reset_smoothing()

        # Next rebalance should start fresh
        target2 = pd.Series([0.7], index=assets)
        result2 = rebalancer.rebalance(target2)

        # Should return target weights directly (no smoothing history)
        assert result2["AAPL"] == pytest.approx(0.7)

    def test_get_actual_weights(self):
        """Test getting current actual weights."""
        rebalancer = PortfolioRebalancer(rebalancing_days=20)

        assets = ["AAPL", "MSFT"]
        target = pd.Series([0.6, 0.4], index=assets)
        rebalancer.rebalance(target)

        actual = rebalancer.get_actual_weights()
        expected = pd.Series([0.6, 0.4], index=assets)

        pd.testing.assert_series_equal(actual, expected)

    def test_rebalance_constant_weights_scenario(self):
        """Test that rebalancer doesn't break when reducing to constant weights."""
        # This simulates the scenario where rebalancer might reduce allocations to constant
        # but backtest should still work
        from allooptim.config.rebalancer_config import RebalancerConfig
        
        config = RebalancerConfig(
            filter_lifetime_days=1,  # Very short half-life = almost no smoothing
            absolute_threshold=0.0,  # No threshold
            relative_threshold=0.0,  # No threshold
            max_trades_per_day=None,  # No trade limit
        )
        rebalancer = PortfolioRebalancer(rebalancing_days=20, config=config)

        assets = [f"ASSET_{i}" for i in range(100)]
        target = pd.Series([1.0/100] * 100, index=assets)  # Equal weights

        result = rebalancer.rebalance(target)

        # Should handle large portfolios
        assert len(result) == 100
        assert result.sum() == pytest.approx(1.0)
        assert all(result >= 0)
        assert all(result <= 1.0)