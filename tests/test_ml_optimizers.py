"""
ML Optimizers Testing

Test that LightGBM and Deep Learning optimizers work correctly after refactoring.
"""

from datetime import datetime
from allo_optim.optimizer.base_ml_optimizer import BaseMLOptimizer
import numpy as np
import pandas as pd
import pytest

from allo_optim.optimizer.light_gbm.light_gbm_optimizer import (
    LightGBMOptimizer,
    AugmentedLightGBMOptimizer,
)
from allo_optim.optimizer.deep_learning.deep_learning_optimizer import (
    LSTMOptimizer,
    MAMBAOptimizer,
    TCNOptimizer,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate sample price data with enough history for both optimizer types."""
    asset_names = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    # Use minimal data for fast testing: 35 periods (30 lookback + 5 for training)
    dates = pd.date_range(start="2023-01-01", periods=35, freq="D")
    
    np.random.seed(42)
    initial_prices = [150, 2800, 400, 800, 3200]
    price_data = {}
    
    for i, asset in enumerate(asset_names):
        prices = [initial_prices[i]]
        for _ in range(34):
            change = np.random.normal(0.001, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))
        price_data[asset] = prices
    
    return pd.DataFrame(price_data, index=dates)


@pytest.fixture
def sample_mu(sample_prices) -> pd.Series:
    """Generate expected returns."""
    return sample_prices.pct_change().mean()


@pytest.fixture
def sample_cov(sample_prices) -> pd.DataFrame:
    """Generate covariance matrix."""
    return sample_prices.pct_change().cov()


def test_lightgbm_optimizer_basic(sample_prices, sample_mu, sample_cov):
    """Test that LightGBMOptimizer can fit and allocate."""
    optimizer = LightGBMOptimizer()
    
    # Fit the optimizer
    optimizer.fit(sample_prices)
    
    # Allocate
    weights = optimizer.allocate(
        ds_mu=sample_mu,
        df_cov=sample_cov,
        df_prices=sample_prices
    )
    
    # Check weights are valid
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(sample_mu)
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=0.02)


def test_augmented_lightgbm_optimizer(sample_prices, sample_mu, sample_cov):
    """Test that AugmentedLightGBMOptimizer uses data augmentation."""
    optimizer = AugmentedLightGBMOptimizer()
    
    # Verify augmentation is enabled
    assert optimizer.config.use_data_augmentation is True
    
    # Fit and allocate
    optimizer.fit(sample_prices)
    weights = optimizer.allocate(
        ds_mu=sample_mu,
        df_cov=sample_cov,
        df_prices=sample_prices
    )
    
    # Check weights are valid
    assert isinstance(weights, pd.Series)
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=0.02)


@pytest.mark.skip(reason="DL training is slow; enable for full testing")
@pytest.mark.parametrize("optimizer_class", [LSTMOptimizer, MAMBAOptimizer, TCNOptimizer, MAMBAOptimizer])
def test_lstm_optimizer_basic(optimizer_class, sample_prices, sample_mu, sample_cov):
    """Test that DL optimizers can fit and allocate."""
    optimizer = optimizer_class()
    
    # Verify model type
    assert optimizer.model_type == "lstm"
    
    # Fit the optimizer (this will take some time)
    optimizer.fit(sample_prices)
    
    # Allocate
    weights = optimizer.allocate(
        ds_mu=sample_mu,
        df_cov=sample_cov,
        df_prices=sample_prices
    )
    
    # Check weights are valid
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(sample_mu)
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=0.02)


def test_mamba_optimizer_type(sample_prices):
    """Test that MAMBAOptimizer has correct model type."""
    optimizer = MAMBAOptimizer()
    assert optimizer.model_type == "mamba"
    assert optimizer.name == "MAMBAOptimizer"


def test_tcn_optimizer_type(sample_prices):
    """Test that TCNOptimizer has correct model type."""
    optimizer = TCNOptimizer()
    assert optimizer.model_type == "tcn"
    assert optimizer.name == "TCNOptimizer"


def test_lstm_optimizer_type(sample_prices):
    """Test that LSTMOptimizer has correct model type."""
    optimizer = LSTMOptimizer()
    assert optimizer.model_type == "lstm"
    assert optimizer.name == "LSTMOptimizer"


def test_optimizer_inheritance():
    """Test that all optimizers inherit from BaseMLOptimizer correctly."""

    
    # Check LightGBM optimizers
    lgbm = LightGBMOptimizer()
    assert isinstance(lgbm, BaseMLOptimizer)
    
    aug_lgbm = AugmentedLightGBMOptimizer()
    assert isinstance(aug_lgbm, BaseMLOptimizer)
    
    # Check Deep Learning optimizers
    lstm = LSTMOptimizer()
    assert isinstance(lstm, BaseMLOptimizer)
    
    mamba = MAMBAOptimizer()
    assert isinstance(mamba, BaseMLOptimizer)
    
    tcn = TCNOptimizer()
    assert isinstance(tcn, BaseMLOptimizer)
