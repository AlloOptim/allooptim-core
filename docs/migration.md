# Migration Guide

This guide helps users migrate from previous versions of AlloOptim or similar portfolio optimization libraries.

## From PyPortfolioOpt

If you're migrating from PyPortfolioOpt, here are the key changes:

### Installation

```bash
# Old
pip install PyPortfolioOpt

# New
pip install allooptim
```

### Basic Usage

```python
# PyPortfolioOpt
from pypfopt import EfficientFrontier
ef = EfficientFrontier(mu, cov)
weights = ef.max_sharpe()

# AlloOptim
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer
optimizer = MaxSharpeOptimizer()
weights = optimizer.allocate(mu, cov)
```

### Key Differences

1. **API Consistency**: All AlloOptim optimizers use the same `allocate(mu, cov)` interface
2. **Type Safety**: Full mypy type checking
3. **Extensibility**: Easy to add custom optimizers
4. **Backtesting**: Built-in comprehensive backtesting framework

### Optimizer Mapping

| PyPortfolioOpt | AlloOptim |
|----------------|-----------|
| EfficientFrontier.max_sharpe() | MaxSharpeOptimizer |
| EfficientFrontier.min_volatility() | EfficientRiskOptimizer(target_risk=...) |
| EfficientFrontier.efficient_return() | EfficientReturnOptimizer(target_return=...) |
| HRPOpt | HRPOptimizer |
| riskparity | RiskParityOptimizer |

## From Riskfolio-Lib

### Installation

```bash
# Old
pip install riskfolio-lib

# New
pip install allooptim
```

### Basic Usage

```python
# Riskfolio-Lib
import riskfolio as rp
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu='hist', method_cov='hist')
weights = port.optimization(model='Classic', rm='MV', obj='Sharpe')

# AlloOptim
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer
optimizer = MaxSharpeOptimizer()
weights = optimizer.allocate(mu, cov)
```

### Advanced Features

```python
# Riskfolio-Lib higher moments
port = rp.Portfolio(returns=returns)
weights = port.optimization(model='FM', rm='MVSK', obj='Sharpe')

# AlloOptim higher moments
from allooptim.optimizer.sequential_quadratic_programming.higher_moments_optimizer import HigherMomentOptimizer
optimizer = HigherMomentOptimizer()
weights = optimizer.allocate(mu, cov, historical_returns=returns)
```

## From Previous AlloOptim Versions

### Breaking Changes in v0.3.x

#### 1. Optimizer Instantiation

```python
# Old (v0.2.x)
from allooptim.optimizer_list import get_optimizer
optimizer = get_optimizer("MaxSharpe")

# New (v0.3.x)
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer
optimizer = MaxSharpeOptimizer()
```

#### 2. Configuration Classes

```python
# Old
config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 1, 1),
    optimizers=["MaxSharpe", "HRP"]
)

# New
config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 1, 1),
    optimizer_configs=["MaxSharpeOptimizer", "HRPOptimizer"]
)
```

#### 3. Backtest Results

```python
# Old
results = backtest_engine.run()
metrics = results.metrics

# New
results = backtest_engine.run_backtest()
metrics = results["MaxSharpeOptimizer"]["metrics"]
```

### Migration Script

```python
def migrate_old_config(old_config):
    """Migrate old BacktestConfig to new format."""

    # Map old optimizer names to new classes
    optimizer_mapping = {
        "MaxSharpe": "MaxSharpeOptimizer",
        "HRP": "HRPOptimizer",
        "RiskParity": "RiskParityOptimizer",
        "Naive": "NaiveOptimizer",
        # Add more mappings as needed
    }

    new_optimizer_configs = [
        optimizer_mapping.get(opt, opt)
        for opt in old_config.get("optimizers", [])
    ]

    new_config = BacktestConfig(
        start_date=old_config["start_date"],
        end_date=old_config["end_date"],
        rebalance_frequency=old_config.get("rebalance_frequency", 21),
        optimizer_configs=new_optimizer_configs,
        # Map other fields...
    )

    return new_config
```

## From CVXPY/Risk Parity Libraries

### Basic Risk Parity

```python
# CVXPY approach
import cvxpy as cp

w = cp.Variable(n)
risk_contributions = # complex CVXPY formulation
prob = cp.Problem(cp.Minimize(cp.sum_squares(risk_contributions - target)), [cp.sum(w) == 1, w >= 0])
prob.solve()

# AlloOptim
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer
optimizer = RiskParityOptimizer()
weights = optimizer.allocate(mu, cov)
```

## From Machine Learning Libraries

### From scikit-learn

```python
# scikit-learn regression approach
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# AlloOptim ML optimizer
from allooptim.optimizer.light_gbm.light_gbm_optimizer import LightGBMOptimizer
optimizer = LightGBMOptimizer()
weights = optimizer.allocate(mu, cov, historical_data=historical_data)
```

## Configuration Migration

### Backtest Configuration

```python
# Old format (if using JSON/YAML)
{
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "optimizers": ["MaxSharpe", "HRP", "RiskParity"],
    "rebalance_days": 21
}

# New format
from allooptim.config.backtest_config import BacktestConfig
from datetime import datetime

config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 1, 1),
    optimizer_configs=["MaxSharpeOptimizer", "HRPOptimizer", "RiskParityOptimizer"],
    rebalance_frequency=21
)
```

### A2A Configuration

```python
# Old
a2a_config = {
    "method": "equal_weight",
    "risk_parity": True
}

# New
from allooptim.config.a2a_config import A2AConfig

a2a_config = A2AConfig(
    ensemble_method="equal_weight",
    risk_parity=True
)
```

## Code Migration Examples

### 1. Simple Optimization

```python
# Before
from pypfopt import EfficientFrontier, expected_returns, risk_models
import yfinance as yf

tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")["Adj Close"]
mu = expected_returns.mean_historical_return(data)
cov = risk_models.sample_cov(data)

ef = EfficientFrontier(mu, cov)
weights = ef.max_sharpe()
ef.portfolio_performance()

# After
import yfinance as yf
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer

tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")["Adj Close"]

# Calculate expected returns and covariance (using pypfopt or similar)
from pypfopt import expected_returns, risk_models
mu = expected_returns.mean_historical_return(data)
cov = risk_models.sample_cov(data)

optimizer = MaxSharpeOptimizer()
weights = optimizer.allocate(mu, cov)

# Performance analysis (using quantstats)
import quantstats as qs
returns = data.pct_change().dropna()
portfolio_returns = returns.dot(weights)
qs.reports.full(portfolio_returns)
```

### 2. Backtesting Migration

```python
# Before (hypothetical old API)
from allooptim.backtest import BacktestEngine

engine = BacktestEngine()
engine.set_period("2020-01-01", "2024-01-01")
engine.add_optimizer("MaxSharpe")
engine.add_optimizer("HRP")
results = engine.run()

# After
from allooptim.config.backtest_config import BacktestConfig
from allooptim.backtest.backtest_engine import BacktestEngine
from datetime import datetime

config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 1, 1),
    optimizer_configs=["MaxSharpeOptimizer", "HRPOptimizer"]
)

engine = BacktestEngine(config)
results = engine.run_backtest()
```

### 3. Custom Optimizer Migration

```python
# Before
from allooptim.optimizer.base import BaseOptimizer

class CustomOptimizer(BaseOptimizer):
    def optimize(self, mu, cov):
        # Custom logic
        return weights

# After
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

class CustomOptimizer(AbstractOptimizer):
    @property
    def name(self) -> str:
        return "CustomOptimizer"

    @property
    def display_name(self) -> str:
        return "My Custom Strategy"

    def allocate(self, mu, cov, **kwargs):
        # Custom logic
        return weights
```

## Data Format Changes

### Expected Returns

```python
# All versions support pandas Series
import pandas as pd

mu = pd.Series([0.12, 0.10, 0.11], index=['AAPL', 'MSFT', 'GOOGL'])
```

### Covariance Matrix

```python
# All versions support pandas DataFrame
cov = pd.DataFrame([
    [0.04, 0.01, 0.015],
    [0.01, 0.05, 0.012],
    [0.015, 0.012, 0.03]
], index=['AAPL', 'MSFT', 'GOOGL'], columns=['AAPL', 'MSFT', 'GOOGL'])
```

## Testing Migration

### Unit Tests

```python
# Before
def test_max_sharpe():
    optimizer = get_optimizer("MaxSharpe")
    weights = optimizer.optimize(mu, cov)
    assert len(weights) == n_assets

# After
def test_max_sharpe():
    from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer
    optimizer = MaxSharpeOptimizer()
    weights = optimizer.allocate(mu, cov)
    assert len(weights) == n_assets
```

## Performance Considerations

### Speed Improvements

- **v0.3.x**: Improved numerical stability and performance
- **Type checking**: Optional mypy validation for better code quality
- **Memory usage**: More efficient data structures

### Accuracy Improvements

- Better handling of edge cases
- Improved covariance matrix conditioning
- Enhanced numerical stability in optimization

## Getting Help

If you encounter issues during migration:

1. Check the [API documentation](api.md)
2. Review the [examples](../examples/)
3. Open an issue on GitHub
4. Check the [changelog](../CHANGELOG.md) for breaking changes

## Validation

After migration, validate your results:

```python
def validate_migration(old_weights, new_weights):
    """Validate that migration preserved optimization behavior."""

    # Check weight sums
    assert abs(old_weights.sum() - 1.0) < 1e-6
    assert abs(new_weights.sum() - 1.0) < 1e-6

    # Check weight bounds (if applicable)
    assert all(new_weights >= 0)  # Assuming long-only

    # Check reasonable weight distribution
    assert new_weights.std() > 0  # Not all equal weights

    print("Migration validation passed!")

# Usage
old_weights = old_optimizer.optimize(mu, cov)
new_weights = new_optimizer.allocate(mu, cov)
validate_migration(old_weights, new_weights)
```