# Examples Guide

This guide provides comprehensive examples for using AlloOptim across different use cases and scenarios.

## Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Backtesting Examples](#backtesting-examples)
- [Advanced Optimization](#advanced-optimization)
- [Machine Learning Integration](#machine-learning-integration)
- [Fundamental Investing](#fundamental-investing)
- [Risk Management](#risk-management)
- [Custom Extensions](#custom-extensions)

## Quick Start Examples

### 1. Basic Portfolio Optimization

```python
import pandas as pd
import numpy as np
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer

# Create sample data
np.random.seed(42)
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Sample expected returns (annualized)
mu = pd.Series([0.12, 0.10, 0.11, 0.13, 0.15], index=assets)

# Sample covariance matrix
cov = pd.DataFrame(
    [[0.04, 0.01, 0.015, 0.012, 0.02],
     [0.01, 0.05, 0.012, 0.008, 0.018],
     [0.015, 0.012, 0.03, 0.011, 0.016],
     [0.012, 0.008, 0.011, 0.06, 0.022],
     [0.02, 0.018, 0.016, 0.022, 0.08]],
    index=assets, columns=assets
)

# Optimize portfolio
optimizer = MaxSharpeOptimizer()
weights = optimizer.allocate(mu, cov)

print("Optimal Portfolio Weights:")
print(weights)
print(f"Expected Return: {mu.dot(weights):.3f}")
print(f"Expected Volatility: {np.sqrt(weights.dot(cov).dot(weights)):.3f}")
```

### 2. Comparing Multiple Strategies

```python
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import (
    MaxSharpeOptimizer, EfficientReturnOptimizer, EfficientRiskOptimizer
)
from allooptim.optimizer.hierarchical_risk_parity.hrp_optimizer import HRPOptimizer
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer

optimizers = [
    ("Max Sharpe", MaxSharpeOptimizer()),
    ("Efficient Return (10%)", EfficientReturnOptimizer(target_return=0.10)),
    ("Efficient Risk (15%)", EfficientRiskOptimizer(target_risk=0.15)),
    ("HRP", HRPOptimizer()),
    ("Risk Parity", RiskParityOptimizer())
]

results = {}
for name, optimizer in optimizers:
    weights = optimizer.allocate(mu, cov)
    expected_return = mu.dot(weights)
    expected_volatility = np.sqrt(weights.dot(cov).dot(weights))
    sharpe_ratio = expected_return / expected_volatility

    results[name] = {
        'weights': weights,
        'return': expected_return,
        'volatility': expected_volatility,
        'sharpe': sharpe_ratio
    }

# Display results
import pandas as pd
summary = pd.DataFrame({
    name: {
        'Return': data['return'],
        'Volatility': data['volatility'],
        'Sharpe': data['sharpe']
    }
    for name, data in results.items()
}).T

print("Strategy Comparison:")
print(summary.round(3))
```

### 3. Real Data with yfinance

```python
import yfinance as yf
from pypfopt import expected_returns, risk_models
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer

# Download real data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

# Calculate expected returns and covariance
mu = expected_returns.mean_historical_return(data)
cov = risk_models.sample_cov(data)

# Optimize
optimizer = MaxSharpeOptimizer()
weights = optimizer.allocate(mu, cov)

print("Real Data Optimization Results:")
print(weights.round(3))
print(f"\nExpected Annual Return: {mu.dot(weights):.1%}")
print(f"Expected Annual Volatility: {np.sqrt(weights.dot(cov).dot(weights)):.1%}")
```

## Backtesting Examples

### 1. Comprehensive Backtest

```python
from allooptim.config.backtest_config import BacktestConfig
from allooptim.backtest.backtest_engine import BacktestEngine
from allooptim.stock_universe import large_stock_universe, extract_symbols_from_list
from datetime import datetime

# Configure comprehensive backtest
config = BacktestConfig(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1),
    rebalance_frequency=21,  # Monthly rebalancing
    lookback_days=252,       # 1-year lookback
    symbols=extract_symbols_from_list(large_stock_universe())[:50],  # Top 50 stocks
    optimizer_configs=[
        "MaxSharpeOptimizer",
        "HRPOptimizer",
        "RiskParityOptimizer",
        "NaiveOptimizer"
    ],
    transformer_names=["OracleCovarianceTransformer"],
    benchmark="SPY",
    quick_test=False
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run_backtest()

# Analyze results
print(f"Backtest completed for {len(results)} optimizers")
print("\nPerformance Summary:")
for optimizer_name, data in results.items():
    metrics = data['metrics']
    print(f"{optimizer_name}:")
    print(f"  Total Return: {metrics['total_return']:.1%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
    print()
```

### 2. Rolling Window Analysis

```python
from allooptim.backtest.backtest_engine import BacktestEngine
import pandas as pd

def rolling_backtest_analysis(start_years, end_years, window_size=3):
    """Analyze performance across different market regimes."""

    results_summary = []

    for start_year in start_years:
        for end_year in end_years:
            if end_year - start_year < window_size:
                continue

            config = BacktestConfig(
                start_date=datetime(start_year, 1, 1),
                end_date=datetime(end_year, 12, 31),
                rebalance_frequency=63,  # Quarterly
                optimizer_configs=["MaxSharpeOptimizer", "HRPOptimizer"],
                symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],  # Smaller universe for speed
                quick_test=True
            )

            engine = BacktestEngine(config)
            results = engine.run_backtest()

            period_result = {
                'period': f"{start_year}-{end_year}",
                'maxsharpe_return': results['MaxSharpeOptimizer']['metrics']['total_return'],
                'hrp_return': results['HRPOptimizer']['metrics']['total_return']
            }
            results_summary.append(period_result)

    return pd.DataFrame(results_summary)

# Analyze different periods
start_years = [2019, 2020, 2021]
end_years = [2021, 2022, 2023, 2024]
rolling_results = rolling_backtest_analysis(start_years, end_years)

print("Rolling Window Analysis:")
print(rolling_results)
```

### 3. Transaction Cost Analysis

```python
def backtest_with_transaction_costs(config, transaction_cost_bps=10):
    """Run backtest with transaction cost penalties."""

    engine = BacktestEngine(config)
    results = engine.run_backtest()

    # Apply transaction cost adjustment (simplified)
    cost_factor = 1 - (transaction_cost_bps / 10000)  # Convert bps to decimal

    adjusted_results = {}
    for optimizer_name, data in results.items():
        original_return = data['metrics']['total_return']
        # Simplified: assume 50% annual turnover
        annual_turnover = 0.5
        annual_cost = annual_turnover * (transaction_cost_bps / 10000)
        adjusted_return = original_return * (1 - annual_cost)

        adjusted_metrics = data['metrics'].copy()
        adjusted_metrics['total_return'] = adjusted_return
        adjusted_metrics['annualized_return'] = adjusted_return ** (1/5) - 1  # Assuming 5-year period

        adjusted_results[optimizer_name] = {
            **data,
            'metrics': adjusted_metrics
        }

    return adjusted_results

# Run with and without costs
config = BacktestConfig(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1),
    optimizer_configs=["MaxSharpeOptimizer", "HRPOptimizer"],
    quick_test=True
)

results_no_cost = BacktestEngine(config).run_backtest()
results_with_cost = backtest_with_transaction_costs(config, transaction_cost_bps=20)

print("Impact of Transaction Costs (20bps):")
for optimizer in results_no_cost.keys():
    ret_no_cost = results_no_cost[optimizer]['metrics']['total_return']
    ret_with_cost = results_with_cost[optimizer]['metrics']['total_return']
    cost_impact = ret_no_cost - ret_with_cost
    print(f"{optimizer}: {cost_impact:.1%} reduction in total return")
```

## Advanced Optimization

### 1. Multi-Objective Optimization

```python
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import EfficientReturnOptimizer
import numpy as np

def efficient_frontier_points(mu, cov, n_points=20):
    """Generate points along the efficient frontier."""

    min_return = mu.min()
    max_return = mu.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    frontier_points = []
    for target_return in target_returns:
        try:
            optimizer = EfficientReturnOptimizer(target_return=target_return)
            weights = optimizer.allocate(mu, cov)

            expected_return = mu.dot(weights)
            expected_volatility = np.sqrt(weights.dot(cov).dot(weights))

            frontier_points.append({
                'return': expected_return,
                'volatility': expected_volatility,
                'weights': weights
            })
        except:
            continue  # Skip infeasible points

    return frontier_points

# Generate efficient frontier
frontier = efficient_frontier_points(mu, cov)

# Plot (if matplotlib available)
try:
    import matplotlib.pyplot as plt

    returns = [p['return'] for p in frontier]
    volatilities = [p['volatility'] for p in frontier]

    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c='blue', alpha=0.6)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True, alpha=0.3)
    plt.show()

except ImportError:
    print("Matplotlib not available for plotting")
```

### 2. Risk Parity with Constraints

```python
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer

class ConstrainedRiskParityOptimizer(RiskParityOptimizer):
    """Risk parity with additional constraints."""

    def __init__(self, max_weight=0.2, min_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        self.max_weight = max_weight
        self.min_weight = min_weight

    def allocate(self, mu, cov, **kwargs):
        weights = super().allocate(mu, cov, **kwargs)

        # Apply constraints
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)

        # Re-normalize
        weights = weights / weights.sum()

        return weights

# Use constrained risk parity
constrained_rp = ConstrainedRiskParityOptimizer(max_weight=0.15, min_weight=0.01)
weights = constrained_rp.allocate(mu, cov)

print("Constrained Risk Parity Weights:")
print(weights)
print(f"Max weight: {weights.max():.3f}")
print(f"Min weight: {weights.min():.3f}")
```

### 3. Black-Litterman Model

```python
from allooptim.optimizer.efficient_frontier.black_litterman_optimizer import BlackLittermanOptimizer

# Define market equilibrium
market_weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=assets)  # Example market cap weights

# Define investor views
views = {
    'AAPL': 0.15,   # Expect AAPL to return 15%
    'TSLA': 0.08,   # Expect TSLA to return only 8%
}

view_confidences = [0.7, 0.6]  # Confidence levels

# Run Black-Litterman optimization
bl_optimizer = BlackLittermanOptimizer(
    market_weights=market_weights,
    views=views,
    view_confidences=view_confidences,
    risk_aversion=2.5
)

weights = bl_optimizer.allocate(mu, cov)

print("Black-Litterman Portfolio:")
print(weights)
print(f"Expected Return: {mu.dot(weights):.3f}")
```

## Machine Learning Integration

### 1. LightGBM Portfolio Optimization

```python
from allooptim.optimizer.light_gbm.light_gbm_optimizer import LightGBMOptimizer
import numpy as np

# Generate synthetic historical data for training
np.random.seed(42)
n_periods = 1000
n_assets = 5

# Simulate historical returns
historical_returns = pd.DataFrame(
    np.random.normal(0.001, 0.02, (n_periods, n_assets)),
    columns=assets
)

# Create features (momentum, volatility, etc.)
features = pd.DataFrame(index=historical_returns.index)

for asset in assets:
    # 20-day momentum
    features[f'{asset}_momentum'] = historical_returns[asset].rolling(20).mean()

    # 20-day volatility
    features[f'{asset}_volatility'] = historical_returns[asset].rolling(20).std()

    # 60-day return
    features[f'{asset}_return_60d'] = historical_returns[asset].rolling(60).sum()

features = features.dropna()

# Target: future 20-day returns
target_returns = historical_returns.shift(-20).rolling(20).sum().dropna()
common_index = features.index.intersection(target_returns.index)
features = features.loc[common_index]
target_returns = target_returns.loc[common_index]

# Train LightGBM optimizer
lgb_optimizer = LightGBMOptimizer(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)

# Note: In practice, you'd split into train/validation sets
weights = lgb_optimizer.allocate(
    mu, cov,
    historical_data={
        'features': features,
        'target_returns': target_returns
    }
)

print("LightGBM Optimized Portfolio:")
print(weights)
```

### 2. LSTM-Based Optimization

```python
from allooptim.optimizer.deep_learning.deep_learning_optimizer import LSTMOptimizer

# Prepare time series data
sequence_length = 60  # 60-day sequences

# Create sequences
sequences = []
targets = []

for i in range(len(historical_returns) - sequence_length):
    seq = historical_returns.iloc[i:i+sequence_length].values
    target = historical_returns.iloc[i+sequence_length:i+sequence_length+20].sum().values
    sequences.append(seq)
    targets.append(target)

sequences = np.array(sequences)
targets = np.array(targets)

# Train LSTM optimizer
lstm_optimizer = LSTMOptimizer(
    sequence_length=sequence_length,
    hidden_size=64,
    num_layers=2,
    epochs=50,
    batch_size=32
)

weights = lstm_optimizer.allocate(
    mu, cov,
    historical_prices=historical_returns
)

print("LSTM Optimized Portfolio:")
print(weights)
```

## Fundamental Investing

### 1. Multi-Factor Fundamental Strategy

```python
from allooptim.optimizer.fundamental.fundamental_optimizer import BalancedFundamentalOptimizer
from allooptim.data.provider_factory import FundamentalDataProviderFactory
from datetime import datetime

# Create fundamental data provider
provider = FundamentalDataProviderFactory.create_provider()

# Create fundamental optimizer
fundamental_optimizer = BalancedFundamentalOptimizer(data_provider=provider)

# Get current fundamental data
current_time = datetime.now()
fundamental_data = provider.get_fundamental_data(assets, current_time)

print("Fundamental Data Available:")
for data in fundamental_data:
    if data.is_valid:
        print(f"{data.ticker}: ROE={data.roe:.1%}, D/E={data.debt_to_equity:.2f}")
    else:
        print(f"{data.ticker}: No fundamental data")

# Allocate based on fundamentals
weights = fundamental_optimizer.allocate(mu, cov, time=current_time)

print("\nFundamental-Based Allocation:")
print(weights)
```

### 2. Quality-Growth Strategy

```python
from allooptim.optimizer.fundamental.fundamental_optimizer import QualityGrowthFundamentalOptimizer

quality_growth_optimizer = QualityGrowthFundamentalOptimizer(data_provider=provider)

weights = quality_growth_optimizer.allocate(mu, cov, time=current_time)

print("Quality-Growth Portfolio:")
print(weights)
```

## Risk Management

### 1. CVaR Optimization

```python
from allooptim.optimizer.sequential_quadratic_programming.monte_carlo_robust_optimizer import MonteCarloMinCVAROptimizer

# Generate historical scenarios for CVaR calculation
n_scenarios = 10000
scenarios = pd.DataFrame(
    np.random.multivariate_normal(mu, cov, n_scenarios),
    columns=assets
)

cvar_optimizer = MonteCarloMinCVAROptimizer(
    n_simulations=n_scenarios,
    alpha=0.05  # 95% CVaR
)

weights = cvar_optimizer.allocate(mu, cov, scenarios=scenarios)

print("CVaR-Optimized Portfolio:")
print(weights)
print(f"Expected CVaR (95%): {np.percentile(scenarios.dot(weights), 5):.3f}")
```

### 2. Robust Optimization

```python
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import RobustMeanVarianceOptimizer

robust_optimizer = RobustMeanVarianceOptimizer(
    uncertainty_set_type="elliptical",
    uncertainty_parameter=0.2
)

weights = robust_optimizer.allocate(mu, cov)

print("Robust Optimized Portfolio:")
print(weights)
```

## Custom Extensions

### 1. Custom Optimizer

```python
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
import numpy as np

class MinimumVarianceOptimizer(AbstractOptimizer):
    """Custom minimum variance optimizer."""

    @property
    def name(self) -> str:
        return "MinimumVarianceOptimizer"

    @property
    def display_name(self) -> str:
        return "Minimum Variance"

    def allocate(self, mu, cov, **kwargs):
        """Minimum variance portfolio using quadratic optimization."""

        # Simple inverse volatility weighting as example
        volatilities = np.sqrt(np.diag(cov))
        weights = 1.0 / volatilities
        weights = weights / weights.sum()

        return pd.Series(weights, index=cov.index)

# Use custom optimizer
custom_optimizer = MinimumVarianceOptimizer()
weights = custom_optimizer.allocate(mu, cov)

print("Custom Minimum Variance Portfolio:")
print(weights)
```

### 2. Custom Covariance Transformer

```python
from allooptim.covariance_transformer.covariance_transformer import AbstractCovarianceTransformer
import numpy as np

class CustomShrinkageTransformer(AbstractCovarianceTransformer):
    """Custom shrinkage estimator."""

    def __init__(self, shrinkage_intensity=0.2):
        self.shrinkage_intensity = shrinkage_intensity

    def transform(self, cov):
        """Apply custom shrinkage to covariance matrix."""

        # Identity matrix target
        n = cov.shape[0]
        target = np.eye(n) * np.trace(cov) / n

        # Shrinkage
        shrunk_cov = (1 - self.shrinkage_intensity) * cov + self.shrinkage_intensity * target

        return shrunk_cov

# Use custom transformer
transformer = CustomShrinkageTransformer(shrinkage_intensity=0.1)
transformed_cov = transformer.transform(cov.values)

print("Original vs Transformed Covariance:")
print("Original trace:", np.trace(cov))
print("Transformed trace:", np.trace(transformed_cov))
```

### 3. Ensemble Strategy

```python
from allooptim.allocation_to_allocators.orchestrator_factory import create_orchestrator

# Create ensemble of optimizers
orchestrator = create_orchestrator(
    optimizer_configs=[
        "MaxSharpeOptimizer",
        "HRPOptimizer",
        "RiskParityOptimizer"
    ],
    orchestration_type="equal_weight",
    transformer_names=["OracleCovarianceTransformer"]
)

# Get ensemble allocation
ensemble_weights = orchestrator.allocate(mu, cov)

print("Ensemble Allocation:")
print(ensemble_weights)

# Compare with individual strategies
individual_weights = {}
for optimizer_name in ["MaxSharpeOptimizer", "HRPOptimizer", "RiskParityOptimizer"]:
    # Note: In practice, you'd instantiate each optimizer
    individual_weights[optimizer_name] = ensemble_weights  # Placeholder

print("\nEnsemble provides diversification across strategies")
```

## Performance Benchmarking

### 1. Speed Comparison

```python
import time

optimizers_to_test = [
    ("MaxSharpe", lambda: MaxSharpeOptimizer()),
    ("HRP", lambda: HRPOptimizer()),
    ("RiskParity", lambda: RiskParityOptimizer()),
]

timing_results = {}

for name, optimizer_factory in optimizers_to_test:
    optimizer = optimizer_factory()

    start_time = time.time()
    for _ in range(100):  # Multiple runs for stable timing
        weights = optimizer.allocate(mu, cov)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    timing_results[name] = avg_time

print("Average Optimization Time (seconds):")
for name, time_taken in timing_results.items():
    print(f"{name}: {time_taken:.4f}")
```

### 2. Scalability Test

```python
def test_scalability(max_assets=100):
    """Test performance with increasing number of assets."""

    scalability_results = []

    for n_assets in [10, 25, 50, 100]:
        if n_assets > max_assets:
            break

        # Generate synthetic data
        assets_subset = assets[:n_assets] if n_assets <= len(assets) else assets
        mu_subset = mu[:n_assets]
        cov_subset = cov.iloc[:n_assets, :n_assets]

        # Time optimization
        optimizer = MaxSharpeOptimizer()
        start_time = time.time()
        weights = optimizer.allocate(mu_subset, cov_subset)
        end_time = time.time()

        scalability_results.append({
            'n_assets': n_assets,
            'time': end_time - start_time,
            'convergence': weights is not None
        })

    return pd.DataFrame(scalability_results)

scalability_df = test_scalability()
print("Scalability Results:")
print(scalability_df)
```

This comprehensive examples guide demonstrates the versatility and power of AlloOptim across various use cases, from basic optimization to advanced machine learning integration and custom extensions.