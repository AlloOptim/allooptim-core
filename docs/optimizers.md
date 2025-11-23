# Portfolio Optimizers

AlloOptim provides 35+ portfolio optimization strategies across multiple categories. This document provides detailed information about each optimizer, including methodology, parameters, and use cases.

## Table of Contents

- [Efficient Frontier](#efficient-frontier)
- [Risk Parity & Diversification](#risk-parity--diversification)
- [Robust Optimization](#robust-optimization)
- [Machine Learning](#machine-learning)
- [Evolutionary Algorithms](#evolutionary-algorithms)
- [Fundamental Investing](#fundamental-investing)
- [Naive Strategies](#naive-strategies)
- [Higher Moments](#higher-moments)
- [Sequential Quadratic Programming](#sequential-quadratic-programming)

## Efficient Frontier

### MaxSharpeOptimizer

**Category**: Efficient Frontier  
**Methodology**: Maximizes Sharpe ratio (expected return / volatility)  
**Use Case**: Classic mean-variance optimization for maximum risk-adjusted returns

```python
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer

optimizer = MaxSharpeOptimizer(risk_free_rate=0.02)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `risk_free_rate` (float): Risk-free rate for Sharpe calculation (default: 0.02)

**Mathematical Formulation:**
```
max  (μᵀw - r_f) / √(wᵀΣw)
s.t. wᵀ1 = 1, w ≥ 0
```

### EfficientReturnOptimizer

**Category**: Efficient Frontier  
**Methodology**: Minimizes volatility for target return  
**Use Case**: When you have a specific return target

```python
optimizer = EfficientReturnOptimizer(target_return=0.10)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `target_return` (float): Target portfolio return (required)

### EfficientRiskOptimizer

**Category**: Efficient Frontier  
**Methodology**: Maximizes return for target volatility  
**Use Case**: When you have a specific risk tolerance

```python
optimizer = EfficientRiskOptimizer(target_risk=0.15)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `target_risk` (float): Target portfolio volatility (required)

## Risk Parity & Diversification

### RiskParityOptimizer

**Category**: Risk Parity  
**Methodology**: Equal risk contribution from each asset  
**Use Case**: Balanced risk allocation across portfolio

```python
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer

optimizer = RiskParityOptimizer()
weights = optimizer.allocate(mu, cov)
```

**Mathematical Formulation:**
```
min  ∑(wᵢ × ∂σ/∂wᵢ - RC)²
s.t. wᵀ1 = 1, w ≥ 0
```
Where RC is the target risk contribution.

### HRPOptimizer

**Category**: Hierarchical Risk Parity  
**Methodology**: Hierarchical clustering + risk parity  
**Use Case**: Complex portfolios with many assets

```python
from allooptim.optimizer.hierarchical_risk_parity.hrp_optimizer import HRPOptimizer

optimizer = HRPOptimizer()
weights = optimizer.allocate(mu, cov)
```

**Algorithm:**
1. Compute correlation matrix
2. Perform hierarchical clustering
3. Allocate risk within clusters
4. Allocate risk between clusters

### DiversificationOptimizer

**Category**: Diversification  
**Methodology**: Maximum diversification ratio  
**Use Case**: Maximize spread of risk across assets

```python
from allooptim.optimizer.hvass_diversification.diversify_optimizer import DiversificationOptimizer

optimizer = DiversificationOptimizer()
weights = optimizer.allocate(mu, cov)
```

**Mathematical Formulation:**
```
max  (wᵀσ) / √(wᵀΣw)
s.t. wᵀ1 = 1, w ≥ 0
```

## Robust Optimization

### RobustMeanVarianceOptimizer

**Category**: Robust Optimization  
**Methodology**: Mean-variance with uncertainty sets  
**Use Case**: When expected returns are uncertain

```python
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import RobustMeanVarianceOptimizer

optimizer = RobustMeanVarianceOptimizer(
    uncertainty_set_type="box",
    uncertainty_parameter=0.1
)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `uncertainty_set_type`: "box" or "elliptical"
- `uncertainty_parameter`: Size of uncertainty set

### MonteCarloMinVarianceOptimizer

**Category**: Robust Optimization  
**Methodology**: Monte Carlo sampling for minimum variance  
**Use Case**: Robust estimation of minimum variance portfolio

```python
from allooptim.optimizer.sequential_quadratic_programming.monte_carlo_robust_optimizer import MonteCarloMinVarianceOptimizer

optimizer = MonteCarloMinVarianceOptimizer(n_simulations=1000)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `n_simulations`: Number of Monte Carlo simulations

### BlackLittermanOptimizer

**Category**: Bayesian Optimization  
**Methodology**: Black-Litterman model combining prior and views  
**Use Case**: Incorporating investment views with market equilibrium

```python
from allooptim.optimizer.efficient_frontier.black_litterman_optimizer import BlackLittermanOptimizer

optimizer = BlackLittermanOptimizer(
    views={'AAPL': 0.15, 'GOOGL': 0.10},
    view_confidences=[0.5, 0.6]
)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `views`: Dictionary of asset views (expected returns)
- `view_confidences`: Confidence levels for each view

## Machine Learning

### LightGBMOptimizer

**Category**: Machine Learning  
**Methodology**: Gradient boosting for portfolio weights  
**Use Case**: Learning optimal weights from historical data

```python
from allooptim.optimizer.light_gbm.light_gbm_optimizer import LightGBMOptimizer

optimizer = LightGBMOptimizer(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
weights = optimizer.allocate(mu, cov, historical_data=historical_data)
```

**Parameters:**
- `n_estimators`: Number of boosting rounds
- `learning_rate`: Learning rate
- `max_depth`: Maximum tree depth

### LSTMOptimizer

**Category**: Deep Learning  
**Methodology**: LSTM networks for time series prediction  
**Use Case**: Capturing temporal patterns in asset returns

```python
from allooptim.optimizer.deep_learning.deep_learning_optimizer import LSTMOptimizer

optimizer = LSTMOptimizer(
    sequence_length=60,
    hidden_size=64,
    num_layers=2
)
weights = optimizer.allocate(mu, cov, historical_prices=price_data)
```

**Parameters:**
- `sequence_length`: Input sequence length
- `hidden_size`: Hidden layer size
- `num_layers`: Number of LSTM layers

### MAMBAOptimizer

**Category**: Deep Learning  
**Methodology**: Mamba architecture for sequence modeling  
**Use Case**: Advanced time series analysis

```python
from allooptim.optimizer.deep_learning.deep_learning_optimizer import MAMBAOptimizer

optimizer = MAMBAOptimizer(
    d_model=64,
    n_layers=4
)
weights = optimizer.allocate(mu, cov, historical_prices=price_data)
```

**Parameters:**
- `d_model`: Model dimension
- `n_layers`: Number of layers

### TCNOptimizer

**Category**: Deep Learning  
**Methodology**: Temporal Convolutional Networks  
**Use Case**: Convolutional approach to time series

```python
from allooptim.optimizer.deep_learning.deep_learning_optimizer import TCNOptimizer

optimizer = TCNOptimizer(
    num_channels=[32, 64, 32],
    kernel_size=7
)
weights = optimizer.allocate(mu, cov, historical_prices=price_data)
```

**Parameters:**
- `num_channels`: Channel dimensions
- `kernel_size`: Convolution kernel size

## Evolutionary Algorithms

### CMA-ES Optimizers

**Category**: Evolutionary Algorithms  
**Methodology**: Covariance Matrix Adaptation Evolution Strategy

#### MeanVarianceCMAOptimizer
```python
from allooptim.optimizer.covariance_matrix_adaption.cma_optimizer import MeanVarianceCMAOptimizer

optimizer = MeanVarianceCMAOptimizer(
    population_size=50,
    max_iterations=100
)
weights = optimizer.allocate(mu, cov)
```

#### LMomentsCMAOptimizer
Uses L-moments for robustness

#### SortinoCMAOptimizer
Optimizes Sortino ratio

#### MaxDrawdownCMAOptimizer
Minimizes maximum drawdown

#### RobustSharpeCMAOptimizer
Robust Sharpe ratio optimization

#### CVARCMAOptimizer
Optimizes CVaR (Conditional Value at Risk)

### Particle Swarm Optimizers

**Category**: Evolutionary Algorithms  
**Methodology**: Particle Swarm Optimization

#### MeanVarianceParticleSwarmOptimizer
```python
from allooptim.optimizer.particle_swarm.pso_optimizer import MeanVarianceParticleSwarmOptimizer

optimizer = MeanVarianceParticleSwarmOptimizer(
    n_particles=100,
    max_iter=200
)
weights = optimizer.allocate(mu, cov)
```

#### LMomentsParticleSwarmOptimizer
Uses L-moments for robustness

## Fundamental Investing

### BalancedFundamentalOptimizer

**Category**: Fundamental Investing  
**Methodology**: Multi-factor fundamental analysis  
**Use Case**: Long-term fundamental investing

```python
from allooptim.optimizer.fundamental.fundamental_optimizer import BalancedFundamentalOptimizer

optimizer = BalancedFundamentalOptimizer(data_provider=provider)
weights = optimizer.allocate(mu, cov, time=datetime.now())
```

**Factors:** ROE, Debt-to-Equity, Current Ratio, etc.

### MarketCapFundamentalOptimizer

**Category**: Fundamental Investing  
**Methodology**: Market capitalization weighting  
**Use Case**: Size-based fundamental strategy

### QualityGrowthFundamentalOptimizer

**Category**: Fundamental Investing  
**Methodology**: Quality and growth factors  
**Use Case**: Quality-growth investing

### ValueInvestingFundamentalOptimizer

**Category**: Fundamental Investing  
**Methodology**: Value investing principles  
**Use Case**: Contrarian value strategy

## Naive Strategies

### NaiveOptimizer

**Category**: Naive  
**Methodology**: Equal weight allocation  
**Use Case**: Benchmark for comparison

```python
from allooptim.optimizer.naive.naive_optimizer import NaiveOptimizer

optimizer = NaiveOptimizer()
weights = optimizer.allocate(mu, cov)
```

### MomentumOptimizer

**Category**: Momentum  
**Methodology**: Momentum-based weighting  
**Use Case**: Trend-following strategy

```python
from allooptim.optimizer.naive.momentum_optimizer import MomentumOptimizer

optimizer = MomentumOptimizer(lookback_period=252)
weights = optimizer.allocate(mu, cov, historical_returns=returns)
```

**Parameters:**
- `lookback_period`: Momentum calculation period

### EMAMomentumOptimizer

**Category**: Momentum  
**Methodology**: Exponential moving average momentum

## Higher Moments

### HigherMomentOptimizer

**Category**: Higher Moments  
**Methodology**: Mean-variance-skewness-kurtosis optimization  
**Use Case**: Accounting for non-normal return distributions

```python
from allooptim.optimizer.sequential_quadratic_programming.higher_moments_optimizer import HigherMomentOptimizer

optimizer = HigherMomentOptimizer(
    target_skewness=0.0,
    target_kurtosis=3.0
)
weights = optimizer.allocate(mu, cov, historical_returns=returns)
```

**Parameters:**
- `target_skewness`: Target portfolio skewness
- `target_kurtosis`: Target portfolio kurtosis

## Sequential Quadratic Programming

### NCOSharpeOptimizer

**Category**: Nested Clustering Optimization  
**Methodology**: Hierarchical clustering + Sharpe optimization  
**Use Case**: Combining clustering with efficient frontier

```python
from allooptim.optimizer.nested_cluster.nco_optimizer import NCOSharpeOptimizer

optimizer = NCOSharpeOptimizer()
weights = optimizer.allocate(mu, cov)
```

### Adjusted Returns Optimizers

#### MeanVarianceAdjustedReturnsOptimizer
Adjusts returns based on mean-variance analysis

#### EMAAdjustedReturnsOptimizer
Uses exponential moving averages

#### LMomentsAdjustedReturnsOptimizer
Uses L-moments for robustness

#### SemiVarianceAdjustedReturnsOptimizer
Focuses on downside risk

### Monte Carlo Optimizers

#### MonteCarloMaxDiversificationOptimizer
Monte Carlo diversification maximization

#### MonteCarloMaxDrawdownOptimizer
Monte Carlo maximum drawdown minimization

#### MonteCarloMaxSortinoOptimizer
Monte Carlo Sortino ratio maximization

#### MonteCarloMinCVAROptimizer
Monte Carlo CVaR minimization

### WikipediaOptimizer

**Category**: Academic  
**Methodology**: Wikipedia-based optimization example  
**Use Case**: Educational purposes

## Performance Comparison

| Category | Best For | Speed | Robustness | Complexity |
|----------|----------|-------|------------|------------|
| Efficient Frontier | Risk-adjusted returns | Fast | Low | Low |
| Risk Parity | Risk balancing | Medium | High | Medium |
| ML-based | Pattern recognition | Slow | High | High |
| Evolutionary | Complex constraints | Slow | Very High | High |
| Fundamental | Long-term investing | Medium | Medium | Medium |

## Choosing an Optimizer

1. **For beginners**: Start with `MaxSharpeOptimizer` or `HRPOptimizer`
2. **For robustness**: Use `RiskParityOptimizer` or robust variants
3. **For large portfolios**: Consider `HRPOptimizer` or `NCOSharpeOptimizer`
4. **For machine learning**: Try `LightGBMOptimizer` or `LSTMOptimizer`
5. **For fundamental investing**: Use `BalancedFundamentalOptimizer`

## Custom Optimizers

To create a custom optimizer, inherit from `AbstractOptimizer`:

```python
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

class CustomOptimizer(AbstractOptimizer):
    @property
    def name(self) -> str:
        return "CustomOptimizer"

    @property
    def display_name(self) -> str:
        return "My Custom Strategy"

    def allocate(self, mu: pd.Series, cov: pd.DataFrame, **kwargs) -> pd.Series:
        # Your optimization logic here
        return weights
```