# AlloOptim API Reference

This document provides comprehensive API documentation for the AlloOptim portfolio optimization library.

## Table of Contents

- [Core Interfaces](#core-interfaces)
- [Optimizers](#optimizers)
- [Covariance Transformers](#covariance-transformers)
- [Configuration](#configuration)
- [Backtesting](#backtesting)
- [Data Generation](#data-generation)
- [Utilities](#utilities)

## Core Interfaces

### AbstractOptimizer

Base class for all portfolio optimizers.

```python
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

class AbstractOptimizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the optimizer name."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the display name for reports."""
        pass

    @abstractmethod
    def allocate(self, mu: pd.Series, cov: pd.DataFrame, **kwargs) -> pd.Series:
        """Allocate portfolio weights.

        Args:
            mu: Expected returns series
            cov: Covariance matrix
            **kwargs: Additional optimizer-specific parameters

        Returns:
            Portfolio weights as pandas Series
        """
        pass
```

### AbstractCovarianceTransformer

Interface for covariance matrix transformations.

```python
from allooptim.covariance_transformer.covariance_transformer import AbstractCovarianceTransformer

class AbstractCovarianceTransformer(ABC):
    @abstractmethod
    def transform(self, cov: np.ndarray) -> np.ndarray:
        """Transform covariance matrix.

        Args:
            cov: Input covariance matrix

        Returns:
            Transformed covariance matrix
        """
        pass
```

## Optimizers

### Efficient Frontier Optimizers

#### MaxSharpeOptimizer

Maximizes the Sharpe ratio (return/risk).

```python
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer

optimizer = MaxSharpeOptimizer()
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `risk_free_rate`: Risk-free rate (default: 0.02)

#### EfficientReturnOptimizer

Minimizes risk for a target return.

```python
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import EfficientReturnOptimizer

optimizer = EfficientReturnOptimizer(target_return=0.10)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `target_return`: Target portfolio return (required)

#### EfficientRiskOptimizer

Maximizes return for a target risk level.

```python
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import EfficientRiskOptimizer

optimizer = EfficientRiskOptimizer(target_risk=0.15)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `target_risk`: Target portfolio volatility (required)

### Risk Parity Optimizers

#### RiskParityOptimizer

Equal risk contribution across assets.

```python
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer

optimizer = RiskParityOptimizer()
weights = optimizer.allocate(mu, cov)
```

#### HRPOptimizer

Hierarchical Risk Parity using hierarchical clustering.

```python
from allooptim.optimizer.hierarchical_risk_parity.hrp_optimizer import HRPOptimizer

optimizer = HRPOptimizer()
weights = optimizer.allocate(mu, cov)
```

### Robust Optimization

#### RobustMeanVarianceOptimizer

Robust mean-variance optimization using uncertainty sets.

```python
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import RobustMeanVarianceOptimizer

optimizer = RobustMeanVarianceOptimizer(
    uncertainty_set_type="box",
    uncertainty_parameter=0.1
)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `uncertainty_set_type`: Type of uncertainty set ("box", "elliptical")
- `uncertainty_parameter`: Size of uncertainty set

### Machine Learning Optimizers

#### LightGBMOptimizer

Uses gradient boosting to learn optimal portfolio weights.

```python
from allooptim.optimizer.light_gbm.light_gbm_optimizer import LightGBMOptimizer

optimizer = LightGBMOptimizer(
    n_estimators=100,
    learning_rate=0.1
)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `n_estimators`: Number of boosting rounds
- `learning_rate`: Learning rate
- `max_depth`: Maximum tree depth

#### LSTMOptimizer

Long Short-Term Memory network for portfolio optimization.

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
- `sequence_length`: Length of input sequences
- `hidden_size`: Hidden layer size
- `num_layers`: Number of LSTM layers

### Evolutionary Algorithms

#### MeanVarianceCMAOptimizer

Covariance Matrix Adaptation Evolution Strategy.

```python
from allooptim.optimizer.covariance_matrix_adaption.cma_optimizer import MeanVarianceCMAOptimizer

optimizer = MeanVarianceCMAOptimizer(
    population_size=50,
    max_iterations=100
)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `population_size`: CMA-ES population size
- `max_iterations`: Maximum iterations

#### MeanVarianceParticleSwarmOptimizer

Particle Swarm Optimization for portfolio allocation.

```python
from allooptim.optimizer.particle_swarm.pso_optimizer import MeanVarianceParticleSwarmOptimizer

optimizer = MeanVarianceParticleSwarmOptimizer(
    n_particles=100,
    max_iter=200
)
weights = optimizer.allocate(mu, cov)
```

**Parameters:**
- `n_particles`: Number of particles
- `max_iter`: Maximum iterations

### Fundamental Optimizers

#### BalancedFundamentalOptimizer

Multi-factor fundamental investing strategy.

```python
from allooptim.optimizer.fundamental.fundamental_optimizer import BalancedFundamentalOptimizer

optimizer = BalancedFundamentalOptimizer(data_provider=provider)
weights = optimizer.allocate(mu, cov, time=datetime.now())
```

**Parameters:**
- `data_provider`: Fundamental data provider instance

## Covariance Transformers

### Shrinkage Estimators

#### OracleCovarianceTransformer

Oracle shrinkage estimator (optimal theoretical shrinkage).

```python
from allooptim.covariance_transformer.covariance_transformer import OracleCovarianceTransformer

transformer = OracleCovarianceTransformer()
transformed_cov = transformer.transform(cov.values)
```

#### LedoitWolfCovarianceTransformer

Ledoit-Wolf shrinkage estimator.

```python
from allooptim.covariance_transformer.covariance_transformer import LedoitWolfCovarianceTransformer

transformer = LedoitWolfCovarianceTransformer()
transformed_cov = transformer.transform(cov.values)
```

### Denoising Methods

#### CovarianceAutoencoder

Neural network-based covariance matrix denoising.

```python
from allooptim.covariance_transformer.covariance_autoencoder import CovarianceAutoencoder

transformer = CovarianceAutoencoder(
    latent_dim=32,
    epochs=100
)
transformer.fit(training_data)
transformed_cov = transformer.transform(cov.values)
```

**Parameters:**
- `latent_dim`: Dimension of latent space
- `epochs`: Training epochs

## Configuration

### BacktestConfig

Configuration for backtesting scenarios.

```python
from allooptim.config.backtest_config import BacktestConfig
from datetime import datetime

config = BacktestConfig(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1),
    rebalance_frequency=21,  # Days
    lookback_days=252,       # 1 year
    benchmark="SPY",
    symbols=["AAPL", "MSFT", "GOOGL"],
    optimizer_configs=["MaxSharpeOptimizer", "HRPOptimizer"],
    transformer_names=["OracleCovarianceTransformer"],
    quick_test=False
)
```

**Key Parameters:**
- `start_date`, `end_date`: Backtest period
- `rebalance_frequency`: Rebalancing frequency in days
- `lookback_days`: Historical lookback window
- `benchmark`: Benchmark ticker symbol
- `optimizer_configs`: List of optimizer names
- `transformer_names`: List of covariance transformers

### A2AConfig

Configuration for Allocation-to-Allocators.

```python
from allooptim.config.a2a_config import A2AConfig

config = A2AConfig(
    ensemble_method="equal_weight",
    risk_parity=True,
    max_weight=0.3,
    min_weight=0.0
)
```

## Backtesting

### BacktestEngine

Main backtesting engine.

```python
from allooptim.backtest.backtest_engine import BacktestEngine

engine = BacktestEngine(backtest_config, a2a_config)
results = engine.run_backtest()
```

**Returns:**
Dictionary with optimizer results containing:
- `metrics`: Performance metrics (Sharpe, Sortino, max drawdown, etc.)
- `weights`: Time series of portfolio weights
- `returns`: Portfolio returns
- `benchmark_returns`: Benchmark returns

### Reporting Functions

#### create_quantstats_reports

Generate QuantStats HTML tearsheets.

```python
from allooptim.backtest.backtest_quantstats import create_quantstats_reports

create_quantstats_reports(
    results=results,
    output_dir="./reports",
    generate_individual=True,
    generate_top_n=5,
    benchmark="SPY"
)
```

#### generate_report

Create comprehensive markdown report.

```python
from allooptim.backtest.backtest_report import generate_report

report = generate_report(results, clustering_results, config)
with open("backtest_report.md", "w") as f:
    f.write(report)
```

## Data Generation

### Training Data Generation

#### generate_full_training_dataset

Generate comprehensive training dataset.

```python
from allooptim.data_generation.generate_30k_dataset import generate_full_training_dataset

dataset = generate_full_training_dataset(
    n_samples=30000,
    n_assets_range=(10, 50),
    save_path="./training_data.h5"
)
```

### Covariance Matrix Generation

#### DiverseCovarianceGenerator

Generate diverse covariance matrices for training.

```python
from allooptim.data_generation.diverse_covariance_generator import DiverseCovarianceGenerator

generator = DiverseCovarianceGenerator()
cov_matrices = generator.generate_covariance_matrices(
    n_matrices=1000,
    n_factors=5,
    n_assets=20
)
```

## Utilities

### Optimizer Management

#### get_all_optimizer_names

Get list of all available optimizer names.

```python
from allooptim.optimizer.optimizer_list import get_all_optimizer_names

names = get_all_optimizer_names()
print(names)  # ['MaxSharpeOptimizer', 'HRPOptimizer', ...]
```

#### get_optimizer_by_names

Get optimizer instances by name.

```python
from allooptim.optimizer.optimizer_list import get_optimizer_by_names

optimizers = get_optimizer_by_names(['MaxSharpeOptimizer', 'HRPOptimizer'])
```

### Allocation-to-Allocators

#### create_orchestrator

Create A2A orchestrator.

```python
from allooptim.allocation_to_allocators.orchestrator_factory import create_orchestrator

orchestrator = create_orchestrator(
    optimizer_configs=['MaxSharpeOptimizer', 'HRPOptimizer'],
    orchestration_type="equal_weight",
    transformer_names=["OracleCovarianceTransformer"]
)
```

### Fundamental Data

#### FundamentalDataProviderFactory

Factory for fundamental data providers.

```python
from allooptim.data.provider_factory import FundamentalDataProviderFactory

provider = FundamentalDataProviderFactory.create_provider()
fundamental_data = provider.get_fundamental_data(tickers, datetime.now())
```

## Error Handling

AlloOptim uses custom exceptions for better error handling:

- `OptimizationError`: Base optimization exception
- `InfeasibleOptimizationError`: Infeasible optimization problem
- `NumericalError`: Numerical computation errors
- `DataError`: Data validation errors

```python
from allooptim.optimizer.optimizer_interface import OptimizationError

try:
    weights = optimizer.allocate(mu, cov)
except OptimizationError as e:
    print(f"Optimization failed: {e}")
```

## Type Hints

AlloOptim is fully type-hinted for better IDE support and reliability:

```python
from typing import Optional
import pandas as pd
import numpy as np

def allocate_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    optimizer_name: str,
    risk_free_rate: Optional[float] = None
) -> pd.Series:
    # Function implementation
    pass
```

## Performance Considerations

- Use `quick_test=True` in BacktestConfig for faster testing
- Pre-compute covariance matrices when possible
- Use appropriate lookback windows for your strategy
- Consider parallel processing for large backtests

## Extending AlloOptim

### Custom Optimizer

```python
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

class CustomOptimizer(AbstractOptimizer):
    @property
    def name(self) -> str:
        return "CustomOptimizer"

    @property
    def display_name(self) -> str:
        return "Custom Strategy"

    def allocate(self, mu: pd.Series, cov: pd.DataFrame, **kwargs) -> pd.Series:
        # Your optimization logic here
        return weights
```

### Custom Covariance Transformer

```python
from allooptim.covariance_transformer.covariance_transformer import AbstractCovarianceTransformer

class CustomTransformer(AbstractCovarianceTransformer):
    def transform(self, cov: np.ndarray) -> np.ndarray:
        # Your transformation logic here
        return transformed_cov
```