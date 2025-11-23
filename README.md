# AlloOptim Core

[![CI](https://github.com/AlloOptim/allooptim-core/workflows/CI/badge.svg)](https://github.com/AlloOptim/allooptim-core/actions)
[![Documentation Status](https://readthedocs.org/projects/allooptim/badge/?version=latest)](https://allooptim.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/allooptim.svg)](https://badge.fury.io/py/allooptim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## 🎯 What is AlloOptim?

AlloOptim is a comprehensive portfolio optimization library providing **35+ allocation strategies**, advanced covariance transformations, and ensemble methods. Built for institutional investors seeking scientific, reproducible, and transparent allocation decisions.

## ✨ Key Features

- **35+ Portfolio Optimizers**: From classic Markowitz to advanced ensemble methods
- **Advanced Risk Management**: Covariance transformations, shrinkage estimators, and robust statistics
- **Professional Reporting**: HTML tearsheets with QuantStats integration for institutional-grade analysis
- **Ensemble Methods**: Combine multiple strategies for improved out-of-sample performance
- **Backtesting Framework**: Comprehensive historical testing with performance metrics
- **Visualization Tools**: Interactive charts and performance analysis (matplotlib/seaborn)
- **Extensible Architecture**: Easy to add custom optimizers and transformations
- **Type-Safe**: Full mypy type checking for production reliability

## 📦 Installation

```bash
# Install from PyPI
pip install allooptim

# Or install from source
git clone https://github.com/AlloOptim/allooptim-core.git
cd allooptim-core
poetry install
```

## 🚀 Quick Start

```python
import pandas as pd
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import MaxSharpeOptimizer

# Your expected returns and covariance matrix
mu = pd.Series([0.12, 0.10, 0.11], index=['AAPL', 'GOOGL', 'MSFT'])
cov = pd.DataFrame([
    [0.04, 0.01, 0.015],
    [0.01, 0.05, 0.012],
    [0.015, 0.012, 0.03]
], index=['AAPL', 'GOOGL', 'MSFT'], columns=['AAPL', 'GOOGL', 'MSFT'])

# Create and run optimizer
optimizer = MaxSharpeOptimizer()
weights = optimizer.allocate(mu, cov)

print(weights)
# AAPL     0.35
# GOOGL    0.45
# MSFT     0.20
```

### Backtesting Multiple Strategies

```python
from allooptim.config.backtest_config import BacktestConfig
from allooptim.backtest.backtest_engine import BacktestEngine
from datetime import datetime

# Configure backtest
config = BacktestConfig(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1),
    rebalance_frequency=21,  # Monthly rebalancing
    optimizer_configs=[
        "MaxSharpeOptimizer",
        "HRPOptimizer",
        "RiskParityOptimizer"
    ]
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run_backtest()

# Results contain performance metrics, weights, and visualizations
```

### Allocation-to-Allocators (A2A)

```python
from allooptim.allocation_to_allocators.orchestrator_factory import create_orchestrator
from allooptim.optimizer.fundamental.fundamental_optimizer import BalancedFundamentalOptimizer

# Create ensemble of optimizers
orchestrator = create_orchestrator(
    optimizer_configs=["MaxSharpeOptimizer", "HRPOptimizer"],
    orchestration_type="equal_weight"
)

# Allocate across strategies
weights = orchestrator.allocate(mu, cov)
```

## 📚 Available Optimizers

AlloOptim provides 35+ optimizers across multiple categories:

### Efficient Frontier

- `MaxSharpeOptimizer` - Maximum Sharpe ratio portfolio
- `EfficientReturnOptimizer` - Target return with minimum risk
- `EfficientRiskOptimizer` - Target risk with maximum return

### Risk Parity & Diversification

- `RiskParityOptimizer` - Equal risk contribution
- `HRPOptimizer` - Hierarchical Risk Parity
- `DiversificationOptimizer` - Maximum diversification

### Robust Optimization

- `RobustMeanVarianceOptimizer` - Robust mean-variance optimization
- `MonteCarloMinVarianceOptimizer` - Monte Carlo minimum variance
- `BlackLittermanOptimizer` - Black-Litterman model

### Machine Learning

- `LightGBMOptimizer` - Gradient boosting optimization
- `LSTMOptimizer` - Long Short-Term Memory networks
- `MAMBAOptimizer` - Mamba architecture for time series

### Evolutionary Algorithms

- `MeanVarianceCMAOptimizer` - CMA-ES for mean-variance
- `MeanVarianceParticleSwarmOptimizer` - PSO optimization

### Fundamental Investing

- `BalancedFundamentalOptimizer` - Multi-factor fundamental
- `MarketCapFundamentalOptimizer` - Market cap weighted
- `QualityGrowthFundamentalOptimizer` - Quality and growth factors

See [optimizer documentation](docs/optimizers.md) for complete list and details.

## 🔧 Configuration System

AlloOptim uses Pydantic for type-safe configuration:

```python
from allooptim.config.backtest_config import BacktestConfig
from allooptim.config.a2a_config import A2AConfig

# Backtest configuration
backtest_config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 1, 1),
    rebalance_frequency=21,
    benchmark="SPY",
    quick_test=True  # For faster testing
)

# A2A configuration
a2a_config = A2AConfig(
    ensemble_method="equal_weight",
    risk_parity=True
)
```

## 📊 Reporting & Visualization

Generate professional reports with QuantStats integration:

```python
from allooptim.backtest.backtest_quantstats import create_quantstats_reports

# Generate HTML tearsheets
create_quantstats_reports(
    results=backtest_results,
    output_dir="./reports",
    benchmark="SPY"
)
```

## 🧪 Testing & Validation

```bash
# Run tests
poetry run pytest

# Run type checking
poetry run mypy allooptim

# Run linting
poetry run ruff check allooptim
```

## 📖 Documentation

📖 Full documentation available at

- [allooptim.readthedocs.io](https://allooptim.readthedocs.io)
- [Getting Started Notebooks](https://github.com/AlloOptim/allooptim-core/blob/main/examples/)
- [Methodology Whitepaper](https://github.com/AlloOptim/allooptim-whitepaper)

## 🤝 For Institutional Users

AlloOptim offers a professional SaaS platform built on this open-source core:

- Web-based UI with no coding required
- Integration with custodian banks
- Enhanced reporting
- Dedicated support

## 📖 Citation

If you use AlloOptim in your research:

```bibtex
@software{allooptim2025,
  author = {ten Haaf, Jonas},
  title = {AlloOptim: Open-Source Portfolio Optimization},
  year = {2025},
  url = {https://github.com/allooptim/allooptim-core}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙋 Contact

- Email: jonas.tenhaaf@mail.de
- LinkedIn: [Jonas ten Haaf](https://de.linkedin.com/in/jonas-ten-haaf-geb-weigand-9854b0198/en)

---

Built with ❤️ in Monheim, Germany

