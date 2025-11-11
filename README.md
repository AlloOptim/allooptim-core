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

## � Installation

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
from pypfopt import expected_returns, risk_models
from allooptim.optimizer.optimizer_list import get_optimizer

# Load your price data (or use yfinance)
prices = pd.DataFrame({
    'AAPL': [100, 102, 105, 103, 107],
    'GOOGL': [200, 198, 202, 205, 210],
    'MSFT': [150, 152, 155, 153, 158]
}, index=pd.date_range('2024-01-01', periods=5))

# Calculate expected returns and covariance
mu = expected_returns.mean_historical_return(prices)
cov = risk_models.sample_cov(prices)

# Choose an optimizer (35+ strategies available)
optimizer = get_optimizer("MaxSharpeOptimizer")

# Get optimal portfolio weights
weights = optimizer.allocate(mu, cov)
print(weights)
# AAPL     0.35
# GOOGL    0.45
# MSFT     0.20
```

### Available Optimizers

```python
from allooptim.optimizer.optimizer_list import get_all_optimizers

# See all 35+ optimizers
optimizers = get_all_optimizers()
for name in optimizers:
    print(name)
# MaxSharpe, HRP, BlackLitterman, NCOSharpe, RiskParity, ...
```

## 📚 Documentation

📖 Full documentation available at [allooptim.readthedocs.io](https://allooptim.readthedocs.io)

- [Getting Started Guide](docs/getting_started.md)
- [Methodology Whitepaper](https://allooptim.io/whitepaper.pdf)
- [API Reference](docs/api.md)

## 🤝 For Institutional Users

AlloOptim offers a professional SaaS platform built on this open-source core:
- Web-based UI with no coding required
- Integration with custodian banks
- Compliance-ready reporting
- Dedicated support

→ **Learn more:** [allooptim.com](https://allooptim.com)

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

- Website: [allooptim.com](https://allooptim.com)
- Email: jonas.tenhaaf@mail.de
- LinkedIn: [Jonas ten Haaf](https://de.linkedin.com/in/jonas-ten-haaf-geb-weigand-9854b0198/en)

---

**Built with ❤️ in Monheim, Germany**

