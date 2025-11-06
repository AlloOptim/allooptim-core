# AlloOptim Core

[![CI](https://github.com/AlloOptim/allooptim-core/workflows/CI/badge.svg)](https://github.com/AlloOptim/allooptim-core/actions)
[![Documentation Status](https://readthedocs.org/projects/allooptim/badge/?version=latest)](https://allooptim.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/allo-optim.svg)](https://badge.fury.io/py/allo-optim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## 🎯 What is AlloOptim?

AlloOptim is an open-source platform for transparent **Allocation-to-Allocators** 
portfolio optimization. Built for institutional investors seeking scientific, 
reproducible, and transparent allocation decisions.

## 🚀 Quick Start
```python
from allooptim import Optimizer

# Your portfolio of asset managers
allocators = ['Manager A', 'Manager B', 'Manager C']
returns = load_returns_data(allocators)

# Optimize
optimizer = Optimizer(returns)
optimal_weights = optimizer.optimize_mean_variance(target_return=0.08)

print(optimal_weights)
# {'Manager A': 0.35, 'Manager B': 0.45, 'Manager C': 0.20}
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

**Built with ❤️ in Cologne, Germany**

