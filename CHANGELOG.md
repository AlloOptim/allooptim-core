# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.5] - 2025-11-25

### Refactored

- Standardized all optimizer name properties to end with "Optimizer" and match class names for consistency
- Simplified optimizer factory interfaces by removing unnecessary fundamental_data_provider parameters
- Improved code discoverability and maintainability through naming standardization

### Fixed

- Corrected optimizer name properties across all optimizer classes (efficient_frontier, momentum, PSO, ensemble)
- Updated test files and examples to use standardized optimizer names
- Fixed docstring formatting issues in test files

## [0.3.1] - 2025-11-11

### Added

- Type-safe dataclass-based return values for backtest results (`BacktestResultMetrics`, `BacktestResults`)
- Improved code structure and maintainability in backtest engine

### Fixed

- QuantStats "'NoneType' object has no attribute 'upper'" error during tearsheet generation
- Defensive benchmark naming to prevent None benchmark titles in QuantStats reports
- Updated test suite to match refactored API signatures
- Removed obsolete test references and function calls

### Changed

- `run_backtest()` method now returns structured dataclasses instead of dictionaries
- Enhanced QuantStats integration with explicit benchmark title handling

## [0.3.0] - 2025-11-06

### Added

- Initial public release of AlloOptim
- 35+ portfolio optimization strategies including:
  - Efficient Frontier optimizers (Mean-Variance, Black-Litterman)
  - Hierarchical Risk Parity (HRP)
  - Nested Clustered Optimization (NCO)
  - Deep Learning optimizers
  - LightGBM-based optimizers
  - Particle Swarm Optimization
  - CMA-ES optimizer
  - Kelly Criterion
  - Risk Parity variants
  - Fundamental analysis-based optimizers
  - Wikipedia pageview-based allocation
- Advanced covariance transformations:
  - Marchenko-Pastur denoising
  - PCA-based denoising
  - Ledoit-Wolf shrinkage
  - Oracle Approximating Shrinkage (OAS)
  - Detoning (market factor removal)
  - Multiple shrinkage methods
- Ensemble optimization methods
- Comprehensive backtesting framework with:
  - Performance metrics calculation
  - Visualization tools
  - Cluster analysis
  - Report generation
- Data generation utilities for synthetic portfolios
- Stock universe configuration system
- Complete documentation with Sphinx/ReadTheDocs
- Full test suite with pytest

### Documentation

- Complete API documentation
- Architecture overview
- Optimizer comparison guide
- Quickstart guide
- Read the Docs integration at <https://allooptim.readthedocs.io>

### Infrastructure

- GitHub Actions CI/CD pipeline
- Automated publishing to PyPI via Trusted Publishers
- Sigstore attestations for supply chain security
- Ruff for linting and formatting
- Poetry for dependency management

[Unreleased]: https://github.com/AlloOptim/allooptim-core/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/AlloOptim/allooptim-core/releases/tag/v0.3.1
[0.3.0]: https://github.com/AlloOptim/allooptim-core/releases/tag/v0.3.0
