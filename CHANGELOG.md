# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- QuantStats integration for professional HTML tearsheets and advanced performance analytics
- Comprehensive risk metrics (VaR, CVaR, Sortino, Calmar ratios)
- Benchmark-relative performance analysis (alpha, beta, information ratio)
- Interactive HTML reports with charts and statistics
- Optional dependency handling for QuantStats library

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
- Read the Docs integration at https://allooptim.readthedocs.io

### Infrastructure
- GitHub Actions CI/CD pipeline
- Automated publishing to PyPI via Trusted Publishers
- Sigstore attestations for supply chain security
- Ruff for linting and formatting
- Poetry for dependency management

[Unreleased]: https://github.com/AlloOptim/allooptim-core/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/AlloOptim/allooptim-core/releases/tag/v0.3.0
