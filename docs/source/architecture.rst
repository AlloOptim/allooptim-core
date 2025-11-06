Architecture Overview
=====================

AlloOptim follows a modular architecture with clear separation of concerns:

.. code-block:: text

   ┌─────────────────────────────────────────────────┐
   │           User Interface / Scripts              │
   └────────────────┬────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────┐
   │        Allocation-to-Allocators (A2A)          │
   │  ┌──────────────────────────────────────────┐  │
   │  │     Orchestrators (Equal/Optimized)      │  │
   │  └────────┬─────────────────────┬───────────┘  │
   └───────────┼─────────────────────┼──────────────┘
               │                     │
   ┌───────────▼──────────┐  ┌──────▼──────────────┐
   │   Optimizer Layer    │  │  Covariance Layer   │
   │  ┌────────────────┐  │  │  ┌──────────────┐   │
   │  │ Mean-Variance  │  │  │  │  Shrinkage   │   │
   │  │ CMA-ES         │  │  │  │  Denoising   │   │
   │  │ HRP            │  │  │  │  Autoencoder │   │
   │  │ ML-based       │  │  │  │              │   │
   │  └────────────────┘  │  │  └──────────────┘   │
   └──────────────────────┘  └─────────────────────┘

Core Components
---------------

Optimizers (``allooptim.optimizer``)
    35+ optimization algorithms implementing AbstractOptimizer interface

Covariance Transformers (``allooptim.covariance_transformer``)
    Statistical methods to improve covariance estimation

Backtest Engine (``allooptim.backtest``)
    Walk-forward validation with performance metrics

A2A Framework (``allooptim.allocation_to_allocators``)
    Meta-allocation layer for ensemble strategies

Data Flow
---------

1. **Data Input**: Price data, market data, or simulated scenarios
2. **Preprocessing**: Returns calculation, covariance estimation
3. **Covariance Enhancement**: Shrinkage, denoising, regularization
4. **Optimization**: Single or ensemble allocation strategies
5. **Validation**: Backtesting with performance metrics
6. **Reporting**: Results analysis and visualization

Key Design Principles
----------------------

**Modularity**
    Each component can be used independently or combined with others

**Extensibility**
    New optimizers and transformers can be added without modifying existing code

**Type Safety**
    Full type hints and Pydantic validation throughout

**Performance**
    Efficient implementations with optional parallel execution

**Reproducibility**
    Deterministic results with configurable random seeds

Component Details
-----------------

AbstractOptimizer
~~~~~~~~~~~~~~~~~

Base class for all portfolio optimization algorithms:

- **Interface**: ``allocate(ds_mu, df_cov, **kwargs) -> pd.Series``
- **Features**: Asset name preservation, consistent return format
- **Extensions**: Support for time-series data, L-moments, custom constraints

AbstractCovarianceTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for covariance matrix transformations:

- **Interface**: ``transform(df_cov, n_observations) -> pd.DataFrame``
- **Features**: Asset name preservation, statistical improvements
- **Methods**: Shrinkage, denoising, regularization, dimensionality reduction

A2AOrchestrator
~~~~~~~~~~~~~~~

Meta-allocation framework for ensemble strategies:

- **Purpose**: Combine multiple optimizers intelligently
- **Strategies**: Equal weight, optimized allocation, ML-based selection
- **Features**: Performance tracking, risk management, adaptive allocation

BacktestEngine
~~~~~~~~~~~~~~

Walk-forward validation system:

- **Methodology**: Realistic out-of-sample testing
- **Metrics**: Sharpe ratio, max drawdown, alpha/beta, volatility
- **Features**: Transaction costs, rebalancing schedules, comparative analysis