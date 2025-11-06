Optimizer Comparison
====================

Performance Characteristics
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 35

   * - Optimizer
     - Speed
     - Stability
     - Data Needs
     - Best For
   * - Mean-Variance
     - Fast
     - High
     - 60+ periods
     - Balanced portfolios
   * - HRP
     - Fast
     - Very High
     - 120+ periods
     - Diversification
   * - CMA-ES
     - Medium
     - Medium
     - 60+ periods
     - Non-convex objectives
   * - LightGBM
     - Fast
     - High
     - 250+ periods
     - Pattern recognition
   * - LSTM/MAMBA
     - Slow
     - Medium
     - 1000+ periods
     - Research, long horizon

When to Use Each Optimizer
---------------------------

Production/Daily Rebalancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mean-Variance Variants**
    Classical optimization with risk-return tradeoffs. Fast, stable, well-understood.
    Best for: Traditional portfolios, risk-managed strategies, regulatory compliance.

**Hierarchical Risk Parity (HRP)**
    Diversification-focused allocation using hierarchical clustering. Very stable
    with good out-of-sample performance. Best for: Risk parity, diversification,
    multi-asset portfolios.

**LightGBM**
    Machine learning approach using gradient boosting. Learns patterns from
    historical data. Best for: Adaptive strategies, pattern recognition,
    high-frequency rebalancing.

**Ensemble Methods**
    Combine multiple optimizers for improved robustness. Best for: Production
    systems requiring stability and performance.

Research/Monthly Rebalancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CMA-ES**
    Evolutionary optimization with covariance adaptation. Good for complex,
    non-convex objectives. Best for: Research, custom objectives, constraint optimization.

**Deep Learning (LSTM, MAMBA, TCN)**
    Neural network approaches for time series prediction. Require large datasets
    but can capture complex patterns. Best for: Academic research, long-term
    horizon strategies, novel approaches.

**Particle Swarm Optimization**
    Population-based metaheuristic. Good for multimodal optimization problems.
    Best for: Research, benchmarking, complex constraint sets.

Implementation Considerations
-----------------------------

Speed vs. Accuracy Tradeoffs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fast (< 1 second)**
    - Mean-Variance variants
    - Risk Parity
    - HRP
    - Simple shrinkage

**Medium (1-10 seconds)**
    - CMA-ES
    - Particle Swarm
    - LightGBM
    - Ensemble methods

**Slow (> 10 seconds)**
    - Deep learning models
    - Large ensemble combinations
    - Complex optimization problems

Data Requirements
~~~~~~~~~~~~~~~~~

**Minimum Data Needs**
    - 60 periods: Basic mean-variance
    - 120 periods: HRP, risk parity
    - 250 periods: ML-based methods
    - 1000+ periods: Deep learning

**Data Quality**
    - Clean price data required
    - Handle missing values gracefully
    - Outlier detection recommended
    - Stationarity assumptions

Stability Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

**High Stability**
    - HRP (very stable)
    - Mean-Variance (stable)
    - Risk Parity (stable)

**Medium Stability**
    - CMA-ES (can be sensitive to parameters)
    - LightGBM (depends on training data)
    - Ensemble methods (generally stable)

**Research Stage**
    - Deep learning models (emerging)
    - Novel ML approaches (experimental)

Best Practices
--------------

**Start Simple**
    Begin with mean-variance or HRP for baseline performance

**Validate Thoroughly**
    Use walk-forward validation, not simple backtesting

**Monitor Stability**
    Track allocation stability across rebalancing periods

**Combine Approaches**
    Use ensemble methods for production robustness

**Scale Gradually**
    Start with small portfolios, expand as confidence grows