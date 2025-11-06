Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install allooptim
   # or
   poetry add allooptim

Basic Usage
-----------

Simple Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from allooptim.optimizer import MeanVarianceOptimizer
   import pandas as pd

   # Load data
   prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
   returns = prices.pct_change().dropna()

   # Calculate inputs
   expected_returns = returns.mean()
   cov_matrix = returns.cov()

   # Optimize
   optimizer = MeanVarianceOptimizer(risk_aversion=2.0)
   weights = optimizer.allocate(expected_returns, cov_matrix)

   print(weights)

Running a Backtest
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from allooptim.backtest import BacktestEngine, BacktestConfig
   from allooptim.optimizer import get_all_optimizers

   config = BacktestConfig(
       start_date="2020-01-01",
       end_date="2023-12-31",
       rebalance_frequency="monthly",
       initial_capital=100_000
   )

   optimizers = get_all_optimizers()[:5]  # Test top 5
   engine = BacktestEngine(config)
   results = engine.run(prices, optimizers)

   print(results.summary())

Advanced Usage
--------------

Covariance Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from allooptim.covariance_transformer import SimpleShrinkageCovarianceTransformer

   # Improve covariance estimation
   transformer = SimpleShrinkageCovarianceTransformer(shrinkage=0.2)
   clean_cov = transformer.transform(cov_matrix, n_observations=252)

Ensemble Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from allooptim.allocation_to_allocators import EqualWeightOrchestrator
   from allooptim.optimizer import get_optimizer_by_names

   # Create ensemble of optimizers
   optimizer_names = ["MeanVariance", "HRPOptimizer", "RiskParityOptimizer"]
   optimizers = get_optimizer_by_names(optimizer_names)

   # Run ensemble allocation
   orchestrator = EqualWeightOrchestrator(optimizers, [])
   result = orchestrator.allocate(data_provider, time_today=datetime.now())

   print(f"Final allocation: {result.final_allocation}")
   print(f"Performance metrics: {result.performance_metrics}")