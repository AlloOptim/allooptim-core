"""Portfolio Backtesting Framework.

This module provides comprehensive backtesting capabilities for portfolio optimization
strategies with walk-forward validation, performance metrics, and comparative analysis.

Key Features:
    - **Walk-forward Validation**: Realistic out-of-sample testing
    - **Performance Metrics**: Sharpe ratio, max drawdown, volatility, alpha/beta
    - **Comparative Analysis**: Side-by-side optimizer performance comparison
    - **Risk Analysis**: VaR, CVaR, and other risk measures
    - **Visualization**: Charts and reports for strategy analysis

Quick Start:
    >>> from allooptim.backtest import BacktestEngine, BacktestConfig
    >>> config = BacktestConfig(start_date="2020-01-01", end_date="2023-12-31")
    >>> engine = BacktestEngine(config)
    >>> results = engine.run(price_data, optimizers)
    >>> print(results.summary())

See Also:
    - :class:`allooptim.backtest.backtest_config.BacktestConfig`: Configuration options
    - :class:`allooptim.backtest.backtest_engine.BacktestEngine`: Main backtesting engine
    - :mod:`allooptim.backtest.performance_metrics`: Performance calculation utilities
"""

from allooptim.backtest.backtest_config import BacktestConfig
from allooptim.backtest.backtest_engine import BacktestEngine

__all__ = ["BacktestConfig", "BacktestEngine"]
