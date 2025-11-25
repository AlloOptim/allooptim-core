"""Test comprehensive backtest execution for errors and warnings."""

import warnings
from datetime import datetime

import pytest

from allooptim.config.backtest_config import BacktestConfig
from allooptim.config.stock_universe import extract_symbols_from_list, large_stock_universe
from allooptim.optimizer.optimizer_config import OptimizerConfig
from examples.comprehensive_backtest import DEFAULT_OPTIMIZER_CONFIG, main_backtest


class TestComprehensiveBacktest:
    """Test that the comprehensive backtest executes without errors or warnings."""

    def test_default_optimizer_config_is_not_empty(self):
        """Test that the default optimizer configuration list is not empty."""
        assert len(DEFAULT_OPTIMIZER_CONFIG) > 0, "Default optimizer configuration should not be empty."

    def test_default_optimizer_config_contains_is_valid(self):
        """Test that the default optimizer configuration list contains valid entries.

        If string, it should be a valid optimizer name.
        If OptimizerConfig, it should pass pydantic validation.

        """
        for opt in DEFAULT_OPTIMIZER_CONFIG:
            assert isinstance(
                opt, (str, OptimizerConfig)
            ), "Each optimizer config should be either a string or an OptimizerConfig instance."

            if isinstance(opt, str):
                OptimizerConfig(name=opt)  # Should not raise errors

            elif isinstance(opt, OptimizerConfig):
                OptimizerConfig.model_validate(opt.model_dump())  # Should not raise errors

    def test_comprehensive_backtest_executes_without_errors(self):
        """Test that the comprehensive backtest main function executes without raising exceptions."""
        optimizer_configs = [
            OptimizerConfig(name="NaiveOptimizer"),
            OptimizerConfig(name="MomentumOptimizer"),
        ]

        symbols = extract_symbols_from_list(large_stock_universe())[:10]

        config_backtest = BacktestConfig(
            start_date=datetime(2015, 1, 1),
            end_date=datetime(2025, 1, 1),
            quick_start_date=datetime(2022, 12, 31),
            quick_end_date=datetime(2023, 2, 28),
            rebalance_frequency=10,
            lookback_days=90,
            quick_test=True,
            symbols=symbols,
            optimizer_configs=optimizer_configs,
        )

        with BacktestWarningCatcher():
            try:
                main_backtest(config_backtest=config_backtest)

            except Exception as e:
                pytest.fail(f"Comprehensive backtest failed to execute: {str(e)}")


class BacktestWarningCatcher:
    """Context manager that catches warnings and converts them to test failures."""

    def __init__(self):
        """Initialize the warning catcher."""
        self.warnings_caught = []

    def __enter__(self):
        """Enter the context manager and set up warning capture."""
        # Set up warning capture
        self.original_showwarning = warnings.showwarning
        warnings.showwarning = self._capture_warning
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and restore warning handler."""
        # Restore original warning handler
        warnings.showwarning = self.original_showwarning

        # Filter out system/infrastructure warnings that are not relevant
        relevant_warnings = self._filter_relevant_warnings()

        # If any relevant warnings were caught, fail the test
        if relevant_warnings:
            warning_messages = [str(w.message) for w in relevant_warnings]
            pytest.fail(f"Comprehensive backtest generated warnings: {'; '.join(warning_messages)}")

    def _filter_relevant_warnings(self):
        """Filter out system warnings that are not relevant to backtest execution."""
        relevant_warnings = []

        for warning in self.warnings_caught:
            # Skip zmq/tornado event loop warnings (infrastructure)
            if "Proactor event loop does not implement add_reader" in str(warning.message):
                continue
            # Skip asyncio selector warnings
            if "Registering an additional selector thread" in str(warning.message):
                continue
            # Skip matplotlib backend warnings
            if "matplotlib" in str(warning.filename).lower() and "backend" in str(warning.message).lower():
                continue
            # Skip seaborn warnings about deprecated parameters
            if "seaborn" in str(warning.filename).lower():
                continue
            # Skip quantstats plotting warnings
            if "quantstats" in str(warning.filename).lower():
                continue
            # Skip pandas future warnings
            if "pandas" in str(warning.filename).lower() and "FutureWarning" in str(warning.category.__name__):
                continue
            # Skip numpy warnings about array types
            if "numpy" in str(warning.filename).lower():
                continue
            # Skip scipy optimization warnings
            if "scipy" in str(warning.filename).lower():
                continue

            # Include all other warnings
            relevant_warnings.append(warning)

        return relevant_warnings

    def _capture_warning(self, message, category, filename, lineno, file=None, line=None):
        """Capture warnings for later processing."""
        warning = warnings.WarningMessage(
            message=message, category=category, filename=filename, lineno=lineno, file=file, line=line
        )
        self.warnings_caught.append(warning)

        # Still show the warning to stderr
        self.original_showwarning(message, category, filename, lineno, file, line)
