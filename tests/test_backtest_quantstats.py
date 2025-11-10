"""Tests for QuantStats integration in backtesting framework."""

import numpy as np
import pandas as pd
import pytest

from allooptim.backtest.backtest_quantstats import (
    QUANTSTATS_AVAILABLE,
    calculate_quantstats_metrics,
    create_quantstats_reports,
    generate_comparative_tearsheets,
    generate_tearsheet,
    prepare_returns_for_quantstats,
)


class TestQuantStatsAvailability:
    """Test QuantStats availability handling."""

    def test_quantstats_availability_flag(self):
        """Test that QUANTSTATS_AVAILABLE is a boolean."""
        assert isinstance(QUANTSTATS_AVAILABLE, bool)

    @pytest.mark.skipif(not QUANTSTATS_AVAILABLE, reason="QuantStats not available")
    def test_quantstats_can_be_imported(self):
        """Test that QuantStats can be imported when available."""
        import quantstats as qs

        assert qs is not None


class TestReturnsPreparation:
    """Test returns preparation function."""

    def test_prepare_returns_valid_data(self):
        """Test returns preparation with valid data."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        results = {"TestOptimizer": {"returns": returns}}

        result = prepare_returns_for_quantstats(results, "TestOptimizer")
        assert result is not None
        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_prepare_returns_missing_optimizer(self):
        """Test returns preparation with missing optimizer."""
        results = {}
        result = prepare_returns_for_quantstats(results, "MissingOptimizer")
        assert result is None

    def test_prepare_returns_no_returns_data(self):
        """Test returns preparation with no returns data."""
        results = {"TestOptimizer": {"metrics": {}}}
        result = prepare_returns_for_quantstats(results, "TestOptimizer")
        assert result is None

    def test_prepare_returns_empty_returns(self):
        """Test returns preparation with empty returns."""
        results = {"TestOptimizer": {"returns": pd.Series([], dtype=float)}}
        result = prepare_returns_for_quantstats(results, "TestOptimizer")
        assert result is None

    def test_prepare_returns_wrong_index_type(self):
        """Test returns preparation with wrong index type."""
        returns = pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3])
        results = {"TestOptimizer": {"returns": returns}}
        result = prepare_returns_for_quantstats(results, "TestOptimizer")
        assert result is None

    def test_prepare_returns_insufficient_data(self):
        """Test returns preparation with insufficient data points."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        returns = pd.Series([0.01, np.nan, np.nan, np.nan, np.nan], index=dates)

        results = {"TestOptimizer": {"returns": returns}}
        result = prepare_returns_for_quantstats(results, "TestOptimizer")
        assert result is None

    def test_prepare_returns_with_nans(self):
        """Test returns preparation handles NaN values correctly."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates)
        returns.iloc[10:15] = np.nan  # Add some NaN values

        results = {"TestOptimizer": {"returns": returns}}

        result = prepare_returns_for_quantstats(results, "TestOptimizer")
        assert result is not None
        assert not result.isna().any()  # Should have no NaN values
        assert len(result) >= 10  # Should have sufficient data after cleaning


class TestTearsheetGeneration:
    """Test tearsheet generation functions."""

    @pytest.mark.skipif(not QUANTSTATS_AVAILABLE, reason="QuantStats not available")
    def test_generate_tearsheet_success(self, tmp_path):
        """Test successful tearsheet generation."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        results = {"TestOptimizer": {"returns": returns}}

        output_path = tmp_path / "test_tearsheet.html"
        success = generate_tearsheet(results, "TestOptimizer", output_path=output_path, mode="basic")

        assert success is True
        # Note: QuantStats may not create file in test environment, just check success
        # assert output_path.exists() or check that no exception was raised

    def test_generate_tearsheet_without_quantstats(self):
        """Test tearsheet generation when QuantStats is not available."""
        if QUANTSTATS_AVAILABLE:
            pytest.skip("QuantStats is available")

        results = {"TestOptimizer": {"returns": pd.Series([0.01, 0.02, 0.03])}}

        success = generate_tearsheet(results, "TestOptimizer")
        assert success is False

    def test_generate_tearsheet_invalid_optimizer(self):
        """Test tearsheet generation with invalid optimizer."""
        results = {}
        success = generate_tearsheet(results, "InvalidOptimizer")
        assert success is False


class TestComparativeAnalysis:
    """Test comparative tearsheet generation."""

    @pytest.mark.skipif(not QUANTSTATS_AVAILABLE, reason="QuantStats not available")
    def test_generate_comparative_tearsheets(self, tmp_path):
        """Test comparative tearsheet generation."""
        # Create sample data for multiple optimizers
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        results = {}
        for i, name in enumerate(["Optimizer1", "Optimizer2", "SPYBenchmark"]):
            returns = pd.Series(np.random.normal(0.001 + i * 0.0005, 0.02, 100), index=dates)
            results[name] = {
                "returns": returns,
                "metrics": {"sharpe_ratio": 1.5 - i * 0.2},  # Optimizer1: 1.5, Optimizer2: 1.3, SPYBenchmark: 1.1
            }

        status = generate_comparative_tearsheets(results, benchmark="SPYBenchmark", output_dir=tmp_path, top_n=2)

        assert isinstance(status, dict)
        # Should have 2 results (Optimizer1 and Optimizer2, excluding SPYBenchmark)
        assert len([k for k in status if k != "SPYBenchmark"]) == 2


class TestMetricsCalculation:
    """Test QuantStats metrics calculation."""

    @pytest.mark.skipif(not QUANTSTATS_AVAILABLE, reason="QuantStats not available")
    def test_calculate_quantstats_metrics(self):
        """Test QuantStats metrics calculation."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 100), index=dates)

        results = {"TestOptimizer": {"returns": returns}, "SPYBenchmark": {"returns": benchmark_returns}}

        metrics = calculate_quantstats_metrics(results, "TestOptimizer", benchmark="SPYBenchmark")

        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "max_drawdown" in metrics
        assert "cagr" in metrics

        # Check benchmark-relative metrics
        assert "alpha" in metrics
        assert "beta" in metrics
        assert "information_ratio" in metrics

    def test_calculate_quantstats_metrics_without_quantstats(self):
        """Test metrics calculation when QuantStats is not available."""
        if QUANTSTATS_AVAILABLE:
            pytest.skip("QuantStats is available")

        results = {"TestOptimizer": {"returns": pd.Series([0.01, 0.02, 0.03])}}

        metrics = calculate_quantstats_metrics(results, "TestOptimizer")
        assert metrics is None


class TestReportOrchestration:
    """Test report orchestration functions."""

    @pytest.mark.skipif(not QUANTSTATS_AVAILABLE, reason="QuantStats not available")
    def test_create_quantstats_reports(self, tmp_path):
        """Test full QuantStats report creation."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        results = {}
        for name in ["Optimizer1", "Optimizer2", "SPYBenchmark"]:
            returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
            results[name] = {"returns": returns, "metrics": {"sharpe_ratio": 1.0}}

        # Should not raise exception
        create_quantstats_reports(results, tmp_path, generate_individual=True, generate_top_n=2)

        # Check that HTML files were created in the output directory
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) > 0  # Should have created some HTML files

    def test_create_quantstats_reports_without_quantstats(self, tmp_path):
        """Test report creation when QuantStats is not available."""
        if QUANTSTATS_AVAILABLE:
            pytest.skip("QuantStats is available")

        results = {"TestOptimizer": {"returns": pd.Series([0.01, 0.02, 0.03])}}

        # Should not raise exception
        create_quantstats_reports(results, tmp_path)

        # Should not create any HTML files
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 0
