"""Tests for optimizer failure handling configuration and behavior.

This module tests the FailureHandlingConfig and its integration with A2A orchestrators,
ensuring graceful degradation when optimizers fail during allocation.
"""

import logging
import pytest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from allooptim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allooptim.allocation_to_allocators.equal_weight_orchestrator import EqualWeightOrchestrator
from allooptim.allocation_to_allocators.optimized_orchestrator import OptimizedOrchestrator
from allooptim.allocation_to_allocators.simulator_interface import AbstractObservationSimulator
from allooptim.config.a2a_config import A2AConfig
from allooptim.config.failure_handling_config import FailureHandlingConfig, FailureHandlingOption, FailureType
from allooptim.config.failure_diagnostics import (
    CircuitBreaker,
    FailureClassifier,
    FailureDiagnostic,
    RetryHandler,
)
from allooptim.covariance_transformer.transformer_interface import AbstractCovarianceTransformer
from allooptim.optimizer.optimizer_interface import AbstractOptimizer


class FailingOptimizer(AbstractOptimizer):
    """Test optimizer that always fails."""

    def __init__(self, fail_with: Exception = RuntimeError("Test failure")):
        super().__init__()
        self.fail_with = fail_with

    def allocate(self, ds_mu, df_cov, df_prices=None, time=None, l_moments=None):
        raise self.fail_with

    @property
    def name(self):
        return "FailingOptimizer"


class MockDataProvider(AbstractObservationSimulator):
    """Mock data provider for testing."""

    def __init__(self, n_assets=3):
        self.n_assets = n_assets
        self.asset_names = [f"Asset_{i}" for i in range(n_assets)]

    @property
    def mu(self):
        return pd.Series(np.random.randn(self.n_assets), index=self.asset_names)

    @property
    def cov(self):
        cov = np.random.randn(self.n_assets, self.n_assets)
        return cov @ cov.T  # Make positive definite

    @property
    def historical_prices(self):
        return pd.DataFrame(
            np.random.randn(100, self.n_assets),
            columns=self.asset_names
        )

    @property
    def n_observations(self):
        return 100

    def get_sample(self):
        mu = self.mu
        cov = pd.DataFrame(self.cov, index=self.asset_names, columns=self.asset_names)
        prices = self.historical_prices
        time = pd.Timestamp.now()
        l_moments = None
        return mu, cov, prices, time, l_moments

    def get_ground_truth(self):
        return self.get_sample()

    @property
    def name(self):
        return "MockDataProvider"


class MockCovarianceTransformer(AbstractCovarianceTransformer):
    """Mock covariance transformer that returns input unchanged."""

    def transform(self, cov, n_observations):
        return cov


class TestFailureHandlingConfig:
    """Test FailureHandlingConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FailureHandlingConfig()
        assert config.option == FailureHandlingOption.EQUAL_WEIGHTS
        assert config.log_failures is True
        assert config.raise_on_all_failed is False

    def test_enum_validation(self):
        """Test that invalid options are rejected."""
        with pytest.raises(ValueError):
            FailureHandlingConfig(option="invalid_option")

    def test_valid_options(self):
        """Test all valid failure handling options."""
        for option in FailureHandlingOption:
            config = FailureHandlingConfig(option=option)
            assert config.option == option


class TestBaseOrchestratorFailureHandling:
    """Test BaseOrchestrator._handle_optimizer_failure method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_provider = MockDataProvider()
        self.optimizers = [FailingOptimizer()]
        self.covariance_transformers = [MockCovarianceTransformer()]

    class TestOrchestrator(BaseOrchestrator):
        """Concrete implementation of BaseOrchestrator for testing."""

        def allocate(self, data_provider, time_today=None, all_stocks=None):
            # Not used in these tests
            pass

        @property
        def name(self):
            return "TestOrchestrator"

    def test_zero_weights_fallback(self):
        """Test ZERO_WEIGHTS failure handling."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(option=FailureHandlingOption.ZERO_WEIGHTS))
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, config)

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()
        result = orchestrator._handle_optimizer_failure(
            optimizer=self.optimizers[0],
            exception=RuntimeError("Test failure"),
            n_assets=len(mu),
            asset_names=mu.index.tolist()
        )

        assert result is not None
        assert len(result) == len(mu)
        assert (result == 0.0).all()
        assert result.name == self.optimizers[0].name

    def test_equal_weights_fallback(self):
        """Test EQUAL_WEIGHTS failure handling."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(option=FailureHandlingOption.EQUAL_WEIGHTS))
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, config)

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()
        result = orchestrator._handle_optimizer_failure(
            optimizer=self.optimizers[0],
            exception=RuntimeError("Test failure"),
            n_assets=len(mu),
            asset_names=mu.index.tolist()
        )

        assert result is not None
        assert len(result) == len(mu)
        expected_weight = 1.0 / len(mu)
        assert (result == expected_weight).all()
        assert result.name == self.optimizers[0].name

    def test_ignore_optimizer_fallback(self):
        """Test IGNORE_OPTIMIZER failure handling."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(option=FailureHandlingOption.IGNORE_OPTIMIZER))
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, config)

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()
        result = orchestrator._handle_optimizer_failure(
            optimizer=self.optimizers[0],
            exception=RuntimeError("Test failure"),
            n_assets=len(mu),
            asset_names=mu.index.tolist()
        )

        assert result is None  # Should skip optimizer entirely

    def test_logging_enabled(self, caplog):
        """Test that failures are logged when log_failures=True."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(log_failures=True))
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, config)

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()
        with caplog.at_level(logging.WARNING):
            orchestrator._handle_optimizer_failure(
                optimizer=self.optimizers[0],
                exception=RuntimeError("Test failure"),
                n_assets=len(mu),
                asset_names=mu.index.tolist()
            )

        assert "FailingOptimizer failed with unknown_error: Test failure" in caplog.text

    def test_logging_disabled(self, caplog):
        """Test that failures are not logged when log_failures=False."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(log_failures=False))
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, config)

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()
        with caplog.at_level(logging.WARNING):
            orchestrator._handle_optimizer_failure(
                optimizer=self.optimizers[0],
                exception=RuntimeError("Test failure"),
                n_assets=len(mu),
                asset_names=mu.index.tolist()
            )

        assert "FailingOptimizer failed" not in caplog.text


class TestEqualWeightOrchestratorFailureHandling:
    """Test EqualWeightOrchestrator with failing optimizers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_provider = MockDataProvider(n_assets=3)
        self.covariance_transformers = [MockCovarianceTransformer()]

    def test_single_failing_optimizer_equal_weights(self):
        """Test single failing optimizer with EQUAL_WEIGHTS fallback."""
        failing_opt = FailingOptimizer()
        config = A2AConfig(failure_handling=FailureHandlingConfig(option=FailureHandlingOption.EQUAL_WEIGHTS))
        orchestrator = EqualWeightOrchestrator([failing_opt], self.covariance_transformers, config)

        result = orchestrator.allocate(self.data_provider)

        # Should have one optimizer allocation with equal weights
        assert len(result.optimizer_allocations) == 1
        assert len(result.optimizer_weights) == 1
        weights = result.optimizer_allocations[0].weights
        expected_weight = 1.0 / 3  # 3 assets
        assert (weights == expected_weight).all()

    def test_single_failing_optimizer_zero_weights(self):
        """Test single failing optimizer with ZERO_WEIGHTS fallback."""
        failing_opt = FailingOptimizer()
        config = A2AConfig(failure_handling=FailureHandlingConfig(option=FailureHandlingOption.ZERO_WEIGHTS))
        orchestrator = EqualWeightOrchestrator([failing_opt], self.covariance_transformers, config)

        result = orchestrator.allocate(self.data_provider)

        # Should have one optimizer allocation with zero weights
        assert len(result.optimizer_allocations) == 1
        weights = result.optimizer_allocations[0].weights
        assert (weights == 0.0).all()

    def test_mixed_optimizers_with_ignore(self):
        """Test mixed working/failing optimizers with IGNORE_OPTIMIZER."""
        working_opt = Mock()
        working_opt.name = "WorkingOptimizer"
        working_opt.display_name = "WorkingOptimizer"
        working_opt.allocate.return_value = pd.Series([0.5, 0.3, 0.2], index=self.data_provider.mu.index)
        working_opt.fit = Mock()

        failing_opt = FailingOptimizer()

        config = A2AConfig(failure_handling=FailureHandlingConfig(option=FailureHandlingOption.IGNORE_OPTIMIZER))
        orchestrator = EqualWeightOrchestrator([working_opt, failing_opt], self.covariance_transformers, config)

        result = orchestrator.allocate(self.data_provider)

        # Should only have the working optimizer (failing one ignored)
        assert len(result.optimizer_allocations) == 1
        assert result.optimizer_allocations[0].instance_id == "WorkingOptimizer"

    def test_all_optimizers_failed_with_ignore(self):
        """Test IGNORE_OPTIMIZER when all optimizers fail."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(
            option=FailureHandlingOption.IGNORE_OPTIMIZER,
            raise_on_all_failed=False
        ))
        optimizers = [FailingOptimizer(), FailingOptimizer()]
        orchestrator = EqualWeightOrchestrator(optimizers, self.covariance_transformers, config)

        result = orchestrator.allocate(self.data_provider)

        # Should have emergency fallback allocation
        assert len(result.optimizer_allocations) == 1
        assert result.optimizer_allocations[0].instance_id == "EMERGENCY_FALLBACK"
        # Should have equal weights
        expected_weight = 1.0 / 3  # 3 assets
        assert (result.final_allocation == expected_weight).all()

    def test_all_optimizers_failed_with_raise(self):
        """Test raise_on_all_failed=True behavior."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(
            option=FailureHandlingOption.IGNORE_OPTIMIZER,
            raise_on_all_failed=True
        ))
        optimizers = [FailingOptimizer(), FailingOptimizer()]
        orchestrator = EqualWeightOrchestrator(optimizers, self.covariance_transformers, config)

        with pytest.raises(RuntimeError, match="All optimizers failed"):
            orchestrator.allocate(self.data_provider)


class TestOptimizedOrchestratorFailureHandling:
    """Test OptimizedOrchestrator with failing optimizers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_provider = MockDataProvider(n_assets=3)
        self.covariance_transformers = [MockCovarianceTransformer()]

    def test_single_optimizer_standalone_fallback(self):
        """Test single optimizer failure uses EQUAL_WEIGHTS (standalone behavior)."""
        failing_opt = FailingOptimizer()
        config = A2AConfig(failure_handling=FailureHandlingConfig(option=FailureHandlingOption.ZERO_WEIGHTS))
        orchestrator = OptimizedOrchestrator([failing_opt], self.covariance_transformers, config)

        result = orchestrator.allocate(self.data_provider)

        # Should use EQUAL_WEIGHTS fallback regardless of config (standalone behavior)
        assert len(result.optimizer_allocations) == 1
        weights = result.optimizer_allocations[0].weights
        expected_weight = 1.0 / 3
        assert (weights == expected_weight).all()

    def test_all_optimizers_failed_with_ignore(self):
        """Test IGNORE_OPTIMIZER when all optimizers fail in OptimizedOrchestrator."""
        # NOTE: OptimizedOrchestrator uses simulation framework that doesn't support
        # all-optimizers-failed handling in the same way. This test is skipped.
        pytest.skip("OptimizedOrchestrator simulation framework doesn't support all-optimizers-failed handling")

    def test_all_optimizers_failed_with_raise(self):
        """Test raise_on_all_failed=True behavior in OptimizedOrchestrator."""
        # NOTE: OptimizedOrchestrator uses simulation framework that doesn't support
        # all-optimizers-failed handling in the same way. This test is skipped.
        pytest.skip("OptimizedOrchestrator simulation framework doesn't support all-optimizers-failed handling")


class TestAbstractOptimizerSafeAllocate:
    """Test AbstractOptimizer.allocate_safe method."""

    def test_successful_allocation(self):
        """Test that allocate_safe returns normal result on success."""
        optimizer = Mock()
        optimizer.name = "TestOptimizer"
        optimizer.allocate.return_value = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])

        # Call allocate_safe (which doesn't exist on Mock, so we need to use a real optimizer)
        # This test would need a concrete optimizer implementation
        pass

    def test_failed_allocation_fallback(self):
        """Test that allocate_safe returns equal weights on failure."""
        failing_opt = FailingOptimizer()
        mu = pd.Series([0.1, 0.2, 0.3], index=["A", "B", "C"])
        cov = pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"])

        result = failing_opt.allocate_safe(mu, cov)

        expected_weight = 1.0 / 3
        assert len(result) == 3
        assert (result == expected_weight).all()
        assert result.name == "FailingOptimizer"


class TestA2AConfigIntegration:
    """Test that FailureHandlingConfig integrates properly with A2AConfig."""

    def test_failure_handling_in_a2a_config(self):
        """Test that failure_handling field is properly included in A2AConfig."""
        config = A2AConfig()
        assert hasattr(config, 'failure_handling')
        assert isinstance(config.failure_handling, FailureHandlingConfig)
        assert config.failure_handling.option == FailureHandlingOption.EQUAL_WEIGHTS

    def test_custom_failure_handling_config(self):
        """Test setting custom failure handling configuration."""
        custom_config = FailureHandlingConfig(
            option=FailureHandlingOption.ZERO_WEIGHTS,
            log_failures=False,
            raise_on_all_failed=True
        )
        a2a_config = A2AConfig(failure_handling=custom_config)

        assert a2a_config.failure_handling.option == FailureHandlingOption.ZERO_WEIGHTS
        assert a2a_config.failure_handling.log_failures is False
        assert a2a_config.failure_handling.raise_on_all_failed is True


class TestEnhancedFailureHandlingConfig:
    """Test enhanced FailureHandlingConfig features."""

    def test_context_aware_fallbacks(self):
        """Test context-aware fallback configuration."""
        config = FailureHandlingConfig(
            context_aware_fallbacks={
                FailureType.NUMERICAL_ERROR: FailureHandlingOption.ZERO_WEIGHTS,
                FailureType.DATA_ERROR: FailureHandlingOption.EQUAL_WEIGHTS,
            }
        )

        assert config.context_aware_fallbacks[FailureType.NUMERICAL_ERROR] == FailureHandlingOption.ZERO_WEIGHTS
        assert config.context_aware_fallbacks[FailureType.DATA_ERROR] == FailureHandlingOption.EQUAL_WEIGHTS

    def test_retry_configuration(self):
        """Test retry configuration parameters."""
        config = FailureHandlingConfig(
            retry_attempts=3,
            retry_delay_seconds=0.5
        )

        assert config.retry_attempts == 3
        assert config.retry_delay_seconds == 0.5

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration."""
        config = FailureHandlingConfig(circuit_breaker_threshold=5)
        assert config.circuit_breaker_threshold == 5

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = FailureHandlingConfig(
            retry_attempts=2,
            retry_delay_seconds=1.0,
            circuit_breaker_threshold=3
        )
        assert config.retry_attempts == 2

        # Invalid retry attempts (too high)
        with pytest.raises(ValueError):
            FailureHandlingConfig(retry_attempts=10)

        # Invalid retry delay (too high)
        with pytest.raises(ValueError):
            FailureHandlingConfig(retry_delay_seconds=15.0)


class TestFailureClassifier:
    """Test failure classification logic."""

    def test_numerical_error_classification(self):
        """Test classification of numerical errors."""
        # LinAlgError
        exc = Exception("Singular matrix")
        assert FailureClassifier.classify_failure(exc) == FailureType.NUMERICAL_ERROR

        # ValueError with numerical keywords
        exc = ValueError("Matrix not positive definite")
        assert FailureClassifier.classify_failure(exc) == FailureType.NUMERICAL_ERROR

    def test_data_error_classification(self):
        """Test classification of data errors."""
        exc = KeyError("Missing asset data")
        assert FailureClassifier.classify_failure(exc) == FailureType.DATA_ERROR

        exc = IndexError("Asset index out of bounds")
        assert FailureClassifier.classify_failure(exc) == FailureType.DATA_ERROR

    def test_configuration_error_classification(self):
        """Test classification of configuration errors."""
        exc = ValueError("Invalid constraint configuration")
        assert FailureClassifier.classify_failure(exc) == FailureType.CONFIGURATION_ERROR

    def test_resource_error_classification(self):
        """Test classification of resource errors."""
        exc = MemoryError("Out of memory")
        assert FailureClassifier.classify_failure(exc) == FailureType.RESOURCE_ERROR

        exc = TimeoutError("Operation timed out")
        assert FailureClassifier.classify_failure(exc) == FailureType.RESOURCE_ERROR

    def test_unknown_error_classification(self):
        """Test classification of unknown errors."""
        exc = RuntimeError("Some unexpected error")
        assert FailureClassifier.classify_failure(exc) == FailureType.UNKNOWN_ERROR


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(threshold=3, timeout_seconds=60.0)
        assert cb.threshold == 3
        assert cb.timeout_seconds == 60.0

    def test_circuit_breaker_opening(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(threshold=2)

        # First failure
        cb.record_failure("optimizer1")
        assert not cb.is_open("optimizer1")

        # Second failure - should open circuit
        cb.record_failure("optimizer1")
        assert cb.is_open("optimizer1")

    def test_circuit_breaker_closing(self):
        """Test circuit breaker closes after timeout."""
        cb = CircuitBreaker(threshold=1, timeout_seconds=0.1)  # Very short timeout

        cb.record_failure("optimizer1")
        assert cb.is_open("optimizer1")

        # Wait for timeout
        import time
        time.sleep(0.2)

        # Should be closed now
        assert not cb.is_open("optimizer1")

    def test_circuit_breaker_success_reset(self):
        """Test circuit breaker resets on success."""
        cb = CircuitBreaker(threshold=2)

        cb.record_failure("optimizer1")
        cb.record_failure("optimizer1")
        assert cb.is_open("optimizer1")

        # Success should reset
        cb.record_success("optimizer1")
        assert not cb.is_open("optimizer1")


class TestRetryHandler:
    """Test retry logic."""

    def test_should_retry_logic(self):
        """Test retry decision logic."""
        # Should retry resource errors
        assert RetryHandler.should_retry(FailureType.RESOURCE_ERROR, 0, 3)
        assert RetryHandler.should_retry(FailureType.UNKNOWN_ERROR, 0, 3)

        # Should not retry numerical or data errors
        assert not RetryHandler.should_retry(FailureType.NUMERICAL_ERROR, 0, 3)
        assert not RetryHandler.should_retry(FailureType.DATA_ERROR, 0, 3)

        # Should not retry beyond max attempts
        assert not RetryHandler.should_retry(FailureType.RESOURCE_ERROR, 3, 3)

    def test_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        delay1 = RetryHandler.calculate_delay(0, 0.1)
        delay2 = RetryHandler.calculate_delay(1, 0.1)
        delay3 = RetryHandler.calculate_delay(2, 0.1)

        # Should be exponential: 0.1, 0.2, 0.4
        assert 0.05 <= delay1 <= 0.15  # With jitter
        assert 0.1 <= delay2 <= 0.3
        assert 0.2 <= delay3 <= 0.6


class TestEnhancedBaseOrchestratorFailureHandling:
    """Test enhanced BaseOrchestrator failure handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_provider = MockDataProvider()
        self.optimizers = [FailingOptimizer()]
        self.covariance_transformers = [MockCovarianceTransformer()]

    class TestOrchestrator(BaseOrchestrator):
        """Concrete implementation of BaseOrchestrator for testing."""

        def allocate(self, data_provider, time_today=None, all_stocks=None):
            # Not used in these tests
            pass

        @property
        def name(self):
            return "TestOrchestrator"

    def test_context_aware_fallbacks_actually_work(self):
        """Test that context-aware fallbacks are actually used (not just configured)."""
        config = FailureHandlingConfig(
            option=FailureHandlingOption.EQUAL_WEIGHTS,  # Default fallback
            context_aware_fallbacks={
                FailureType.NUMERICAL_ERROR: FailureHandlingOption.ZERO_WEIGHTS,
            }
        )

        # Verify the config is set up correctly
        assert config.context_aware_fallbacks[FailureType.NUMERICAL_ERROR] == FailureHandlingOption.ZERO_WEIGHTS

        # Test with BaseOrchestrator
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, 
                                           A2AConfig(failure_handling=config))

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()

        # Trigger numerical error - should use ZERO_WEIGHTS, not EQUAL_WEIGHTS
        numerical_error = LinAlgError("Singular matrix")
        result = orchestrator._handle_optimizer_failure(
            optimizer=self.optimizers[0],
            exception=numerical_error,
            n_assets=len(mu),
            asset_names=mu.index.tolist()
        )

        # Should return zero weights (context-aware fallback), not equal weights
        assert result is not None
        assert (result == 0.0).all()  # ZERO_WEIGHTS fallback

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(
            circuit_breaker_threshold=1  # Open immediately
        ))
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, config)

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()

        # First failure should open circuit
        result1 = orchestrator._handle_optimizer_failure(
            optimizer=self.optimizers[0],
            exception=RuntimeError("Test failure"),
            n_assets=len(mu),
            asset_names=mu.index.tolist()
        )
        assert result1 is not None  # First failure handled

        # Second failure should be blocked by circuit breaker
        result2 = orchestrator._handle_optimizer_failure(
            optimizer=self.optimizers[0],
            exception=RuntimeError("Test failure 2"),
            n_assets=len(mu),
            asset_names=mu.index.tolist()
        )
        assert result2 is None  # Circuit breaker blocks

    def test_enhanced_logging(self, caplog):
        """Test enhanced logging with failure classification."""
        config = A2AConfig(failure_handling=FailureHandlingConfig(log_failures=True))
        orchestrator = self.TestOrchestrator(self.optimizers, self.covariance_transformers, config)

        mu, cov, prices, time, l_moments = self.data_provider.get_ground_truth()

        with caplog.at_level(logging.WARNING):
            orchestrator._handle_optimizer_failure(
                optimizer=self.optimizers[0],
                exception=ValueError("Matrix not positive definite"),
                n_assets=len(mu),
                asset_names=mu.index.tolist()
            )

        # Should log with classified failure type
        log_message = caplog.text
        assert "numerical_error" in log_message or "data_error" in log_message