"""Failure diagnostics and classification for optimizer error handling.

This module provides utilities for classifying optimizer failures, implementing
retry logic, circuit breaker patterns, and collecting diagnostic information.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from allooptim.config.failure_handling_config import FailureType

logger = logging.getLogger(__name__)


@dataclass
class FailureDiagnostic:
    """Detailed diagnostic information about an optimizer failure."""

    optimizer_name: str
    failure_type: FailureType
    exception: Exception
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    total_retry_time: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic to dictionary for logging/serialization."""
        return {
            "optimizer_name": self.optimizer_name,
            "failure_type": self.failure_type.value,
            "exception_type": type(self.exception).__name__,
            "exception_message": str(self.exception),
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "total_retry_time": self.total_retry_time,
            "context": self.context,
            "stack_trace": self.stack_trace,
        }


class CircuitBreaker:
    """Circuit breaker pattern for temporarily disabling failing optimizers."""

    def __init__(self, threshold: int, timeout_seconds: float = 300.0):
        """Initialize circuit breaker.

        Args:
            threshold: Number of consecutive failures before opening circuit
            timeout_seconds: Time to wait before attempting to close circuit
        """
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, datetime] = {}
        self.open_circuits: Dict[str, datetime] = {}

    def is_open(self, optimizer_name: str) -> bool:
        """Check if circuit is open for given optimizer."""
        if optimizer_name in self.open_circuits:
            open_time = self.open_circuits[optimizer_name]
            if (datetime.now() - open_time).total_seconds() >= self.timeout_seconds:
                # Circuit timeout expired, try to close it
                del self.open_circuits[optimizer_name]
                self.failure_counts[optimizer_name] = 0
                logger.info(f"Circuit breaker closed for optimizer {optimizer_name}")
                return False
            return True
        return False

    def record_failure(self, optimizer_name: str) -> None:
        """Record a failure and potentially open circuit."""
        self.failure_counts[optimizer_name] = self.failure_counts.get(optimizer_name, 0) + 1
        self.last_failure_times[optimizer_name] = datetime.now()

        if self.failure_counts[optimizer_name] >= self.threshold:
            self.open_circuits[optimizer_name] = datetime.now()
            logger.warning(
                f"Circuit breaker opened for optimizer {optimizer_name} "
                f"after {self.failure_counts[optimizer_name]} consecutive failures"
            )

    def record_success(self, optimizer_name: str) -> None:
        """Record a success and reset failure count."""
        if optimizer_name in self.failure_counts:
            self.failure_counts[optimizer_name] = 0
        if optimizer_name in self.open_circuits:
            del self.open_circuits[optimizer_name]


class FailureClassifier:
    """Classifies optimizer failures by type for context-aware handling."""

    @staticmethod
    def classify_failure(exception: Exception) -> FailureType:
        """Classify an exception into a failure type.

        Args:
            exception: The exception that occurred

        Returns:
            Classified failure type
        """
        exception_type = type(exception).__name__
        exception_message = str(exception).lower()

        # Check exception type first for specific classifications
        if exception_type in ["KeyError", "IndexError"]:
            return FailureType.DATA_ERROR
        elif exception_type in ["MemoryError", "TimeoutError", "OSError", "ConnectionError"]:
            return FailureType.RESOURCE_ERROR
        elif exception_type in ["LinAlgError", "ZeroDivisionError", "OverflowError"]:
            return FailureType.NUMERICAL_ERROR

        # Then check message content for more specific classification
        # Numerical errors - specific numerical computation issues
        numerical_keywords = [
            "nan",
            "inf",
            "infinity",
            "overflow",
            "underflow",
            "convergence",
            "singular",
            "not positive definite",
            "linear algebra",
            "eigenvalue",
            "matrix decomposition",
            "cholesky",
            "svd",
            "qr decomposition",
        ]
        if any(keyword in exception_message for keyword in numerical_keywords):
            return FailureType.NUMERICAL_ERROR

        # Data errors - missing or invalid data
        data_keywords = [
            "missing",
            "none",
            "empty",
            "shape",
            "dimension",
            "index",
            "key",
            "not found",
            "does not exist",
            "invalid data",
        ]
        if any(keyword in exception_message for keyword in data_keywords):
            return FailureType.DATA_ERROR

        # Configuration errors - parameter or constraint issues
        config_keywords = [
            "config",
            "parameter",
            "constraint",
            "bound",
            "limit",
            "invalid",
            "configuration",
            "setting",
            "option",
        ]
        if any(keyword in exception_message for keyword in config_keywords):
            return FailureType.CONFIGURATION_ERROR

        # Resource errors - memory, timeout, external service issues
        resource_keywords = [
            "memory",
            "timeout",
            "resource",
            "connection",
            "network",
            "disk",
            "out of memory",
            "timed out",
            "connection refused",
            "service unavailable",
        ]
        if any(keyword in exception_message for keyword in resource_keywords):
            return FailureType.RESOURCE_ERROR

        # Type and attribute errors are typically data-related
        if exception_type in ["TypeError", "AttributeError"]:
            return FailureType.DATA_ERROR

        # ValueError can be many things - check message content more carefully
        if exception_type == "ValueError":
            if any(keyword in exception_message for keyword in numerical_keywords):
                return FailureType.NUMERICAL_ERROR
            elif any(keyword in exception_message for keyword in config_keywords):
                return FailureType.CONFIGURATION_ERROR
            else:
                return FailureType.DATA_ERROR

        # Default to unknown
        return FailureType.UNKNOWN_ERROR


class RetryHandler:
    """Handles retry logic with exponential backoff for transient failures."""

    @staticmethod
    def should_retry(failure_type: FailureType, retry_count: int, max_retries: int) -> bool:
        """Determine if a failure should be retried.

        Args:
            failure_type: Type of failure that occurred
            retry_count: Number of retries already attempted
            max_retries: Maximum number of retries allowed

        Returns:
            True if retry should be attempted
        """
        if retry_count >= max_retries:
            return False

        # Only retry certain types of failures that might be transient
        retryable_types = [
            FailureType.RESOURCE_ERROR,  # Memory, timeout issues
            FailureType.UNKNOWN_ERROR,  # Might be transient
        ]

        return failure_type in retryable_types

    @staticmethod
    def calculate_delay(retry_count: int, base_delay: float) -> float:
        """Calculate delay for retry with exponential backoff.

        Args:
            retry_count: Current retry attempt (0-based)
            base_delay: Base delay in seconds

        Returns:
            Delay in seconds before next retry
        """
        # Exponential backoff with jitter
        import random

        delay = base_delay * (2**retry_count)
        jitter = random.uniform(0.5, 1.5)  # nosec B311 - Pseudo-random for retry jitter, not cryptography
        return delay * jitter

    @staticmethod
    def execute_with_retry(
        func: Callable,
        failure_classifier: FailureClassifier,
        max_retries: int = 1,
        base_delay: float = 0.1,
        enable_diagnostics: bool = False,
    ) -> tuple:
        """Execute a function with retry logic.

        Args:
            func: Function to execute (should return pd.Series)
            failure_classifier: Classifier for failure types
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries
            enable_diagnostics: Whether to collect diagnostics

        Returns:
            Tuple of (result, diagnostic) where diagnostic is None on success
        """
        diagnostic = None

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                result = func()
                execution_time = time.time() - start_time

                if enable_diagnostics and attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1} after {execution_time:.3f}s")

                return result, None

            except Exception as e:
                failure_type = failure_classifier.classify_failure(e)

                if enable_diagnostics:
                    diagnostic = FailureDiagnostic(
                        optimizer_name="unknown",  # Will be set by caller
                        failure_type=failure_type,
                        exception=e,
                        retry_count=attempt,
                        context={"attempt": attempt + 1},
                    )

                if attempt < max_retries and RetryHandler.should_retry(failure_type, attempt, max_retries):
                    delay = RetryHandler.calculate_delay(attempt, base_delay)
                    logger.warning(
                        f"Retryable {failure_type.value} on attempt {attempt + 1}, " f"retrying in {delay:.3f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    # Final failure
                    if enable_diagnostics:
                        diagnostic.total_retry_time = time.time() - start_time if "start_time" in locals() else 0.0
                    return None, diagnostic

        return None, diagnostic


@contextmanager
def failure_diagnostics_context(optimizer_name: str, enable_diagnostics: bool = False):
    """Context manager for collecting failure diagnostics.

    Args:
        optimizer_name: Name of the optimizer being executed
        enable_diagnostics: Whether to enable diagnostic collection

    Yields:
        Diagnostic collector function
    """
    diagnostics = []

    def collect_diagnostic(failure_type: FailureType, exception: Exception, **context):
        """Collect a diagnostic entry."""
        if enable_diagnostics:
            diagnostic = FailureDiagnostic(
                optimizer_name=optimizer_name, failure_type=failure_type, exception=exception, context=context
            )
            diagnostics.append(diagnostic)
            logger.debug(f"Collected diagnostic for {optimizer_name}: {failure_type.value}")

    try:
        yield collect_diagnostic
    finally:
        if diagnostics:
            # Log summary of diagnostics
            logger.warning(f"Optimizer {optimizer_name} had {len(diagnostics)} failure(s) during execution")
