"""Configuration for optimizer failure handling.

This module defines the configuration classes for handling optimizer failures
gracefully in A2A orchestration, following the same pattern as cash_config.py.
"""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG


class FailureHandlingOption(str, Enum):
    """Control flow for optimizer failure handling in A2A orchestration."""

    ZERO_WEIGHTS = "zero_weights"
    """Return all-zero weights (no investment) on optimizer failure."""

    EQUAL_WEIGHTS = "equal_weights"
    """Return 1/N naive diversification weights on optimizer failure."""

    IGNORE_OPTIMIZER = "ignore_optimizer"
    """Skip failed optimizer entirely in ensemble combination."""


class FailureType(str, Enum):
    """Classification of optimizer failure types for context-aware handling."""

    NUMERICAL_ERROR = "numerical_error"
    """Numerical computation errors (NaN, inf, convergence issues)."""

    DATA_ERROR = "data_error"
    """Data-related errors (missing data, invalid inputs, covariance issues)."""

    CONFIGURATION_ERROR = "configuration_error"
    """Configuration errors (invalid parameters, constraint conflicts)."""

    RESOURCE_ERROR = "resource_error"
    """Resource-related errors (memory, timeout, external service failures)."""

    UNKNOWN_ERROR = "unknown_error"
    """Unclassified or unexpected errors."""


class FailureHandlingConfig(BaseModel):
    """Configuration for handling optimizer failures in A2A orchestration.

    This config controls how the system responds when individual optimizers
    fail during A2A (Allocation-to-Allocators) orchestration. It provides
    three global strategies for graceful degradation with enhanced features.

    Attributes:
        option: The default failure handling strategy to use
        log_failures: Whether to log optimizer failures at WARNING level
        raise_on_all_failed: Whether to raise exception if ALL optimizers fail
        retry_attempts: Number of retry attempts for transient failures
        retry_delay_seconds: Delay between retry attempts
        context_aware_fallbacks: Type-specific fallback strategies
        enable_diagnostics: Whether to collect detailed failure diagnostics
        circuit_breaker_threshold: Number of consecutive failures before disabling optimizer
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    option: FailureHandlingOption = Field(
        default=FailureHandlingOption.EQUAL_WEIGHTS,
        description=(
            "Default failure handling strategy for A2A orchestration. "
            "ZERO_WEIGHTS: Return all-zero weights. "
            "EQUAL_WEIGHTS: Return 1/N naive diversification. "
            "IGNORE_OPTIMIZER: Skip optimizer entirely in ensemble."
        ),
    )

    log_failures: bool = Field(
        default=True,
        description="Whether to log optimizer failures at WARNING level"
    )

    raise_on_all_failed: bool = Field(
        default=False,
        description="Raise exception if ALL optimizers fail (vs returning equal weights)"
    )

    retry_attempts: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Number of retry attempts for transient failures (0 = no retries)"
    )

    retry_delay_seconds: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Delay in seconds between retry attempts"
    )

    context_aware_fallbacks: Dict[FailureType, FailureHandlingOption] = Field(
        default_factory=dict,
        description=(
            "Type-specific fallback strategies. If not specified for a failure type, "
            "the default 'option' will be used. Example: "
            "{'numerical_error': 'zero_weights', 'data_error': 'equal_weights'}"
        )
    )

    enable_diagnostics: bool = Field(
        default=False,
        description="Whether to collect detailed failure diagnostics and metrics"
    )

    circuit_breaker_threshold: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of consecutive failures before temporarily disabling an optimizer. "
            "None disables circuit breaker functionality."
        )
    )

    @field_validator('context_aware_fallbacks')
    @classmethod
    def validate_context_aware_fallbacks(cls, v):
        """Validate that context-aware fallbacks use valid FailureHandlingOption values."""
        if not isinstance(v, dict):
            raise ValueError("context_aware_fallbacks must be a dictionary")

        for failure_type, handling_option in v.items():
            if not isinstance(failure_type, FailureType):
                # Try to convert string to enum
                try:
                    FailureType(failure_type)
                except ValueError:
                    raise ValueError(f"Invalid failure type: {failure_type}")

            if not isinstance(handling_option, FailureHandlingOption):
                # Try to convert string to enum
                try:
                    FailureHandlingOption(handling_option)
                except ValueError:
                    raise ValueError(f"Invalid handling option: {handling_option}")

        return v