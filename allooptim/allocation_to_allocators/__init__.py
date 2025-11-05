"""
Allocation to Allocators Package

This package contains components for the A2A (Allocation-to-Allocators) architecture,
including data providers, simulators, optimizers, and error estimation.
"""

from .data_provider_factory import (
    AbstractDataProviderFactory,
    BacktestDataProviderFactory,
    MCOSDataProviderFactory,
    get_data_provider_factory,
)
from .error_estimator import (
    AbstractErrorEstimator,
    ErrorEstimatorRegistry,
    error_estimator_registry,
)
from .observation_simulator import MuCovPartialObservationSimulator
from .simulator_interface import AbstractObservationSimulator

__all__ = [
    # Data provider factory
    "AbstractDataProviderFactory",
    "BacktestDataProviderFactory",
    "MCOSDataProviderFactory",
    "get_data_provider_factory",
    # Error estimation
    "AbstractErrorEstimator",
    "ErrorEstimatorRegistry",
    "error_estimator_registry",
    # Existing components
    "MuCovPartialObservationSimulator",
    "AbstractObservationSimulator",
]
