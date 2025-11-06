"""A2A Orchestrator Base Classes

Abstract base classes for Allocation-to-Allocators orchestration.
Provides clean separation between different orchestration strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import pandas as pd

from allooptim.allocation_to_allocators.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.a2a_result import A2AResult
from allooptim.allocation_to_allocators.simulator_interface import (
    AbstractObservationSimulator,
)
from allooptim.config.stock_dataclasses import StockUniverse
from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer


class A2AOrchestrator(ABC):
    """Abstract base class for Allocation-to-Allocators orchestration.

    Design Philosophy:
    - INTERFACE ONLY: No implementation in ABC
    - Receives data provider (not pre-computed allocations)
    - Controls when/how optimizers are called
    - Time-agnostic (operates on single time step)
    - Configuration-driven (Pydantic, not dicts)

    Note: Concrete implementations should inherit from BaseOrchestrator, not this class directly.
    """

    @abstractmethod
    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List["StockUniverse"]] = None,
    ) -> A2AResult:
        """Orchestrate allocation process for current time step.

        Args:
            data_provider: Provides ground truth and sampling capability for current time step
            time_today: Current time step (optional, can be derived from data_provider)
            all_stocks: List of all available stocks (optional, used by some orchestrators)

        Returns:
            A2AResult with final allocation and all statistics
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Orchestrator name for logging/reporting."""
        pass


class BaseOrchestrator(A2AOrchestrator):
    """Base implementation providing shared functionality for all orchestrators.

    Design Philosophy:
    - CONCRETE IMPLEMENTATION: Provides utility methods
    - NO ABSTRACT METHODS: Except allocate() from parent
    - SHARED LOGIC: Covariance transformation, result creation
    - EXTENSIBLE: Subclasses override allocate() only
    """

    def __init__(
        self,
        optimizers: List[AbstractOptimizer],
        covariance_transformers: List[AbstractCovarianceTransformer],
        config: A2AConfig,
    ):
        """Initialize orchestrator.

        Args:
            optimizers: List of optimizer instances
            covariance_transformers: List of covariance transformers to apply
            config: A2A configuration object
        """
        self.optimizers = optimizers
        self.covariance_transformers = covariance_transformers
        self.config = config

    def _apply_covariance_transformers(self, cov: pd.DataFrame, n_observations: int) -> pd.DataFrame:
        """Apply covariance transformation pipeline.

        Args:
            cov: Raw covariance matrix from data provider
            n_observations: Number of observations used to compute covariance

        Returns:
            Transformed covariance matrix
        """
        cov_transformed = cov
        for transformer in self.covariance_transformers:
            cov_transformed = transformer.transform(cov_transformed, n_observations)
        return cov_transformed

    @abstractmethod
    def allocate(
        self,
        data_provider: AbstractObservationSimulator,
        time_today: Optional[datetime] = None,
        all_stocks: Optional[List[StockUniverse]] = None,
    ) -> A2AResult:
        """Orchestrate allocation process for current time step.

        Args:
            data_provider: Provides ground truth and sampling capability for current time step
            time_today: Current time step (optional, can be derived from data_provider)
            all_stocks: List of all available stocks (optional, used by some orchestrators)

        Returns:
            A2AResult with final allocation and all statistics

        Implementation Guide:
        1. Call data_provider.get_sample() as needed (Monte Carlo, Bootstrap, etc.)
        2. Unpack sample: mu, cov, prices, time, l_moments = data_provider.get_sample()
        3. Apply covariance transformers: cov_transformed = self._apply_covariance_transformers(cov, n_obs)
        4. Call optimizers: allocation = optimizer.allocate(mu, cov_transformed, prices, time, l_moments)
        5. Aggregate optimizer results (equal weight, PSO, meta-learning, etc.)
        6. Return A2AResult
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Orchestrator name for logging/reporting."""
        pass
