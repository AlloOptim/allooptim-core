"""Pydantic models for A2A result structures - Phase 3: Result Structure Refinement.

This module defines nested Pydantic models to eliminate dict usage in favor of
structured, type-safe result objects as specified in the future A2A architecture.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from allooptim.config.a2a_config import A2AConfig


class OptimizerAllocation(BaseModel):
    """Single optimizer's allocation result."""

    instance_id: str = Field(description="Unique identifier of the optimizer instance")
    weights: pd.Series = Field(description="Asset weights (asset_name -> weight)")
    runtime_seconds: Optional[float] = Field(
        default=None, description="Time taken to compute this allocation (seconds)"
    )
    memory_usage_mb: Optional[float] = Field(default=None, description="Peak memory usage during computation (MB)")

    # Backward compatibility alias
    @property
    def optimizer_name(self) -> str:
        """Deprecated: Use instance_id instead."""
        return self.instance_id

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OptimizerWeight(BaseModel):
    """Weight assigned to an optimizer in ensemble."""

    instance_id: str = Field(description="Unique identifier of the optimizer instance")
    weight: float = Field(description="Contribution weight (all weights sum to 1)")

    # Backward compatibility alias
    @property
    def optimizer_name(self) -> str:
        """Deprecated: Use instance_id instead."""
        return self.instance_id


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""

    expected_return: float = Field(description="Portfolio expected return")
    volatility: float = Field(description="Portfolio volatility")
    sharpe_ratio: float = Field(description="Portfolio Sharpe ratio")
    diversity_score: float = Field(description="Optimizer diversity score (1 - avg correlation)")
    max_drawdown: Optional[float] = Field(default=None, description="Maximum drawdown")
    turnover: Optional[float] = Field(default=None, description="Portfolio turnover")

    model_config = ConfigDict(frozen=True)


class OptimizerError(BaseModel):
    """Error metric for an optimizer."""

    instance_id: str = Field(description="Unique identifier of the optimizer instance")
    error: float = Field(description="Error metric value")
    error_components: List[float] = Field(default_factory=list, description="Individual error estimator values")

    # Backward compatibility alias
    @property
    def optimizer_name(self) -> str:
        """Deprecated: Use instance_id instead."""
        return self.instance_id


class A2AResult(BaseModel):
    """Pydantic result structure from A2A orchestration.

    Design Principles:
    - NO dicts for structured data - all typed models
    - Type-safe result with Pydantic validation
    - Immutable after creation (frozen=True)
    - Pandas Series for final allocation (natural format for weights)
    """

    # Primary output
    final_allocation: pd.Series = Field(description="Final portfolio weights (asset_name -> weight, sum to 1)")

    # Optimizer information
    optimizer_allocations: List[OptimizerAllocation] = Field(description="List of optimizer allocation results")
    optimizer_weights: List[OptimizerWeight] = Field(description="Weights assigned to each optimizer")

    # Performance metrics (structured model, not dict)
    metrics: PerformanceMetrics = Field(description="Portfolio performance metrics")

    # Detailed statistics
    runtime_seconds: float = Field(description="Execution time in seconds")
    n_simulations: int = Field(description="Number of simulations performed")
    optimizer_errors: List[OptimizerError] = Field(description="Error metrics per optimizer")

    # Metadata
    orchestrator_name: str = Field(description="Name of orchestrator used")
    timestamp: datetime = Field(description="Time step for this allocation")
    config: A2AConfig = Field(description="Configuration used (Pydantic model, not dict snapshot)")

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)  # Allow pandas Series, datetime

    def to_dataframe(self) -> pd.DataFrame:
        """Convert optimizer allocations to DataFrame.

        Returns:
            DataFrame with optimizers as columns, assets as rows
        """
        alloc_dict = {alloc.instance_id: alloc.weights for alloc in self.optimizer_allocations}
        return pd.DataFrame(alloc_dict)

    def get_optimizer_weights_series(self) -> pd.Series:
        """Get optimizer weights as pandas Series.

        Returns:
            Series with optimizer names as index, weights as values
        """
        return pd.Series({w.instance_id: w.weight for w in self.optimizer_weights})


# Rebuild model to resolve forward references
A2AResult.model_rebuild()
