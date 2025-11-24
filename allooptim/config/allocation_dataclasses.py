"""Data classes for portfolio allocation results and statistics.

This module defines Pydantic models and data structures for representing
portfolio allocation outputs, performance statistics, and optimization results.
These classes ensure type safety and validation for allocation data throughout
the AlloOptim pipeline.

Key components:
- StatisticsType: Enumeration of allocation statistics types
- Allocation result models with validation
- Performance metrics data structures
- Type-safe interfaces for optimizer outputs
"""

from enum import Enum
from typing import Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from .default_pydantic_config import DEFAULT_PYDANTIC_CONFIG


class StatisticsType(str, Enum):
    """Enumeration of allocation statistics types."""

    A2A = "A2A"
    WIKIPEDIA = "WIKIPEDIA"
    NONE = "NONE"


class NoStatistics(BaseModel):
    """Represents absence of allocation statistics."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    type: StatisticsType = StatisticsType.NONE


class A2AStatistics(BaseModel):
    """Statistics from Allocation-to-Allocators (A2A) optimization."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    asset_returns: dict[str, float]
    asset_volatilities: dict[str, float]
    algo_runtime: dict[str, float]
    algo_weights: dict[str, float]
    algo_memory_usage: dict[str, float]  # Memory usage per algorithm (MB)
    algo_computation_time: dict[str, float]  # Computation time per algorithm (seconds)
    type: StatisticsType = StatisticsType.A2A


class WikipediaStatistics(BaseModel):
    """Statistics from Wikipedia-based stock allocation."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    end_date: str
    r_squared: float
    p_value: float
    std_err: float
    slope: float
    intercept: float
    all_symbols: list[str]
    valid_data_symbols: list[str]
    significant_positive_stocks: list[str]
    top_n_symbols: list[str]
    type: StatisticsType = StatisticsType.WIKIPEDIA


class AllocationResult(BaseModel):
    """Result of an allocation operation with comprehensive metadata."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    asset_weights: dict[str, float]
    success: bool
    statistics: Union[A2AStatistics, WikipediaStatistics, NoStatistics]
    computation_time: Optional[float] = None
    error_message: Optional[str] = None
    df_allocation: Optional[pd.DataFrame] = None
    optimizer_memory_usage: Optional[dict[str, float]] = None
    optimizer_computation_time: Optional[dict[str, float]] = None

    def __hash__(self):
        """Hash function for the dataclass to enable use in sets and as dict keys."""
        return hash((type(self),) + tuple(self.__dict__.values()))

    @field_validator("asset_weights", mode="before")
    @classmethod
    def check_asset_weights(cls, values: dict[str, float]) -> dict[str, float]:
        """Validate that asset weights are non-negative."""
        for asset, weight in values.items():
            if weight < 0.0:
                raise ValueError(f"Asset weights must be non-negative, got {weight} for {asset}")
            return values


class AllocationStatisticsResult(BaseModel):
    """Statistics from allocation operations."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    returns: dict[str, float]
    volatilities: dict[str, float]
    runtime: dict[str, float]
    algo_weights: dict[str, float]
    asset_weights: dict[str, float]


def validate_asset_weights_length(asset_weights: dict[str, float], n_assets: int) -> None:
    """Validate that asset weights dictionary has the correct length."""
    if len(asset_weights) != n_assets:
        raise ValueError(f"Asset weights length {len(asset_weights)} does not match number of assets {n_assets}")
