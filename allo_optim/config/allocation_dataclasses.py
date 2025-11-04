from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, field_validator


class StatisticsType(str, Enum):
    A2A = "A2A"
    WIKIPEDIA = "WIKIPEDIA"
    NONE = "NONE"


class NoStatistics(BaseModel):
    type: StatisticsType = StatisticsType.NONE


class A2AStatistics(BaseModel):
    asset_returns: dict[str, float]
    asset_volatilities: dict[str, float]
    algo_runtime: dict[str, float]
    algo_weights: dict[str, float]
    type: StatisticsType = StatisticsType.A2A


class WikipediaStatistics(BaseModel):
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
    asset_weights: dict[str, float]
    success: bool
    statistics: Union[A2AStatistics, WikipediaStatistics, NoStatistics]
    computation_time: Optional[float] = None
    error_message: Optional[str] = None

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    @field_validator("asset_weights")
    @classmethod
    def check_asset_weights(cls, values: dict[str, float]) -> dict[str, float]:
        for asset, weight in values.items():
            if weight < 0.0:
                raise ValueError(f"Asset weights must be non-negative, got {weight} for {asset}")
            return values


class AllocationStatisticsResult(BaseModel):
    """Statistics from allocation operations"""

    returns: dict[str, float]
    volatilities: dict[str, float]
    runtime: dict[str, float]
    algo_weights: dict[str, float]
    asset_weights: dict[str, float]


def validate_asset_weights_length(asset_weights: dict[str, float], n_assets: int) -> None:
    if len(asset_weights) != n_assets:
        raise ValueError(f"Asset weights length {len(asset_weights)} does not match number of assets {n_assets}")
