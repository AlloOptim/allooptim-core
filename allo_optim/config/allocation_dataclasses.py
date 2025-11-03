from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, field_validator


class StatisticsType(str, Enum):
    A2A = "A2A"
    WIKIPEDIA = "WIKIPEDIA"
    NONE = "NONE"


class NoStatistics(BaseModel):
    type: StatisticsType = StatisticsType.NONE


class A2AStatistics(BaseModel):
    asset_returns: Dict[str, float]
    asset_volatilities: Dict[str, float]
    algo_runtime: Dict[str, float]
    algo_weights: Dict[str, float]
    type: StatisticsType = StatisticsType.A2A


class WikipediaStatistics(BaseModel):
    end_date: str
    r_squared: float
    p_value: float
    std_err: float
    slope: float
    intercept: float
    all_symbols: List[str]
    valid_data_symbols: List[str]
    significant_positive_stocks: List[str]
    top_n_symbols: List[str]
    type: StatisticsType = StatisticsType.WIKIPEDIA


class AllocationResult(BaseModel):
    asset_weights: Dict[str, float]
    success: bool
    statistics: Union[A2AStatistics, WikipediaStatistics, NoStatistics]
    computation_time: Optional[float] = None
    error_message: Optional[str] = None

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    @field_validator("asset_weights")
    @classmethod
    def check_asset_weights(cls, values: Dict[str, float]) -> Dict[str, float]:
        for asset, weight in values.items():
            if weight < 0.0:
                raise ValueError(f"Asset weights must be non-negative, got {weight} for {asset}")        
            return values


class AllocationStatisticsResult(BaseModel):
    """Statistics from allocation operations"""

    returns: Dict[str, float]
    volatilities: Dict[str, float]
    runtime: Dict[str, float]
    algo_weights: Dict[str, float]
    asset_weights: Dict[str, float]


def validate_asset_weights_length(asset_weights: Dict[str, float], n_assets: int) -> None:
    if len(asset_weights) != n_assets:
        raise ValueError(f"Asset weights length {len(asset_weights)} does not match number of assets {n_assets}")
