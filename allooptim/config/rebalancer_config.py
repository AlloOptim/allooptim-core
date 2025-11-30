from typing import Optional
from pydantic import BaseModel, Field

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG

class RebalancerConfig(BaseModel):
    """Configuration for Portfolio Rebalancer.

    This config holds parameters for multi-strategy portfolio rebalancing
    including EMA smoothing, threshold settings, and trade filtering.
    """

    filter_lifetime_days: Optional[int] = Field(default=5, ge=0, description="Number of days to keep trades in the filter")
    
    absolute_threshold: float = Field(default=0.05, ge=0, description="Absolute threshold for rebalancing")
    
    relative_threshold: float = Field(default=0.15, ge=0, description="Relative threshold for rebalancing")
    
    min_trade_pct: Optional[float] = Field(default=None, ge=0, description="Minimum trade percentage")
    
    max_trades_per_day: Optional[int] = Field(default=None, ge=0, description="Maximum number of trades per day")
    
    trade_to_band_edge: bool = Field(default=False, description="Whether to trade to the band edge")
    
    model_config = DEFAULT_PYDANTIC_CONFIG

