"""Fundamental data models and types.

This module contains data models for fundamental company data
used across different data providers.
"""

from pydantic import BaseModel

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG


class FundamentalData(BaseModel):
    """Fundamental data for a single ticker."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    ticker: str
    market_cap: float | None = None
    roe: float | None = None  # Return on Equity
    debt_to_equity: float | None = None
    pb_ratio: float | None = None  # Price to Book ratio
    current_ratio: float | None = None

    @property
    def is_valid(self) -> bool:
        """Determine if this fundamental data is valid."""
        return any(
            [
                self.market_cap is not None,
                self.roe is not None,
                self.debt_to_equity is not None,
                self.pb_ratio is not None,
                self.current_ratio is not None,
            ]
        )