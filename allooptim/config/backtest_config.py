"""Configuration classes for backtesting portfolio optimization strategies.

This module defines Pydantic models and configuration structures for setting up
comprehensive backtesting scenarios. It includes optimizer configurations, data
sources, performance metrics, and reporting options for evaluating portfolio
strategies over historical periods.
"""

import logging
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from allooptim.allocation_to_allocators.a2a_manager_config import A2AManagerConfig
from allooptim.config.cash_config import CashConfig
from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG

logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """Pydantic configuration model for backtest parameters."""

    model_config = DEFAULT_PYDANTIC_CONFIG


    # Time periods
    start_date: datetime = Field(default=datetime(2020, 1, 1), description="Start date for the backtest period")
    end_date: datetime = Field(default=datetime(2024, 12, 31), description="End date for the backtest period")
    quick_start_date: datetime = Field(default=datetime(2022, 12, 31), description="Start date for quick debug testing")
    quick_end_date: datetime = Field(default=datetime(2023, 2, 28), description="End date for quick debug testing")

    # Test mode
    quick_test: bool = Field(default=True, description="Whether to run in quick test mode with shorter time periods")

    # Rebalancing parameters
    rebalance_frequency: int = Field(
        default=20,
        ge=1,
        le=252,  # Max trading days per year
        description="Number of trading days between rebalancing",
    )

    store_results: bool = Field(
        default=True, description="Whether to create a results directory for storing backtest outputs"
    )

    # QuantStats reporting options
    generate_quantstats_reports: bool = Field(
        default=True, description="Whether to generate QuantStats HTML tearsheets"
    )
    quantstats_mode: str = Field(default="full", description="QuantStats tearsheet mode: 'basic' or 'full'")
    quantstats_top_n: int = Field(
        default=5, ge=1, le=50, description="Number of top-performing optimizers to analyze in comparative tearsheets"
    )
    quantstats_individual: bool = Field(
        default=True, description="Whether to generate individual tearsheets for each optimizer"
    )
    quantstats_dir: str = Field(
        default="quantstats_reports", description="Directory name for QuantStats reports within results directory"
    )
    


    @field_validator("quantstats_mode", mode="before")
    @classmethod
    def validate_quantstats_mode(cls, v: str) -> str:
        """Validate that quantstats_mode is either 'basic' or 'full'."""
        allowed_modes = {"basic", "full"}
        if v not in allowed_modes:
            raise ValueError(f"Invalid quantstats_mode: {v}. Must be one of {allowed_modes}")
        return v

    @cached_property
    def results_dir(self) -> Path:
        """Generate results directory path with timestamp."""
        return Path("backtest_results") / datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_report_date_range(self) -> tuple[datetime, datetime]:
        """Get start and end dates based on debug mode."""
        if self.quick_test:
            return self.quick_start_date, self.quick_end_date
        return self.start_date, self.end_date
