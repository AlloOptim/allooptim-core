import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from functools import cached_property

import yaml
from pydantic import BaseModel, Field, field_validator

from allo_optim.allocation_to_allocators.orchestrator_factory import OrchestratorType
from allo_optim.covariance_transformer.transformer_list import get_all_transformers
from allo_optim.optimizer.optimizer_list import get_all_optimizer_names

logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """Pydantic configuration model for backtest parameters."""

    benchmark: str = Field(default="SPY", description="Benchmark symbol for the backtest (e.g., SPY)")

    symbols: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        description="List of asset symbols to include in the backtest",
    )

    # Exception handling
    rerun_allocator_exceptions: bool = Field(
        default=False, description="Whether to re-raise exceptions from allocators during backtesting"
    )

    # Return calculation
    log_returns: bool = Field(default=True, description="Whether to use log returns for calculations")

    # Time periods
    start_date: datetime = Field(..., description="Start date for the backtest period")
    end_date: datetime = Field(..., description="End date for the backtest period")
    quick_start_date: datetime = Field(default=datetime(2022, 12, 31), description="Start date for quick debug testing")
    quick_end_date: datetime = Field(default=datetime(2023, 2, 28), description="End date for quick debug testing")

    # Test mode
    quick_test: bool = Field(default=True, description="Whether to run in quick test mode with shorter time periods")

    # Rebalancing parameters
    rebalance_frequency: int = Field(
        default=10,
        ge=1,
        le=252,  # Max trading days per year
        description="Number of trading days between rebalancing",
    )
    lookback_days: int = Field(default=60, ge=1, description="Number of days to look back for historical data")

    # Fallback behavior
    use_equal_weights_fallback: bool = Field(
        default=True, description="Whether to use equal weights as fallback when optimization fails"
    )

    # Optimizer and transformer names
    optimizer_names: List[str] = Field(
        default=["RiskParity", "Naive", "CappedMomentum", "HRP", "NCO"],
        min_length=1,
        description="List of optimizer names to include in the backtest",
    )
    transformer_names: List[str] = Field(
        default=["OracleCovarianceTransformer"],
        min_length=1,
        description="List of covariance transformer names to include in the backtest",
    )

    # AllocationOrchestrator options
    orchestration_type: OrchestratorType = Field(
        default=OrchestratorType.AUTO,
        description="Type of orchestration: 'equal_weight', 'optimized', 'wikipedia_pipeline', or 'auto' for automatic selection",
    )

    @field_validator("optimizer_names", mode="before")
    @classmethod
    def validate_optimizer_names(cls, v: List[str]) -> List[str]:
        """Validate that all optimizer names exist and at least one is present."""
        if not v:
            raise ValueError("At least one optimizer name must be provided")

        available_optimizers = get_all_optimizer_names()
        invalid_names = [name for name in v if name not in available_optimizers]

        if invalid_names:
            raise ValueError(
                f"Invalid optimizer names: {invalid_names}. " f"Available optimizers: {available_optimizers}"
            )

        return v

    @field_validator("transformer_names", mode="before")
    @classmethod
    def validate_transformer_names(cls, v: List[str]) -> List[str]:
        """Validate that all transformer names exist and at least one is present."""
        if not v:
            raise ValueError("At least one transformer name must be provided")

        available_transformers = [t.name for t in get_all_transformers()]
        invalid_names = [name for name in v if name not in available_transformers]

        if invalid_names:
            raise ValueError(
                f"Invalid transformer names: {invalid_names}. " f"Available transformers: {available_transformers}"
            )

        return v

    @field_validator("orchestration_type", mode="before")
    @classmethod
    def validate_orchestration_type(cls, v: str) -> OrchestratorType:
        """Validate that orchestration type is one of the allowed values."""
        return OrchestratorType(v)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BacktestConfig":
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Convert string dates to datetime objects
        if "start_date" in data and isinstance(data["start_date"], str):
            data["start_date"] = datetime.fromisoformat(data["start_date"])
        if "end_date" in data and isinstance(data["end_date"], str):
            data["end_date"] = datetime.fromisoformat(data["end_date"])
        if "debug_start_date" in data and isinstance(data["debug_start_date"], str):
            data["debug_start_date"] = datetime.fromisoformat(data["debug_start_date"])
        if "debug_end_date" in data and isinstance(data["debug_end_date"], str):
            data["debug_end_date"] = datetime.fromisoformat(data["debug_end_date"])

        return cls(**data)

    @cached_property
    def results_dir(self) -> Path:
        """Generate results directory path with timestamp."""
        return Path("backtest_results") / datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_report_date_range(self) -> tuple[datetime, datetime]:
        """Get start and end dates based on debug mode."""
        if self.quick_test:
            return self.quick_start_date, self.quick_end_date
        return self.start_date, self.end_date

    def get_data_date_range(self) -> tuple[datetime, datetime]:
        """Get start and end dates for data loading with lookback period."""
        previous_days = timedelta(days=self.lookback_days)

        if self.quick_test:
            return self.quick_start_date - previous_days, self.quick_end_date
        return self.start_date - previous_days, self.end_date
