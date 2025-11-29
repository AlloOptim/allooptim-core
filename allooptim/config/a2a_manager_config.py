"""Configuration classes for backtesting portfolio optimization strategies.

This module defines Pydantic models and configuration structures for setting up
comprehensive backtesting scenarios. It includes optimizer configurations, data
sources, performance metrics, and reporting options for evaluating portfolio
strategies over historical periods.
"""

import logging
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from allooptim.allocation_to_allocators.orchestrator_factory import OrchestratorType
from allooptim.config.a2a_config import A2AConfig
from allooptim.config.cash_config import CashConfig
from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.covariance_transformer.transformer_list import get_all_transformers
from allooptim.optimizer.optimizer_config import OptimizerConfig
from allooptim.optimizer.optimizer_config_registry import get_optimizer_config_schema
from allooptim.config.a2a_config import A2AConfig
from allooptim.config.rebalancer_config import RebalancerConfig

logger = logging.getLogger(__name__)


class A2AManagerConfig(BaseModel):
    """Pydantic configuration model for backtest parameters."""

    model_config = DEFAULT_PYDANTIC_CONFIG

    benchmark: str = Field(default="SPY", description="Benchmark symbol for the backtest (e.g., SPY)")

    symbols: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        description="List of asset symbols to include in the backtest",
    )

    a2a_config: A2AConfig = Field(
        default_factory=A2AConfig, description="Configuration for A2A orchestration"
    )

    rebalancer_config: RebalancerConfig = Field(
        default_factory=RebalancerConfig, description="Configuration for portfolio rebalancing"
    )

    cash_config: CashConfig = Field(default_factory=CashConfig, description="Cash and leverage settings")

    # Exception handling
    rerun_allocator_exceptions: bool = Field(
        default=False, description="Whether to re-raise exceptions from allocators during backtesting"
    )

    # Return calculation
    log_returns: bool = Field(default=True, description="Whether to use log returns for calculations")

    lookback_days: int = Field(default=90, ge=1, description="Number of days to look back for historical data")

    data_interval: str = Field(
        default="1d", description="Data interval for price data (e.g., '1d', '1wk', '1mo')"
    )

    # Fallback behavior
    use_equal_weights_fallback: bool = Field(
        default=True, description="Whether to use equal weights as fallback when optimization fails"
    )

    # Optimizer and transformer names
    optimizer_configs: List[OptimizerConfig] = Field(
        default_factory=lambda: [
            OptimizerConfig(name="RiskParityOptimizer"),
            OptimizerConfig(name="NaiveOptimizer"),
            OptimizerConfig(name="MomentumOptimizer"),
            OptimizerConfig(name="HRPOptimizer"),
            OptimizerConfig(name="NCOSharpeOptimizer"),
        ],
        min_length=1,
        description="List of optimizer configurations. Can be optimizer names (strings) or OptimizerConfig objects",
    )
    transformer_names: List[str] = Field(
        default=["OracleCovarianceTransformer"],
        min_length=1,
        description="List of covariance transformer names to include in the backtest",
    )

    # AllocationOrchestrator options
    orchestration_type: OrchestratorType = Field(
        default=OrchestratorType.VOLATILITY_ADJUSTED,
        description="Type of orchestration: 'equal_weight', 'optimized', 'wikipedia_pipeline', or "
        "'auto' for automatic selection",
    )

    @field_validator("optimizer_configs", mode="before")
    @classmethod
    def convert_strings_to_optimizer_configs(cls, v: List[Union[str, OptimizerConfig]]) -> List[OptimizerConfig]:
        """Convert string optimizer names to OptimizerConfig objects."""
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(OptimizerConfig(name=item))
            elif isinstance(item, OptimizerConfig):
                result.append(item)
            else:
                raise ValueError(f"Invalid optimizer_config item: {item}. Must be str or OptimizerConfig.")

        return result

    @field_validator("optimizer_configs", mode="after")
    @classmethod
    def validate_unique_display_names(cls, configs: List[OptimizerConfig]) -> List[OptimizerConfig]:
        """Ensure all display names are unique."""
        display_names = [c.display_name for c in configs]
        duplicates = [name for name in set(display_names) if display_names.count(name) > 1]

        if duplicates:
            raise ValueError(
                f"Duplicate display names found: {duplicates}. "
                f"Each optimizer instance must have a unique display_name. "
                f"Provide explicit display_name values to resolve conflicts."
            )

        return configs

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

    @cached_property
    def results_dir(self) -> Path:
        """Generate results directory path with timestamp."""
        return Path("backtest_results") / datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def optimizer_names(self) -> List[str]:
        """Get list of optimizer names for backward compatibility."""
        return [config.name for config in self.optimizer_configs]

    def get_optimizer_configs_dict(self) -> Dict[str, Optional[Dict]]:
        """Get optimizer configs as a dict mapping display_names to config dicts."""
        return {str(config.display_name): config.config for config in self.optimizer_configs}

    def get_optimizer_config_schemas(self) -> Dict[str, Dict]:
        """Get JSON schemas for all configured optimizers."""
        schemas = {}
        for config in self.optimizer_configs:
            try:
                schema = get_optimizer_config_schema(config.name)
                schemas[config.name] = schema
            except Exception as e:
                logger.warning(f"Could not get schema for optimizer {config.name}: {e}")
                schemas[config.name] = {"error": str(e)}
        return schemas
