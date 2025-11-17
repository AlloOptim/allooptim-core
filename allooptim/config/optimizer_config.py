"""Configuration classes for optimizer instances.

This module defines Pydantic models for configuring optimizer instances,
including display name generation and validation.
"""

import logging
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from allooptim.optimizer.optimizer_config_registry import get_all_optimizer_names, validate_optimizer_config

logger = logging.getLogger(__name__)


class OptimizerConfig(BaseModel):
    """Configuration for a single optimizer with optional custom parameters."""

    name: str = Field(..., description="Optimizer class name for instantiation")
    display_name: Optional[str] = Field(
        default=None,
        description="Unique identifier for results and reporting. "
                    "Auto-generated from config if not provided."
    )
    config: Optional[Dict] = Field(
        default=None,
        description="Optional custom configuration parameters"
    )

    @field_validator("name", mode="before")
    @classmethod
    def validate_optimizer_name(cls, v: str) -> str:
        """Validate that the optimizer name exists."""
        available_optimizers = get_all_optimizer_names()
        if v not in available_optimizers:
            raise ValueError(f"Invalid optimizer name: {v}. " f"Available optimizers: {available_optimizers}")
        return v

    @field_validator("config", mode="before")
    @classmethod
    def validate_config(cls, v: Optional[Dict], info) -> Optional[Dict]:
        """Validate the config against the optimizer's schema if provided."""
        if v is None:
            return v

        # Get the optimizer name from the current values
        name = info.data.get("name")
        if name:
            try:
                validate_optimizer_config(name, v)
            except Exception as e:
                raise ValueError(f"Invalid config for optimizer {name}: {e}") from e

        return v

    @field_validator("display_name", mode="after")
    @classmethod
    def generate_display_name(cls, v: Optional[str], info) -> str:
        """Auto-generate display name if not provided.

        Generation strategy:
        1. If display_name provided explicitly, use it
        2. If no config, use class name
        3. If config exists, append key parameters to class name
        """
        if v is not None:
            return v

        name = info.data.get("name")
        config = info.data.get("config")

        if not config:
            return name

        # Generate suffix from config
        suffix = cls._generate_config_suffix(config)
        if suffix:
            return f"{name}[{suffix}]"

        return name

    @model_validator(mode="after")
    def ensure_display_name_set(self) -> "OptimizerConfig":
        """Ensure display_name is set after model construction."""
        if self.display_name is None:
            if not self.config:
                self.display_name = self.name
            else:
                # Generate suffix from config
                suffix = self._generate_config_suffix(self.config)
                if suffix:
                    self.display_name = f"{self.name}[{suffix}]"
                else:
                    self.display_name = self.name
        return self

    @staticmethod
    def _generate_config_suffix(config: Dict, max_params: int = 2) -> str:
        """Generate compact suffix from config parameters.

        Args:
            config: Configuration dictionary
            max_params: Maximum number of parameters to include

        Returns:
            Suffix string like "param1=value1-param2=value2"
        """
        if not config:
            return ""

        # Filter to serializable primitive values
        simple_params = {
            k: v for k, v in config.items()
            if isinstance(v, (str, int, float, bool))
        }

        if not simple_params:
            return ""

        # Take first max_params, sorted for consistency
        items = sorted(simple_params.items())[:max_params]

        # Format values
        parts = []
        for key, value in items:
            if isinstance(value, float):
                value_str = f"{value:.2g}"  # Compact float representation
            elif isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)

            parts.append(f"{key}={value_str}")

        return "-".join(parts)