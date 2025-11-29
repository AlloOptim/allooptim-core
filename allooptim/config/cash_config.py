"""Cash and leverage configuration shared across contexts."""

import logging
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG

logger = logging.getLogger(__name__)


class AllowCashOption(str, Enum):
    """Control flow for cash allowance decisions.

    Determines who decides whether optimizers can hold cash positions.
    """

    GLOBAL_ALLOW_CASH = "global_allow_cash"
    """Force all optimizers to allow cash (override optimizer defaults)."""

    OPTIMIZER_DECIDES = "optimizer_decides"
    """Let each optimizer use its own class default."""

    GLOBAL_FORBID_CASH = "global_forbid_cash"
    """Force all optimizers to forbid cash (override optimizer defaults)."""


class CashConfig(BaseModel):
    """Cash and leverage settings for portfolio allocation.

    Attributes:
        allow_cash_option: Control who decides cash allowance
        max_leverage: Maximum leverage factor (sum(weights) <= max_leverage)
    """

    model_config = DEFAULT_PYDANTIC_CONFIG

    allow_cash_option: AllowCashOption = Field(
        default=AllowCashOption.GLOBAL_ALLOW_CASH,
        description=(
            "Control flow for cash allowance. "
            "GLOBAL_ALLOW_CASH: force all optimizers to allow cash. "
            "OPTIMIZER_DECIDES: each optimizer uses its class default. "
            "GLOBAL_FORBID_CASH: force all optimizers to forbid cash."
        ),
    )

    max_leverage: Optional[float] = Field(
        default=None,
        le=10.0,
        ge=0.0,
        description="Maximum leverage factor (sum(weights) <= max_leverage). " "None = no leverage allowed.",
    )

    @field_validator("allow_cash_option", mode="before")
    @classmethod
    def validate_allow_cash_option(cls, v) -> AllowCashOption:
        """Validate that the allow_cash_option is a valid enum value."""
        return AllowCashOption(v)


def normalize_weights_optimizers(weights: np.ndarray, allow_cash: bool, max_leverage: Optional[float]) -> np.ndarray:
    """Normalize weights based on cash and leverage settings.

    Args:
        weights: Raw weights from optimizer
        allow_cash: Whether cash positions are allowed
        max_leverage: Maximum leverage factor (sum(weights) <= max_leverage)
    """
    if np.sum(weights) <= 0:
        logger.debug("Total weight is non-positive; returning zero weights.")
        return np.zeros_like(weights)

    if max_leverage is None:
        # if leverage is not set, default to 1.0 (no leverage)
        max_leverage = 1.0

    weights = np.clip(weights, a_min=0.0, a_max=max_leverage)

    total_weight = np.sum(weights)

    if total_weight > max_leverage:
        # reduce to max_leverage in all cases
        weights = weights * (max_leverage / total_weight)
        return weights

    if not allow_cash:
        # if cash not allowed, scale up to full investment (sum to 1.0)
        weights = weights / total_weight

    return weights


def normalize_weights_a2a(weights: np.ndarray, cash_config: CashConfig) -> np.ndarray:
    """Normalize weights based on cash and leverage settings from CashConfig.

    Args:
        weights: Raw weights from optimizer
        cash_config: CashConfig object with allow_cash_option and max_leverage
    """
    max_leverage = cash_config.max_leverage

    match cash_config.allow_cash_option:
        case AllowCashOption.GLOBAL_ALLOW_CASH:
            return normalize_weights_optimizers(weights, True, max_leverage)

        case AllowCashOption.GLOBAL_FORBID_CASH:
            return normalize_weights_optimizers(weights, False, max_leverage)

        case AllowCashOption.OPTIMIZER_DECIDES:
            return normalize_weights_optimizers(weights, True, max_leverage)

    raise NotImplementedError(f"Unhandled AllowCashOption: {cash_config.allow_cash_option}")
