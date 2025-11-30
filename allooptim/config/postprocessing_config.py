"""Pydantic configuration for A2A orchestrator.

Design Principles:
- NO dict access: Always use config.attribute
- NO hard-coded defaults: All defaults defined here
- Type safe: Automatic validation
- Immutable: frozen=True prevents modification
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG


class PostProcessingConfig(BaseModel):

    max_quantile_assets: Optional[float] = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of assets to keep based on weight quantiles (e.g., 0.3 means keep top 30% by weight). Disabled if None.",
    )

    max_asset_concentration_pct: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum concentration for any single asset (e.g., 0.3 means weights above 30% get clipped to 30%). Disabled if None.",
    )
    n_min_assets: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum number of assets with weights > 0. Disabled if None.",
    )
    # Asset allocation constraints
    n_max_assets: Optional[int] = Field(
        default=200,
        ge=1,
        description="Maximum number of assets with weights > 0. If more assets are active, reduce by setting the smallest ones to 0. Disabled if None.",
    )
    
    # Define these configs here again, to be safe
    allow_leverage: bool = Field(
        default=False,
        description="Whether leverage is allowed in the portfolio",
    )

    full_investment: bool = Field(
        default=True,
        description="Whether the portfolio should be fully invested (weights sum to 1)",
    )
    
    model_config = DEFAULT_PYDANTIC_CONFIG
    
    @model_validator(mode="after")
    def check_n_max_assets(self):
        if (
            self.n_min_assets is not None
            and self.n_max_assets is not None
            and self.n_max_assets < self.n_min_assets
        ):
            raise ValueError("n_max_assets must be greater than or equal to n_min_assets")
        return self
