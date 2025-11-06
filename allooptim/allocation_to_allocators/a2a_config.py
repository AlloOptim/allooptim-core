"""Pydantic configuration for A2A orchestrator.

Design Principles:
- NO dict access: Always use config.attribute
- NO hard-coded defaults: All defaults defined here
- Type safe: Automatic validation
- Immutable: frozen=True prevents modification
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class A2AConfig(BaseModel):
    """Pydantic configuration for A2A orchestrator."""

    # Error estimation
    error_estimator_names: List[str] = Field(
        default=["expected_outcome", "sharpe_ratio"], description="List of error estimator names to use"
    )

    # Orchestrator-specific params
    n_simulations: int = Field(default=100, description="Number of simulations for Monte Carlo/Bootstrap")
    n_particles: int = Field(default=30, description="Number of particles for PSO optimization")
    n_pso_iterations: int = Field(default=50, description="Number of PSO iterations")
    meta_model_type: str = Field(
        default="lightgbm", description="Meta-model type for stacking: 'lightgbm', 'xgboost', etc."
    )

    # General settings
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    parallel: bool = Field(default=False, description="Enable parallel execution")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 = all CPUs)")
    allow_partial_investment: bool = Field(
        default=False,
        description="Allow partial investment (0 <= sum(asset_weights) <= 1)",
    )

    model_config = ConfigDict(frozen=True)
