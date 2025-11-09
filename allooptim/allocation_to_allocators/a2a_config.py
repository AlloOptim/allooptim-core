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
    # Simulation settings
    n_simulations: int = Field(
        default=20,
        ge=1,
        description="Number of Monte Carlo simulations to run",
    )
    n_data_observations: int = Field(
        default=20,
        description="Number of data observations per simulation (same as n_simulations)",
    )
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

    # Convergence settings
    convergence_threshold: float = Field(
        default=2.0,
        ge=0.0,
        description="Statistical threshold in standard deviations for convergence detection",
    )
    min_points_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of time steps needed before checking convergence",
    )
    run_all_steps: bool = Field(
        default=True,
        description="Run all simulation steps instead of using convergence detection",
    )
    
    custom_a2a_weights: Optional[dict] = Field(
        default=None,
        description="Custom A2A weights for each optimizer"
    )

    model_config = ConfigDict(frozen=True)
