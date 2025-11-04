"""
Orchestrator Factory

Factory for creating allocation-to-allocators orchestrators based on configuration.
"""

from enum import Enum
from typing import List

from allo_optim.allocation_to_allocators.a2a_config import A2AConfig
from allo_optim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allo_optim.allocation_to_allocators.equal_weight_orchestrator import (
    EqualWeightOrchestrator,
)
from allo_optim.allocation_to_allocators.optimized_orchestrator import (
    OptimizedOrchestrator,
)
from allo_optim.allocation_to_allocators.wikipedia_pipeline_orchestrator import (
    WikipediaPipelineOrchestrator,
)
from allo_optim.covariance_transformer.transformer_list import get_transformer_by_names
from allo_optim.optimizer.optimizer_list import get_optimizer_by_names


class OrchestratorType(str, Enum):
    """Enumeration of available orchestrator types."""

    AUTO = "auto"
    EQUAL_WEIGHT = "equal_weight"
    OPTIMIZED = "optimized"
    WIKIPEDIA_PIPELINE = "wikipedia_pipeline"


def create_orchestrator(
    orchestrator_type: str, optimizer_names: List[str], transformer_names: List[str], config: A2AConfig, **kwargs
) -> BaseOrchestrator:
    """
    Factory function to create the appropriate orchestrator based on type.

    Args:
        orchestrator_type: Type of orchestrator to create
        optimizer_names: List of optimizer names to use
        transformer_names: List of covariance transformer names to use
        config: A2AConfig for orchestrator configuration
        **kwargs: Additional arguments specific to orchestrator type

    Returns:
        Configured orchestrator instance

    Raises:
        ValueError: If orchestrator_type is not recognized
    """
    # Get optimizers and transformers
    optimizers = get_optimizer_by_names(optimizer_names)
    transformers = get_transformer_by_names(transformer_names)

    if orchestrator_type == OrchestratorType.AUTO:
        orchestrator_type = get_default_orchestrator_type()

    if orchestrator_type == OrchestratorType.EQUAL_WEIGHT:
        return EqualWeightOrchestrator(optimizers, transformers, config)

    elif orchestrator_type == OrchestratorType.OPTIMIZED:
        return OptimizedOrchestrator(
            optimizers=optimizers,
            covariance_transformers=transformers,
            config=config,
        )

    elif orchestrator_type == OrchestratorType.WIKIPEDIA_PIPELINE:
        # Extract wikipedia pipeline specific parameters
        n_historical_days = kwargs.get("n_historical_days", 60)
        use_wiki_database = kwargs.get("use_wiki_database", False)
        wiki_database_path = kwargs.get("wiki_database_path", None)
        return WikipediaPipelineOrchestrator(
            optimizers=optimizers,
            covariance_transformers=transformers,
            config=config,
            n_historical_days=n_historical_days,
            use_wiki_database=use_wiki_database,
            wiki_database_path=wiki_database_path,
        )

    else:
        raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")


def get_default_orchestrator_type() -> OrchestratorType:
    """
    Determine the default orchestrator type based on optimizer names.

    Args:
        optimizer_names: List of optimizer names being used

    Returns:
        Default orchestrator type string
    """

    return OrchestratorType.EQUAL_WEIGHT
