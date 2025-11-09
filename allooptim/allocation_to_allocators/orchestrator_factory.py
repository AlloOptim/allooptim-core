"""Orchestrator Factory.

Factory for creating allocation-to-allocators orchestrators based on configuration.
"""

from enum import Enum
from typing import Dict, List, Optional

from allooptim.allocation_to_allocators.a2a_config import A2AConfig
from allooptim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allooptim.allocation_to_allocators.equal_weight_orchestrator import (
    EqualWeightOrchestrator,
    MedianWeightOrchestrator,
    CustomWeightOrchestrator,
)
from allooptim.allocation_to_allocators.optimized_orchestrator import (
    OptimizedOrchestrator,
)
from allooptim.allocation_to_allocators.wikipedia_pipeline_orchestrator import (
    WikipediaPipelineOrchestrator,
)
from allooptim.covariance_transformer.transformer_list import get_transformer_by_names
from allooptim.optimizer.optimizer_factory import get_optimizer_by_names_with_configs


class OrchestratorType(str, Enum):
    """Enumeration of available orchestrator types."""

    AUTO = "auto"
    EQUAL_WEIGHT = "equal_weight"
    MEDIAN_WEIGHT = "median_weight"
    CUSTOM_WEIGHT = "custom_weight"
    OPTIMIZED = "optimized"
    WIKIPEDIA_PIPELINE = "wikipedia_pipeline"


def create_orchestrator(
    orchestrator_type: str,
    optimizer_names: Optional[List[str]] = None,
    optimizer_configs: Optional[Dict[str, Optional[Dict]]] = None,
    transformer_names: List[str] = None,
    config: A2AConfig = None,
    **kwargs,
) -> BaseOrchestrator:
    """Factory function to create the appropriate orchestrator based on type.

    Args:
        orchestrator_type: Type of orchestrator to create
        optimizer_names: List of optimizer names to use (deprecated, use optimizer_configs)
        optimizer_configs: Dict mapping optimizer names to optional config dicts
        transformer_names: List of covariance transformer names to use
        config: A2AConfig for orchestrator configuration
        **kwargs: Additional arguments specific to orchestrator type

    Returns:
        Configured orchestrator instance

    Raises:
        ValueError: If orchestrator_type is not recognized
    """
    # Handle backward compatibility
    if optimizer_names is not None and optimizer_configs is None:
        optimizer_configs = {name: None for name in optimizer_names}
    elif optimizer_configs is None:
        raise ValueError("Either optimizer_names (deprecated) or optimizer_configs must be provided")

    # Get optimizers and transformers
    optimizers = get_optimizer_by_names_with_configs(optimizer_configs)
    transformers = get_transformer_by_names(transformer_names)

    if orchestrator_type == OrchestratorType.AUTO:
        orchestrator_type = get_default_orchestrator_type()

    match orchestrator_type:
        case OrchestratorType.EQUAL_WEIGHT:
            return EqualWeightOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                config=config,
            )
        case OrchestratorType.MEDIAN_WEIGHT:
            return MedianWeightOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                config=config,
            )
        case OrchestratorType.CUSTOM_WEIGHT:
            return CustomWeightOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                config=config,
            )

        case OrchestratorType.OPTIMIZED:
            return OptimizedOrchestrator(
            optimizers=optimizers,
            covariance_transformers=transformers,
            config=config,
        )

        case OrchestratorType.WIKIPEDIA_PIPELINE:
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

        case _:
            raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")


def get_default_orchestrator_type() -> OrchestratorType:
    """Determine the default orchestrator type based on optimizer names.

    Args:
        optimizer_names: List of optimizer names being used

    Returns:
        Default orchestrator
    """
    return OrchestratorType.EQUAL_WEIGHT
