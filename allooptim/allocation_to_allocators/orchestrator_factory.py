"""Orchestrator Factory.

Factory for creating allocation-to-allocators orchestrators based on configuration.
"""

import logging
from enum import Enum
from typing import List, Optional

from allooptim.allocation_to_allocators.a2a_orchestrator import BaseOrchestrator
from allooptim.allocation_to_allocators.equal_weight_orchestrator import (
    CustomWeightOrchestrator,
    EqualWeightOrchestrator,
    MedianWeightOrchestrator,
)
from allooptim.allocation_to_allocators.optimized_orchestrator import (
    OptimizedOrchestrator,
)
from allooptim.allocation_to_allocators.wikipedia_pipeline_orchestrator import (
    WikipediaPipelineOrchestrator,
)
from allooptim.config.a2a_config import A2AConfig
from allooptim.covariance_transformer.transformer_list import get_transformer_by_names
from allooptim.optimizer.optimizer_config import OptimizerConfig
from allooptim.optimizer.optimizer_factory import get_optimizer_by_config

logger = logging.getLogger(__name__)


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
    optimizer_configs: List[OptimizerConfig],
    transformer_names: List[str] = None,
    a2a_config: Optional[A2AConfig] = None,
    **kwargs,
) -> BaseOrchestrator:
    """Factory function to create the appropriate orchestrator based on type.

    Args:
        orchestrator_type: Type of orchestrator to create
        optimizer_configs: List of OptimizerConfig objects
        transformer_names: List of covariance transformer names to use
        a2a_config: A2AConfig for orchestrator configuration
        **kwargs: Additional arguments specific to orchestrator type

    Returns:
        Configured orchestrator instance

    Raises:
        ValueError: If orchestrator_type is not recognized
    """
    # Get optimizers and transformers
    optimizers = get_optimizer_by_config(optimizer_configs)
    transformers = get_transformer_by_names(transformer_names)

    if a2a_config is None:
        logger.info("No A2AConfig provided, using default configuration.")
        a2a_config = A2AConfig()

    if orchestrator_type == OrchestratorType.AUTO:
        orchestrator_type = get_default_orchestrator_type()

    match orchestrator_type:
        case OrchestratorType.EQUAL_WEIGHT:
            return EqualWeightOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                a2a_config=a2a_config,
            )
        case OrchestratorType.MEDIAN_WEIGHT:
            return MedianWeightOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                a2a_config=a2a_config,
            )
        case OrchestratorType.CUSTOM_WEIGHT:
            return CustomWeightOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                a2a_config=a2a_config,
            )

        case OrchestratorType.OPTIMIZED:
            return OptimizedOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                a2a_config=a2a_config,
            )

        case OrchestratorType.WIKIPEDIA_PIPELINE:
            if "n_historical_days" not in kwargs:
                logger.info("n_historical_days not provided, using default of 60.")
                n_historical_days = 60
            else:
                n_historical_days = kwargs["n_historical_days"]

            if "use_wiki_database" not in kwargs:
                logger.info("use_wiki_database not provided, using default of False.")
                use_wiki_database = False
            else:
                use_wiki_database = kwargs["use_wiki_database"]

            if "wiki_database_path" not in kwargs:
                logger.info("wiki_database_path not provided, using default of None.")
                wiki_database_path = None
            else:
                wiki_database_path = kwargs["wiki_database_path"]

            return WikipediaPipelineOrchestrator(
                optimizers=optimizers,
                covariance_transformers=transformers,
                config=a2a_config,
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
