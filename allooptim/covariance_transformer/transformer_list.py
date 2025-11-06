"""Registry and factory functions for covariance transformers.

This module provides a centralized registry of all available covariance
transformers and utility functions for creating and managing transformer
instances. It serves as the main entry point for accessing covariance
transformation capabilities.

Key features:
- Complete registry of all transformer implementations
- Factory functions for transformer creation
- Name-based transformer lookup and instantiation
- Integration with configuration systems
- Comprehensive transformer catalog
"""

import logging

from allooptim.covariance_transformer.covariance_autoencoder import (
    AutoencoderCovarianceTransformer,
)
from allooptim.covariance_transformer.covariance_transformer import (
    DeNoiserCovarianceTransformer,
    DetoneCovarianceTransformer,
    EllipticEnvelopeShrinkageCovarianceTransformer,
    EmpiricalCovarianceTransformer,
    LedoitWolfCovarianceTransformer,
    MarcenkoPasturCovarianceTransformer,
    OracleCovarianceTransformer,
    PCACovarianceTransformer,
    SimpleShrinkageCovarianceTransformer,
)
from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)

logger = logging.getLogger(__name__)

TRANSFORMER_LIST: list[type[AbstractCovarianceTransformer]] = [
    AutoencoderCovarianceTransformer,
    SimpleShrinkageCovarianceTransformer,
    EllipticEnvelopeShrinkageCovarianceTransformer,
    EmpiricalCovarianceTransformer,
    OracleCovarianceTransformer,
    LedoitWolfCovarianceTransformer,
    MarcenkoPasturCovarianceTransformer,
    PCACovarianceTransformer,
    DeNoiserCovarianceTransformer,
    DetoneCovarianceTransformer,
]


def get_all_transformers() -> list[AbstractCovarianceTransformer]:
    """Get instances of all available covariance transformers."""
    return [transformer() for transformer in TRANSFORMER_LIST]


def get_transformer_by_names(names: list[str]) -> list[AbstractCovarianceTransformer]:
    """Retrieve transformer instances by their names."""
    all_transformers = get_all_transformers()
    name_to_transformer = {transformer.name: transformer for transformer in all_transformers}

    for name in names:
        if name not in name_to_transformer:
            logger.warning(
                f"Transformer '{name}' is not recognized. Available transformers: {list(name_to_transformer.keys())}"
            )

    selected_transformers = [transformer for name, transformer in name_to_transformer.items() if name in names]

    if len(selected_transformers) == 0:
        logger.error("No valid transformers found for the provided names.")

    return selected_transformers
