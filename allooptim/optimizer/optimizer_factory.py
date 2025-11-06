"""
Enhanced Optimizer Factory with Configuration Support

Provides factory functions for creating optimizers with custom configurations.
Supports both default configs and custom parameter overrides.
"""

import logging
from typing import List, Dict, Any, Optional
from allooptim.optimizer.optimizer_config_registry import get_all_optimizer_configs
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from allooptim.optimizer.optimizer_config_registry import (
    get_optimizer_class,
    validate_optimizer_config,
    get_registered_optimizer_names,
    NAME_TO_OPTIMIZER_CLASS,
)
from allooptim.optimizer.optimizer_config_registry import get_optimizer_names_without_configs

logger = logging.getLogger(__name__)


class OptimizerConfig:
    """Wrapper for optimizer configuration parameters."""

    def __init__(self, optimizer_name: str, params: Optional[Dict[str, Any]] = None):
        self.optimizer_name = optimizer_name
        self.params = params or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        """Create from dictionary representation."""
        return cls(optimizer_name=data["optimizer_name"], params=data.get("params", {}))


def get_optimizer_by_names_with_configs(
    names: List[str], optimizer_configs: Optional[List[OptimizerConfig]] = None
) -> List[AbstractOptimizer]:
    """
    Enhanced factory function that creates optimizers with custom configurations.

    Args:
        names: List of optimizer names to create
        optimizer_configs: Optional list of custom configurations

    Returns:
        List of configured optimizer instances

    Raises:
        ValueError: If optimizer name is unknown or config validation fails
    """
    if optimizer_configs is None:
        optimizer_configs = []

    # Create config map for quick lookup
    config_map = {config.optimizer_name: config.params for config in optimizer_configs}

    optimizers = []
    for name in names:
        optimizer_class = NAME_TO_OPTIMIZER_CLASS.get(name)
        if optimizer_class is None:
            available = list(NAME_TO_OPTIMIZER_CLASS.keys())
            raise ValueError(f"Unknown optimizer '{name}'. Available optimizers: {available}")

        # Check if custom config provided
        params = config_map.get(name, {})

        if params:
            # Validate and create config
            try:
                config = validate_optimizer_config(name, params)
                optimizer = optimizer_class(config=config)
                logger.info(f"Created {name} with custom config: {params}")
            except ValueError as e:
                logger.error(f"Failed to create {name} with config {params}: {e}")
                raise
        else:
            # Use default config
            optimizer = optimizer_class()
            logger.debug(f"Created {name} with default config")

        optimizers.append(optimizer)

    return optimizers


def get_optimizer_by_names(names: List[str]) -> List[AbstractOptimizer]:
    """
    Backward-compatible factory function for creating optimizers with default configs.

    This maintains compatibility with existing code.
    """
    return get_optimizer_by_names_with_configs(names, [])


def create_optimizer_config_template(optimizer_name: str) -> Dict[str, Any]:
    """
    Create a template config dictionary for an optimizer.

    Args:
        optimizer_name: Name of the optimizer

    Returns:
        Dictionary with optimizer_name and empty params dict

    Raises:
        ValueError: If optimizer is not registered
    """
    if optimizer_name not in NAME_TO_OPTIMIZER_CLASS:
        available = list(NAME_TO_OPTIMIZER_CLASS.keys())
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available optimizers: {available}")

    return {"optimizer_name": optimizer_name, "params": {}}


def get_available_optimizer_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available optimizer configurations.

    Returns:
        Dictionary mapping optimizer names to their config schema info
    """

    result = {}
    configs = get_all_optimizer_configs()

    for name, config_class in configs.items():
        schema = config_class.model_json_schema()
        result[name] = {
            "has_config": True,
            "schema": schema,
            "required_fields": schema.get("required", []),
            "properties": schema.get("properties", {}),
        }

    # Add optimizers without configs
    for name in get_optimizer_names_without_configs():
        result[name] = {"has_config": False, "schema": None, "note": "This optimizer uses default configuration only"}

    return result


def validate_optimizer_config_list(configs: List[OptimizerConfig]) -> List[str]:
    """
    Validate a list of optimizer configs and return any errors.

    Args:
        configs: List of OptimizerConfig instances

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    for config in configs:
        try:
            validate_optimizer_config(config.optimizer_name, config.params)
        except ValueError as e:
            errors.append(str(e))

    return errors
