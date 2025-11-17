"""Enhanced Optimizer Factory with Configuration Support.

Provides factory functions for creating optimizers with custom configurations.
Supports both default configs and custom parameter overrides.
"""

import logging
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

from allooptim.optimizer.optimizer_config_registry import (
    NAME_TO_OPTIMIZER_CLASS,
    get_all_optimizer_configs,
    get_optimizer_names_without_configs,
    validate_optimizer_config,
)
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

if TYPE_CHECKING:
    from allooptim.config.optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


def get_optimizer_by_names_with_configs(
    configs: List["OptimizerConfig"]  # Changed from Dict
) -> List[AbstractOptimizer]:
    """Create optimizers from OptimizerConfig objects.
    
    Args:
        configs: List of OptimizerConfig objects with class names and display names
        
    Returns:
        List of configured optimizer instances
    """
    optimizers = []
    
    for opt_config in configs:
        # Use opt_config.name for class lookup
        optimizer_class = NAME_TO_OPTIMIZER_CLASS.get(opt_config.name)
        if optimizer_class is None:
            available = list(NAME_TO_OPTIMIZER_CLASS.keys())
            raise ValueError(
                f"Unknown optimizer '{opt_config.name}'. "
                f"Available optimizers: {available}"
            )
        
        # Create with config
        if opt_config.config:
            config = validate_optimizer_config(opt_config.name, opt_config.config)
            optimizer = optimizer_class(config=config)
            logger.info(
                f"Created {opt_config.name} as '{opt_config.display_name}' "
                f"with config: {opt_config.config}"
            )
        else:
            optimizer = optimizer_class()
            logger.debug(
                f"Created {opt_config.name} as '{opt_config.display_name}' "
                f"with default config"
            )
        
        # Store display_name for later use
        optimizer._display_name = opt_config.display_name
        
        optimizers.append(optimizer)
    
    return optimizers


def get_optimizer_by_names(names: List[str]) -> List[AbstractOptimizer]:
    """Backward-compatible factory function for creating optimizers with default configs.

    This maintains compatibility with existing code.
    """
    # Convert names to OptimizerConfig objects
    configs = [OptimizerConfig(name=name) for name in names]
    return get_optimizer_by_names_with_configs(configs)


def create_optimizer_config_template(optimizer_name: str) -> Dict[str, Any]:
    """Create a template config dictionary for an optimizer.

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
    """Get information about all available optimizer configurations.

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


def validate_optimizer_config_list(configs: List["OptimizerConfig"]) -> List[str]:
    """Validate a list of optimizer configs and return any errors.

    Args:
        configs: List of OptimizerConfig instances

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    for config in configs:
        try:
            validate_optimizer_config(config.name, config.config)
        except ValueError as e:
            errors.append(str(e))

    return errors
