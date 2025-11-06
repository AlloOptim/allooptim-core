"""Optimizer Configuration Registry

Auto-discovers and registers optimizer configuration classes from the OPTIMIZER_LIST.
Provides type-safe access to optimizer configs for API and factory use.
"""

import logging
from typing import Any, Dict, Optional, Type, get_type_hints

from pydantic import BaseModel

from allooptim.optimizer.optimizer_list import OPTIMIZER_LIST, get_all_optimizer_names

logger = logging.getLogger(__name__)

# Global registry mapping optimizer names to their config classes
OPTIMIZER_CONFIG_REGISTRY: Dict[str, Type[BaseModel]] = {}

# Mapping from optimizer names to their classes (for factory use)
NAME_TO_OPTIMIZER_CLASS: Dict[str, Type] = {}


def register_optimizer_configs():
    """Automatically populate registry from OPTIMIZER_LIST.

    This function introspects each optimizer class to extract its config class
    from the __init__ method type hints.
    """
    for optimizer_class in OPTIMIZER_LIST:
        try:
            # Create instance to get the name
            instance = optimizer_class()
            name = instance.name

            # Store the class mapping for factory use
            NAME_TO_OPTIMIZER_CLASS[name] = optimizer_class

            # Extract config class from type hints
            hints = get_type_hints(optimizer_class.__init__)
            if "config" in hints:
                config_type = hints["config"]
                # Handle Optional[ConfigClass] -> extract ConfigClass
                if hasattr(config_type, "__args__"):
                    actual_config_class = config_type.__args__[0]
                    if issubclass(actual_config_class, BaseModel):
                        OPTIMIZER_CONFIG_REGISTRY[name] = actual_config_class
                        logger.debug(f"Registered config for {name}: {actual_config_class.__name__}")
                    else:
                        logger.debug(f"Config for {name} is not a Pydantic BaseModel: {actual_config_class}")
                else:
                    logger.debug(f"Config for {name} is not Optional: {config_type}")

        except Exception as e:
            logger.warning(f"Failed to register config for {optimizer_class.__name__}: {e}")


def get_optimizer_config_class(optimizer_name: str) -> Optional[Type[BaseModel]]:
    """Get the config class for a specific optimizer."""
    return OPTIMIZER_CONFIG_REGISTRY.get(optimizer_name)


def get_all_optimizer_configs() -> Dict[str, Type[BaseModel]]:
    """Get all registered optimizer config classes."""
    return OPTIMIZER_CONFIG_REGISTRY.copy()


def get_optimizer_class(optimizer_name: str) -> Optional[Type]:
    """Get the optimizer class for a specific optimizer name."""
    return NAME_TO_OPTIMIZER_CLASS.get(optimizer_name)


def validate_optimizer_config(optimizer_name: str, config_params: Dict[str, Any]) -> BaseModel:
    """Validate config parameters against the registered config class.

    Args:
        optimizer_name: Name of the optimizer
        config_params: Dictionary of config parameters

    Returns:
        Validated Pydantic config instance

    Raises:
        ValueError: If optimizer not found or config validation fails
    """
    if optimizer_name not in OPTIMIZER_CONFIG_REGISTRY:
        available = list(OPTIMIZER_CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available optimizers: {available}")

    config_class = OPTIMIZER_CONFIG_REGISTRY[optimizer_name]
    try:
        return config_class(**config_params)
    except Exception as e:
        raise ValueError(f"Invalid config for {optimizer_name}: {str(e)}")


def get_registered_optimizer_names() -> list[str]:
    """Get list of optimizer names that have registered configs."""
    return list(OPTIMIZER_CONFIG_REGISTRY.keys())


def get_optimizer_names_without_configs() -> list[str]:
    """Get list of optimizer names that don't have registered configs."""
    all_names = set(get_all_optimizer_names())
    registered_names = set(OPTIMIZER_CONFIG_REGISTRY.keys())
    return list(all_names - registered_names)


def get_optimizer_config_schema(optimizer_name: str) -> Dict[str, Any]:
    """Get the JSON schema for an optimizer's configuration.

    Args:
        optimizer_name: Name of the optimizer

    Returns:
        JSON schema dict for the optimizer config

    Raises:
        ValueError: If optimizer is not found or has no config class
    """
    config_class = get_optimizer_config_class(optimizer_name)
    if config_class is None:
        raise ValueError(f"No config class found for optimizer: {optimizer_name}")

    return config_class.model_json_schema()


# Auto-register on module import
register_optimizer_configs()

# Log registration summary
logger.debug(f"Registered {len(OPTIMIZER_CONFIG_REGISTRY)} optimizer configs")
if unregistered := get_optimizer_names_without_configs():
    logger.warning(f"Optimizers without configs: {unregistered}")
