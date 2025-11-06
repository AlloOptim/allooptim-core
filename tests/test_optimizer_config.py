"""
Unit tests for optimizer configuration registry and validation.

Tests the core optimizer configuration functionality including registry,
validation, and basic config operations.
"""

import pytest
from typing import Dict, Any
from pydantic import BaseModel

from allooptim.optimizer.optimizer_config_registry import (
    get_registered_optimizer_names,
    get_optimizer_config_class,
    get_all_optimizer_configs,
    validate_optimizer_config,
    get_optimizer_config_schema,
    OPTIMIZER_CONFIG_REGISTRY,
)


class TestOptimizerConfigRegistry:
    """Test the optimizer configuration registry functionality."""

    def test_registry_is_populated(self):
        """Test that the registry is populated with optimizer configs."""
        assert len(OPTIMIZER_CONFIG_REGISTRY) > 0
        assert isinstance(OPTIMIZER_CONFIG_REGISTRY, dict)

    def test_get_registered_optimizer_names(self):
        """Test getting list of registered optimizer names."""
        names = get_registered_optimizer_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

    def test_get_optimizer_config_class(self):
        """Test getting config class for registered optimizers."""
        names = get_registered_optimizer_names()

        # Test first few optimizers
        for name in names[:5]:
            config_class = get_optimizer_config_class(name)
            assert config_class is not None
            assert issubclass(config_class, BaseModel)

    def test_get_optimizer_config_class_unknown_optimizer(self):
        """Test getting config class for unknown optimizer returns None."""
        config_class = get_optimizer_config_class("UnknownOptimizer")
        assert config_class is None

    def test_get_all_optimizer_configs(self):
        """Test getting all optimizer config classes."""
        configs = get_all_optimizer_configs()
        assert isinstance(configs, dict)
        assert len(configs) > 0

        # All values should be Pydantic BaseModel classes
        for name, config_class in configs.items():
            assert isinstance(name, str)
            assert issubclass(config_class, BaseModel)

    def test_validate_optimizer_config_valid(self):
        """Test validating valid optimizer configurations."""
        # Test with simple optimizer that accepts empty config
        config = validate_optimizer_config("HRPOptimizer", {})
        assert config is not None
        assert isinstance(config, BaseModel)

    def test_validate_optimizer_config_invalid_optimizer(self):
        """Test validating config for unknown optimizer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            validate_optimizer_config("UnknownOptimizer", {})

    def test_validate_optimizer_config_invalid_params(self):
        """Test validating invalid config parameters raises ValueError."""
        with pytest.raises(ValueError, match="Invalid config"):
            validate_optimizer_config("CMAMeanVariance", {"invalid_param": "value"})

    def test_get_optimizer_config_schema(self):
        """Test getting JSON schema for optimizer configs."""
        schema = get_optimizer_config_schema("HRPOptimizer")
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"

    def test_get_optimizer_config_schema_unknown_optimizer(self):
        """Test getting schema for unknown optimizer raises ValueError."""
        with pytest.raises(ValueError, match="No config class found"):
            get_optimizer_config_schema("UnknownOptimizer")


class TestOptimizerConfigValidation:
    """Test optimizer configuration validation logic."""

    def test_cma_optimizer_configs(self):
        """Test validation of CMA optimizer configurations."""
        cma_configs = {
            "CMAMeanVariance": {
                "budget": 1000,
                "risk_aversion": 3.0,
                "sigma": 0.2,
                "n_popsize": 100,
                "enable_simple_warm_start": True,
                "patience": 50,
            },
            "CMASortino": {
                "budget": 800,
                "risk_aversion": 2.5,
                "sigma": 0.15,
                "n_popsize": 80,
                "target_return_sortino": 0.02,
                "enable_simple_warm_start": False,
                "patience": 40,
            },
        }

        for optimizer_name, config in cma_configs.items():
            validated_config = validate_optimizer_config(optimizer_name, config)
            assert validated_config is not None
            assert isinstance(validated_config, BaseModel)

    def test_pso_optimizer_configs(self):
        """Test validation of PSO optimizer configurations."""
        pso_configs = {
            "PSOMeanVariance": {
                "n_particles": 100,
                "n_iters": 50,
                "risk_aversion": 3.0,
                "c1": 1.5,
                "c2": 1.5,
                "w": 0.7,
                "enable_warm_start": True,
                "ftol": 1e-4,
                "ftol_iter": 10,
            }
        }

        for optimizer_name, config in pso_configs.items():
            validated_config = validate_optimizer_config(optimizer_name, config)
            assert validated_config is not None
            assert isinstance(validated_config, BaseModel)

    def test_simple_optimizer_configs(self):
        """Test validation of simple optimizer configurations."""
        simple_optimizers = ["HRPOptimizer", "RiskParityOptimizer", "NaiveOptimizer", "NCOSharpeOptimizer"]

        for optimizer_name in simple_optimizers:
            # These should accept empty configs
            validated_config = validate_optimizer_config(optimizer_name, {})
            assert validated_config is not None
            assert isinstance(validated_config, BaseModel)

    def test_config_field_types(self):
        """Test that config validation enforces correct field types."""
        # Test integer field
        with pytest.raises(ValueError):
            validate_optimizer_config("CMAMeanVariance", {"budget": "not_an_int"})

        # Test float field
        with pytest.raises(ValueError):
            validate_optimizer_config("CMAMeanVariance", {"risk_aversion": "not_a_float"})

        # Test boolean field
        with pytest.raises(ValueError):
            validate_optimizer_config("CMAMeanVariance", {"enable_simple_warm_start": "not_a_bool"})


class TestOptimizerConfigSchema:
    """Test optimizer configuration schema generation."""

    def test_schema_structure(self):
        """Test that schemas have expected structure."""
        schema = get_optimizer_config_schema("CMAMeanVariance")

        assert "type" in schema
        assert "properties" in schema
        assert isinstance(schema["properties"], dict)

        # Should have expected CMA fields
        expected_fields = ["budget", "risk_aversion", "sigma", "n_popsize"]
        for field in expected_fields:
            assert field in schema["properties"]

    def test_schema_required_fields(self):
        """Test that schemas have properties field."""
        schema = get_optimizer_config_schema("CMAMeanVariance")

        # Not all schemas have required fields if all fields have defaults
        assert "properties" in schema
        # If required exists, it should be a list
        if "required" in schema:
            assert isinstance(schema["required"], list)

    def test_empty_config_schema(self):
        """Test schema for optimizers with no required config."""
        schema = get_optimizer_config_schema("HRPOptimizer")

        assert "properties" in schema
        # Should have empty or minimal properties
        assert isinstance(schema["properties"], dict)
