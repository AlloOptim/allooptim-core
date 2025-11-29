"""Unit tests for the optimizer configuration interface.

Tests the optimizer configuration system for type safety, validation,
and integration with factory and backtest components.
"""

from datetime import datetime

import pytest

from allooptim.config.a2a_manager_config import A2AManagerConfig
from allooptim.config.backtest_config import BacktestConfig
from allooptim.optimizer.optimizer_config import OptimizerConfig
from allooptim.optimizer.optimizer_config_registry import (
    get_all_optimizer_configs,
    get_optimizer_config_class,
    get_registered_optimizer_names,
    validate_optimizer_config,
)


class TestOptimizerConfigRegistry:
    """Test the optimizer configuration registry functionality."""

    def test_get_registered_optimizer_names_returns_list(self):
        """Test that get_registered_optimizer_names returns a list."""
        names = get_registered_optimizer_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_optimizer_config_class_returns_pydantic_model(self):
        """Test that get_optimizer_config_class returns a Pydantic BaseModel."""
        from pydantic import BaseModel

        names = get_registered_optimizer_names()
        for name in names[:3]:  # Test first 3 to avoid too many iterations
            config_class = get_optimizer_config_class(name)
            assert config_class is not None
            assert issubclass(config_class, BaseModel)

    def test_get_all_optimizer_configs_returns_dict(self):
        """Test that get_all_optimizer_configs returns a dictionary."""
        configs = get_all_optimizer_configs()
        assert isinstance(configs, dict)
        assert len(configs) > 0

    def test_validate_optimizer_config_accepts_valid_config(self):
        """Test that validate_optimizer_config accepts valid configurations."""
        # Test with a simple optimizer that has minimal config
        config = validate_optimizer_config("HRPOptimizer", {})
        assert config is not None

    def test_validate_optimizer_config_rejects_invalid_optimizer(self):
        """Test that validate_optimizer_config rejects unknown optimizers."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            validate_optimizer_config("NonExistentOptimizer", {})

    def test_validate_optimizer_config_rejects_invalid_config(self):
        """Test that validate_optimizer_config rejects invalid configurations."""
        # Try to pass an invalid parameter to an optimizer
        with pytest.raises(ValueError, match="Invalid config"):
            validate_optimizer_config("CMAMeanVariance", {"invalid_param": "value"})


class TestOptimizerFactoryIntegration:
    """Test integration with optimizer factory."""

    def test_mixed_optimizer_specs_validation(self):
        """Test that mixed string/dict optimizer specs can be validated."""
        optimizer_specs = [
            "HRPOptimizer",  # String spec
            {"name": "CMAMeanVariance", "config": {"budget": 1000, "risk_aversion": 3.0}},  # Dict spec
            {"name": "MeanVarianceParticleSwarmOptimizer", "config": {"n_particles": 500, "n_iters": 100}},  # Dict spec
        ]

        # Validate each spec
        for spec in optimizer_specs:
            if isinstance(spec, str):
                # String specs should be registered optimizers
                registered_names = get_registered_optimizer_names()
                assert spec in registered_names, f"Unknown optimizer: {spec}"
            else:
                # Dict specs should have valid configs
                assert "name" in spec
                assert "config" in spec
                validate_optimizer_config(spec["name"], spec["config"])

    def test_optimizer_specs_with_invalid_string_name(self):
        """Test that invalid string optimizer names are rejected."""
        optimizer_specs = ["InvalidOptimizerName"]

        with pytest.raises(AssertionError, match="Unknown optimizer"):
            for spec in optimizer_specs:
                if isinstance(spec, str):
                    registered_names = get_registered_optimizer_names()
                    assert spec in registered_names, f"Unknown optimizer: {spec}"

    def test_optimizer_specs_with_invalid_config(self):
        """Test that dict specs with invalid configs are rejected."""
        optimizer_specs = [{"name": "CMAMeanVariance", "config": {"invalid_param": 123}}]

        with pytest.raises(ValueError, match="Invalid config"):
            for spec in optimizer_specs:
                if isinstance(spec, dict):
                    validate_optimizer_config(spec["name"], spec["config"])


class TestA2AManagerConfigIntegration:
    """Test integration with BacktestConfig."""

    def test_a2a_manager_config_accepts_mixed_optimizer_configs(self):
        """Test that A2AManagerConfig accepts mixed optimizer configurations."""
        EXPECTED_OPTIMIZER_COUNT = 3

        optimizer_configs = [
            "HRPOptimizer",  # String
            OptimizerConfig(name="CMAMeanVariance", config={"budget": 2000}),  # OptimizerConfig with config
            OptimizerConfig(name="MeanVarianceParticleSwarmOptimizer", config={}),  # OptimizerConfig with empty config
        ]

        a2a_manager_config = A2AManagerConfig(optimizer_configs=optimizer_configs)

        assert len(a2a_manager_config.optimizer_configs) == EXPECTED_OPTIMIZER_COUNT

    def test_a2a_manager_config_accepts_optimizer_configs(self):
        """Test that A2AManagerConfig accepts optimizer configurations."""
        # Test with valid optimizer names
        optimizer_configs = ["HRPOptimizer", "NaiveOptimizer"]
        a2a_manager_config = A2AManagerConfig(optimizer_configs=optimizer_configs)
        assert len(a2a_manager_config.optimizer_configs) == 2

    def test_backtest_config_validates_config_parameters(self):
        """Test that BacktestConfig validates config parameters."""
        # Test with invalid config parameters
        with pytest.raises(ValueError, match="Invalid config"):
            BacktestConfig(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 1, 1),
                optimizer_configs=[OptimizerConfig(name="CMAMeanVariance", config={"invalid_param": "value"})],
            )

    def test_a2a_manager_config_optimizer_names_property(self):
        """Test that A2AManagerConfig.optimizer_names property works correctly."""
        EXPECTED_OPTIMIZER_COUNT = 3

        optimizer_configs = [
            "HRPOptimizer",
            OptimizerConfig(name="CMAMeanVariance", config={}),
            OptimizerConfig(name="MeanVarianceParticleSwarmOptimizer", config={}),
        ]

        a2a_manager_config = A2AManagerConfig(optimizer_configs=optimizer_configs)

        optimizer_names = a2a_manager_config.optimizer_names
        assert isinstance(optimizer_names, list)
        assert len(optimizer_names) == EXPECTED_OPTIMIZER_COUNT
        assert "HRPOptimizer" in optimizer_names
        assert "CMAMeanVariance" in optimizer_names
        assert "MeanVarianceParticleSwarmOptimizer" in optimizer_names

    def test_a2a_manager_config_get_optimizer_configs_dict(self):
        """Test that get_optimizer_configs_dict returns correct mapping."""
        EXPECTED_CONFIG_COUNT = 3

        optimizer_configs = [
            "HRPOptimizer",
            OptimizerConfig(name="CMAMeanVariance", config={"budget": 1000}),
            OptimizerConfig(name="MeanVarianceParticleSwarmOptimizer", config={}),
        ]

        a2a_manager_config = A2AManagerConfig(optimizer_configs=optimizer_configs)

        config_dict = a2a_manager_config.get_optimizer_configs_dict()
        assert isinstance(config_dict, dict)
        assert len(config_dict) == EXPECTED_CONFIG_COUNT
        assert "HRPOptimizer" in config_dict
        assert "CMAMeanVariance[budget=1000]" in config_dict
        assert "MeanVarianceParticleSwarmOptimizer" in config_dict
        assert config_dict["CMAMeanVariance[budget=1000]"] == {"budget": 1000}
        assert (
            config_dict["MeanVarianceParticleSwarmOptimizer"] is None
            or config_dict["MeanVarianceParticleSwarmOptimizer"] == {}
        )


class TestConfigurationIntegration:
    """Test integration between different components."""

    def test_registry_and_a2a_manager_config_integration(self):
        """Test that registry and A2AManagerConfig work together."""
        EXPECTED_OPTIMIZER_COUNT = 3

        # Get registered optimizers
        registered_names = get_registered_optimizer_names()

        # Create A2AManagerConfig with some registered optimizers
        test_optimizers = registered_names[:3]  # Use first 3

        a2a_manager_config = A2AManagerConfig(optimizer_configs=test_optimizers)

        # Verify the config was created successfully
        assert len(a2a_manager_config.optimizer_configs) == EXPECTED_OPTIMIZER_COUNT

        # Verify optimizer names match
        assert a2a_manager_config.optimizer_names == test_optimizers

    def test_config_validation_integration(self):
        """Test that config validation works end-to-end."""
        # Test valid config
        valid_config = {"budget": 1000, "risk_aversion": 3.0}
        validated = validate_optimizer_config("CMAMeanVariance", valid_config)
        assert validated is not None

        # Test that same config works in A2AManagerConfig
        a2a_manager_config = A2AManagerConfig(
            optimizer_configs=[OptimizerConfig(name="CMAMeanVariance", config=valid_config)],
        )

        EXPECTED_CONFIG_COUNT = 1
        assert len(a2a_manager_config.optimizer_configs) == EXPECTED_CONFIG_COUNT
        config_dict = a2a_manager_config.get_optimizer_configs_dict()
        assert config_dict["CMAMeanVariance[budget=1000-risk_aversion=3]"] == valid_config
