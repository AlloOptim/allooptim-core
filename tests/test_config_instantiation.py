"""Test that all config classes can be instantiated with default values.

This test ensures that all configuration classes in the allooptim.config module
can be instantiated without errors using their default values. This validates
that the config system is properly set up and that all dependencies are
correctly defined.
"""

from datetime import datetime

from allooptim.config.a2a_config import A2AConfig
from allooptim.config.backtest_config import BacktestConfig
from allooptim.config.optimizer_config import OptimizerConfig


class TestConfigInstantiation:
    """Test instantiation of all config classes with default values."""

    def test_a2a_config_instantiation(self):
        """Test that A2AConfig can be instantiated with defaults."""
        config = A2AConfig()
        assert isinstance(config, A2AConfig)
        # Validate that key fields have reasonable defaults
        assert config.n_simulations > 0
        assert config.n_data_observations > 0
        assert config.n_particles > 0

    def test_backtest_config_instantiation(self):
        """Test that BacktestConfig can be instantiated with defaults."""
        # Provide required fields
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        config = BacktestConfig(start_date=start_date, end_date=end_date)
        assert isinstance(config, BacktestConfig)
        # Validate that key fields exist and are properly typed
        assert hasattr(config, 'optimizer_configs')
        assert isinstance(config.optimizer_configs, list)
        assert config.start_date == start_date
        assert config.end_date == end_date

    def test_optimizer_config_instantiation(self):
        """Test that OptimizerConfig can be instantiated with defaults."""
        # Provide required name field with a valid optimizer name
        config = OptimizerConfig(name="MaxSharpe")
        assert isinstance(config, OptimizerConfig)
        # Validate that key fields have reasonable defaults
        assert config.name == "MaxSharpe"
        assert config.display_name == "MaxSharpe"  # Should be auto-generated from name