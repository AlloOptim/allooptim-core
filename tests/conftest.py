"""
Test configuration for allooptim-core tests.
"""

from pathlib import Path

import pandas as pd
import pytest

from allo_optim.allocation_to_allocators.allocation_orchestrator import (
    AllocationOrchestratorConfig,
    OrchestrationType,
)

TESTS_DIRECTORY = Path(__file__).parent.resolve()


@pytest.fixture
def prices_df():
    """Load sample price data from CSV file."""
    df = pd.read_csv(TESTS_DIRECTORY / "resources" / "stock_prices.csv", index_col="date", parse_dates=True)
    return df


@pytest.fixture
def wikipedia_test_db_path() -> Path:
    """Path to the test Wikipedia database."""
    return TESTS_DIRECTORY / "resources" / "wikipedia" / "test_wikipedia.db"


FAST_TEST_PARTICLES = 2
FAST_TEST_ITERATIONS = 2
FAST_TEST_OBSERVATIONS = 2
USE_WIKI_DATABASE = True
N_HISTORICAL_DAYS = 30


@pytest.fixture
def fast_a2a_config():
    """Create fast A2A config for testing."""

    fast_a2a_config = AllocationOrchestratorConfig(
        orchestration_type=OrchestrationType.EQUAL,
        n_particles=FAST_TEST_PARTICLES,
        n_particle_swarm_iterations=FAST_TEST_ITERATIONS,
        n_data_observations=FAST_TEST_OBSERVATIONS,
        use_wiki_database=USE_WIKI_DATABASE,
        n_historical_days=N_HISTORICAL_DAYS,
    )

    return fast_a2a_config
