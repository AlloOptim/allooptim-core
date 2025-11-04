"""
Test configuration for allooptim-core tests.
"""

from pathlib import Path

import pandas as pd
import pytest

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
