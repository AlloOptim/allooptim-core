"""
Test configuration for allooptim-core tests.
"""

import pandas as pd
import pytest


@pytest.fixture
def prices_df():
    """Load sample price data from CSV file."""
    df = pd.read_csv("tests/stock_prices.csv", index_col="date", parse_dates=True)
    return df
