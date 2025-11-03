"""
Test configuration and utilities for Phase 7 - Testing Strategy Implementation

Shared test utilities, fixtures, and configuration for all test suites.
"""

import os
import shutil
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import psutil
import pytest

from common.database.database_service import IDatabaseService, create_database_service


class EnvironmentType(Enum):
    TESTING = "testing"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class DatabaseType(Enum):
    SQLITE = "sqlite"
    MEMORY = ":memory:"


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def test_db_path(test_data_dir: Path) -> Generator[str, None, None]:
    """Create a test database path"""
    db_path = test_data_dir / "test_lumibot.db"
    yield str(db_path)


@pytest.fixture
def test_db_interface(test_db_path: str) -> Generator[IDatabaseService, None, None]:
    """Create a test database interface with initialized schema"""
    from common.database.database_service import DatabaseService

    db = DatabaseService(db_path=test_db_path)
    # Use connect() instead of initialize_database() to follow actual API
    db.connect()
    db._ensure_tables()
    db.disconnect()
    yield db
    try:
        db.close()
    except AttributeError:
        pass  # DatabaseService doesn't have close method


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Sample market data for testing"""
    np.random.seed(42)  # For reproducible tests

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    dates = pd.date_range("2025-01-01", periods=100, freq="D")

    data: Dict[str, np.ndarray] = {}
    for symbol in symbols:
        # Generate realistic price data with trend and volatility
        base_price = {
            "AAPL": 150,
            "GOOGL": 2800,
            "MSFT": 300,
            "AMZN": 3300,
            "TSLA": 250,
        }[symbol]
        trend = np.random.normal(0.0001, 0.02, len(dates))  # Small daily drift
        volatility = np.random.normal(0, 0.03, len(dates))  # Daily volatility

        prices = base_price * np.exp(np.cumsum(trend + volatility))
        data[symbol] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_portfolio_data() -> pd.DataFrame:
    """Sample portfolio data for testing"""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN"],
            "allocation": [0.3, 0.3, 0.2, 0.2],
            "current_price": [155.0, 2850.0, 310.0, 3350.0],
            "shares": [100, 5, 50, 3],
            "value": [15500.0, 14250.0, 15500.0, 10050.0],
            "pnl": [500.0, 250.0, -200.0, 150.0],
            "avg_cost": [150.0, 2800.0, 320.0, 3300.0],
        }
    )


@pytest.fixture
def sample_trading_history() -> pd.DataFrame:
    """Sample trading history for testing"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="H"),
            "symbol": ["AAPL"] * 8 + ["GOOGL"] * 6 + ["MSFT"] * 6,
            "action": ["BUY", "SELL"] * 10,
            "quantity": [10, 5] * 10,
            "price": [155.0, 157.0] * 10,
            "value": [1550.0, 785.0] * 10,
            "commission": [1.0] * 20,
        }
    )


@pytest.fixture
def mock_alpaca_credentials() -> Dict[str, str]:
    """Mock Alpaca API credentials for testing"""
    return {
        "ALPACA_API_KEY": "test_key_12345",
        "ALPACA_API_SECRET": "test_secret_67890",
        "ALPACA_IS_PAPER": "true",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    }


@pytest.fixture
def test_env_vars(
    mock_alpaca_credentials: Dict[str, str],
) -> Generator[Dict[str, str], None, None]:
    """Set up test environment variables"""
    original_env = dict(os.environ)

    # Set test environment variables
    test_vars = {
        "DATABASE_URL": "sqlite:///test.db",
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "testing",
        **mock_alpaca_credentials,
    }

    os.environ.update(test_vars)
    yield test_vars

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "docker: marks tests that require Docker")
    config.addinivalue_line("markers", "database: marks tests that require database access")


def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True) -> None:
    """Assert that two DataFrames are equal with detailed error messages"""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    except AssertionError as e:
        print(f"DataFrame comparison failed:\n{e}")
        print(f"DF1 shape: {df1.shape}, DF2 shape: {df2.shape}")
        print(f"DF1 columns: {list(df1.columns)}")
        print(f"DF2 columns: {list(df2.columns)}")
        raise


def assert_series_equal(s1: pd.Series, s2: pd.Series, check_dtype: bool = True) -> None:
    """Assert that two Series are equal with detailed error messages"""
    try:
        pd.testing.assert_series_equal(s1, s2, check_dtype=check_dtype)
    except AssertionError as e:
        print(f"Series comparison failed:\n{e}")
        print(f"S1 length: {len(s1)}, S2 length: {len(s2)}")
        print(f"S1 index: {s1.index}")
        print(f"S2 index: {s2.index}")
        raise


def create_test_database_with_data(db_path: str, num_records: int = 100) -> str:
    """Create a test database populated with sample data"""
    from common.database.database_service import DatabaseService

    db = DatabaseService(db_path=db_path)
    db.connect()
    db._ensure_tables()

    np.random.seed(42)  # For reproducible data

    # Generate sample trades
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    actions = ["BUY", "SELL"]

    cursor = db.connection.cursor()

    for i in range(num_records):
        symbol = np.random.choice(symbols)
        action = np.random.choice(actions)
        quantity = np.random.randint(1, 100)
        price = np.random.uniform(100, 5000)
        timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(days=np.random.randint(0, 365))

        cursor.execute(
            """
            INSERT INTO trades (symbol, action, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (symbol, action, quantity, price, timestamp.isoformat()),
        )

    # Generate sample positions
    for symbol in symbols:
        quantity = np.random.randint(0, 200)
        if quantity > 0:
            avg_price = np.random.uniform(100, 5000)
            current_value = quantity * np.random.uniform(avg_price * 0.8, avg_price * 1.2)

            cursor.execute(
                """
                INSERT INTO positions (symbol, quantity, average_price, current_value)
                VALUES (?, ?, ?, ?)
            """,
                (symbol, quantity, avg_price, current_value),
            )

    db.connection.commit()
    db.close()

    return db_path


def cleanup_test_files(*file_paths: str) -> None:
    """Clean up test files and directories"""
    for path in file_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                path_obj.unlink()
            elif path_obj.is_dir():
                shutil.rmtree(path_obj)


def time_function_execution(func: callable, *args: Any, **kwargs: Any) -> tuple:
    """Time function execution and return result with timing"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time
    return result, execution_time


def profile_memory_usage(func: callable, *args: Any, **kwargs: Any) -> tuple:
    """Profile memory usage of function execution"""
    try:
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory

        return result, memory_delta

    except ImportError:
        # psutil not available, return result without memory profiling
        result = func(*args, **kwargs)
        return result, None
