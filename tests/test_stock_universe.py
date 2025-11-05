from dataclasses import fields

import pytest
from allooptim.config.stock_universe import (
    INDUSTRY_NAMES,
    StockUniverse,
    get_sp500_companies_0,
    get_sp500_companies_1,
    get_sp500_companies_2,
    get_sp500_companies_3,
    get_sp500_companies_4,
    get_sp500_companies_5,
    list_major_sp500_stocks,
    list_of_dax_stocks,
)


@pytest.fixture
def all_stocks():
    """Fixture that combines all stock lists for comprehensive testing"""
    dax_stocks = list_of_dax_stocks()
    major_sp500 = list_major_sp500_stocks()
    sp500_0 = get_sp500_companies_0()
    sp500_1 = get_sp500_companies_1()
    sp500_2 = get_sp500_companies_2()
    sp500_3 = get_sp500_companies_3()
    sp500_4 = get_sp500_companies_4()
    sp500_5 = get_sp500_companies_5()

    return dax_stocks + major_sp500 + sp500_0 + sp500_1 + sp500_2 + sp500_3 + sp500_4 + sp500_5


def test_stock_dataclass_structure(all_stocks):
    """Test that each stock object is a valid StockUniverse dataclass with exactly the defined fields"""
    expected_fields = {field.name for field in fields(StockUniverse)}

    for stock in all_stocks:
        # Test that it's an instance of StockUniverse
        assert isinstance(stock, StockUniverse), f"Stock {stock} is not an instance of StockUniverse"

        # Test that it has exactly the expected fields
        actual_fields = set(stock.__dict__.keys())
        assert (
            expected_fields == actual_fields
        ), f"Stock {stock.symbol} has incorrect fields. Expected {expected_fields}, got {actual_fields}"

        # Test that no field is None
        for field_name in expected_fields:
            assert getattr(stock, field_name) is not None, f"Stock {stock.symbol} has None value for {field_name}"


def test_industry_names(all_stocks):
    """Test that all industry names are from the predefined INDUSTRY_NAMES list"""
    for stock in all_stocks:
        assert stock.industry in INDUSTRY_NAMES, f"Stock {stock.symbol} has invalid industry name: {stock.industry}"
