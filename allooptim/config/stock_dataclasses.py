"""Data classes for stock universe and financial data structures.

This module defines immutable data structures for representing stock information,
Wikipedia page view data, and financial metrics. These dataclasses provide
type-safe containers for stock universe management and alternative data sources.

Key components:
- StockData: Individual stock information with Wikipedia views and prices
- StockUniverse: Collections of stocks with filtering capabilities
- Immutable dataclasses for data integrity
- Support for multi-language Wikipedia view data
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StockData:
    """Data structure for stock information including Wikipedia views and price."""

    symbol: str
    company_name: str
    wikipedia_name: str
    wiki_views_fr: float = np.nan
    wiki_views_de: float = np.nan
    wiki_views_en: float = np.nan
    wiki_views: float = np.nan
    stock_price: float = np.nan


@dataclass(frozen=True)
class StockUniverse:
    """Data structure for stock universe information."""

    symbol: str
    company_name: str
    wikipedia_name: str
    industry: str
