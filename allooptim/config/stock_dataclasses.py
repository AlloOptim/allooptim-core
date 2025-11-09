"""Data classes for stock universe and financial data structures.

This module defines immutable data structures for representing stock information,
Wikipedia page view data, and financial metrics. These dataclasses provide
type-safe containers for stock universe management and alternative data sources.

Key components:
- StockUniverse: Collections of stocks with filtering capabilities
- Immutable dataclasses for data integrity
- Support for multi-language Wikipedia view data
"""

from dataclasses import dataclass

from typing import Optional


@dataclass(frozen=True)
class StockUniverse:
    """Data structure for stock universe information."""

    symbol: str
    company_name: Optional[str] = None
    wikipedia_name: Optional[str] = None
    industry: Optional[str] = None
