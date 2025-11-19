"""Wikipedia page view correlation-based portfolio allocation.

This package implements portfolio optimization using Wikipedia page view
data as an alternative data source. The approach correlates stock price
movements with Wikipedia page views to identify investment opportunities
based on public interest and attention.

Key features:
- Wikipedia page view data integration
- Correlation analysis between views and stock prices
- Alternative data-driven allocation
- Exponential decay weighting for recency
- SQLite database caching for performance
"""

import logging

logger = logging.getLogger(__name__)
