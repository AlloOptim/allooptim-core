from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StockData:
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
    symbol: str
    company_name: str
    wikipedia_name: str
    industry: str
