import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from pydantic import BaseModel

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.config.stock_universe import get_stocks_by_symbols
from allo_optim.optimizer.allocation_metric import (
	LMoments,
)
from allo_optim.optimizer.asset_name_utils import (
	create_weights_series,
	get_asset_names,
	validate_asset_names,
)
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer
from allo_optim.optimizer.wikipedia.allocate_wikipedia import allocate_wikipedia

logger = logging.getLogger(__name__)


class WikipediaOptimizerConfig(BaseModel):
	model_config = DEFAULT_PYDANTIC_CONFIG

	# Wikipedia optimizer doesn't need specific parameters currently


class WikipediaOptimizer(AbstractOptimizer):
	def __init__(self) -> None:
		self.config = WikipediaOptimizerConfig()

	def allocate(
		self,
		ds_mu: pd.Series,
		df_cov: pd.DataFrame,
		df_prices: Optional[pd.DataFrame] = None,
		df_allocations: Optional[pd.DataFrame] = None,
		time: Optional[datetime] = None,
		l_moments: Optional[LMoments] = None,
	) -> pd.Series:
		# Validate inputs
		validate_asset_names(ds_mu, df_cov)
		assert time is not None, "Time parameter must be provided"

		# Ensure time is timezone-aware (required by allocate_wikipedia)
		if time.tzinfo is None:
			time = time.replace(tzinfo=pytz.UTC)
		else:
			# If already timezone-aware, convert to UTC
			time = time.astimezone(pytz.UTC)

		# Get asset names
		asset_names = get_asset_names(mu=ds_mu)
		n_assets = len(asset_names)

		try:
			all_stocks = get_stocks_by_symbols(asset_names)

			# Filter asset_names to only include stocks available in the universe
			available_symbols = {stock.symbol for stock in all_stocks}
			filtered_asset_names = [name for name in asset_names if name in available_symbols]

			if not filtered_asset_names:
				logger.warning("No assets available in stock universe for Wikipedia allocation")
				equal_weight = 1.0 / n_assets
				weights = np.ones(n_assets) * equal_weight
			else:
				allocation_result = allocate_wikipedia(
					all_stocks=all_stocks,
					time_today=time,
					use_sql_database=True,
				)

				if allocation_result.success:
					weights_dict = allocation_result.asset_weights
					# Create weights array for filtered assets
					filtered_weights = np.array([weights_dict[key] for key in filtered_asset_names])

					# Create full weights array with zeros for unavailable assets
					weights = np.zeros(n_assets)
					for i, asset_name in enumerate(asset_names):
						if asset_name in filtered_asset_names:
							idx = filtered_asset_names.index(asset_name)
							weights[i] = filtered_weights[idx]
						# Unavailable assets get 0 weight
				else:
					equal_weight = 1.0 / n_assets
					weights = np.ones(n_assets) * equal_weight

		except Exception as e:
			logger.error(f"Error in Wikipedia allocation: {e}")
			equal_weight = 1.0 / n_assets
			weights = np.ones(n_assets) * equal_weight

		# Return as pandas Series with asset names
		return create_weights_series(weights, asset_names)

	@property
	def name(self) -> str:
		return "WikipediaOptimizer"
