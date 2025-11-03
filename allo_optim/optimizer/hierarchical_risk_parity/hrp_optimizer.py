import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pypfopt.hierarchical_portfolio import HRPOpt

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allo_optim.optimizer.allocation_metric import LMoments
from allo_optim.optimizer.asset_name_utils import create_weights_series, validate_asset_names
from allo_optim.optimizer.optimizer_interface import AbstractOptimizer

logger = logging.getLogger(__name__)

# Constants for portfolio weight validation
PORTFOLIO_WEIGHT_SUM_UPPER_TOLERANCE = 1.001
PORTFOLIO_WEIGHT_SUM_LOWER_TOLERANCE = 0.999


class HRPOptimizerConfig(BaseModel):
	model_config = DEFAULT_PYDANTIC_CONFIG

	# HRP typically doesn't need many parameters, but adding for consistency
	# Could add linkage method, distance metric parameters here if needed
	pass


class HRPOptimizer(AbstractOptimizer):
	def __init__(self) -> None:
		self.config = HRPOptimizerConfig()

	def allocate(
		self,
		ds_mu: pd.Series,
		df_cov: pd.DataFrame,
		df_prices: Optional[pd.DataFrame] = None,
		time: Optional[datetime] = None,
		l_moments: Optional[LMoments] = None,
	) -> pd.Series:
		# Validate asset names consistency
		validate_asset_names(ds_mu, df_cov)
		asset_names = ds_mu.index.tolist()

		hrp = HRPOpt(cov_matrix=df_cov)

		weights_dict = hrp.optimize()
		weights_array = np.array([weights_dict[key] for key in asset_names])

		if (
			weights_array.sum() > PORTFOLIO_WEIGHT_SUM_UPPER_TOLERANCE
			or weights_array.sum() < PORTFOLIO_WEIGHT_SUM_LOWER_TOLERANCE
		):
			logger.error("Portfolio allocations don't sum to 1.")
			return create_weights_series(np.zeros(len(asset_names)), asset_names)

		return create_weights_series(weights_array, asset_names)

	@property
	def name(self) -> str:
		return "HRP"
