import logging
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Constants for solver validation
SOLVER_FAILURE_WEIGHT_SUM_THRESHOLD = 1.05


def minimize_with_multistart(
	objective_function: Callable,
	n_assets: int,
	allow_cash: bool,
	previous_best_weights: Optional[np.ndarray],
) -> np.ndarray:
	"""
	Perform optimization with multiple starting points to avoid local minima.

	Tries:
	1. Equal weights (1/n for each asset)
	2. Previous best weights (if available)

	Returns the best result.
	"""
	# Constraint: 0 <= sum(weights) <= 1.0 (allow cash holding)
	if allow_cash:
		constraints = ({"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)},)  # sum <= 1
	else:
		constraints = ({"type": "eq", "fun": lambda x: 1.0 - np.sum(x)},)  # sum == 1

	bounds = [(0, 1) for _ in range(n_assets)]

	best_weights = None
	best_cost = np.inf

	# Starting point 1: Equal weights
	x0_equal = np.ones(n_assets) / n_assets

	res_equal = minimize(
		objective_function,
		x0=x0_equal,
		method="SLSQP",
		constraints=constraints,
		bounds=bounds,
		options={"disp": False},
	)

	if res_equal.success and res_equal.fun < best_cost:
		best_weights = res_equal.x
		best_cost = res_equal.fun

	# Starting point 2: Previous best weights (if available)
	if previous_best_weights is not None and len(previous_best_weights) == n_assets:
		res_prev = minimize(
			objective_function,
			x0=previous_best_weights,
			method="SLSQP",
			constraints=constraints,
			bounds=bounds,
			options={"disp": False},
		)

		if res_prev.success and res_prev.fun < best_cost:
			best_weights = res_prev.x
			best_cost = res_prev.fun

	elif previous_best_weights is not None:
		logger.warning("Previous best weights length does not match number of assets, skipping that start point.")

	# Return best result, or equal weights start if all failed
	if best_weights is None:
		logger.error("All optimization attempts failed, using equal weights fallback")
		best_weights = x0_equal

	weight_sum = np.sum(best_weights)
	if weight_sum > 1.0:
		if weight_sum > SOLVER_FAILURE_WEIGHT_SUM_THRESHOLD:
			logger.error(
				f"Weight sum {weight_sum:.4f} > {SOLVER_FAILURE_WEIGHT_SUM_THRESHOLD}, "
				"likely the solver failed, normalizing weights"
			)
		else:
			logger.debug(f"Weight sum {weight_sum:.4f} > 1.0, normalizing for safety")
		best_weights = best_weights / weight_sum

	return best_weights
