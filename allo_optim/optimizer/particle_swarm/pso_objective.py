import logging

import numpy as np

from allo_optim.optimizer.allocation_metric import (
	LMoments,
	expected_return_classical,
	expected_return_moments,
)

logger = logging.getLogger(__name__)

# Constants for simulation parameters
SKEW_EVENT_PROBABILITY = 0.05  # 5% chance of skew events
TAIL_EVENT_PROBABILITY = 0.03  # 3% chance of tail events
SKEW_EVENT_MAGNITUDE = 0.05  # Magnitude of skew events


def price_based_objective_function(
	weights: np.ndarray,
	prices: np.ndarray,
	risk_aversion: float,
) -> np.ndarray:
	"""
	Calculate risk-adjusted returns for multiple particles using efficient matrix operations.

	This function computes the mean-variance utility for multiple portfolio weight configurations
	simultaneously, making it ideal for particle swarm optimization and other population-based
	optimization algorithms.

	The utility function used is: U = E[R] - (risk_aversion / 2) * Var[R]
	where E[R] is expected return and Var[R] is portfolio variance.

	WARNING: This metric assumes normal distribution of returns!

	Args:
	    weights: 2D array of shape (n_particles, n_assets) - each row is a particle's portfolio weights
	             Can also be 1D array of shape (n_assets,) for single portfolio evaluation
	    prices: 2D array of shape (n_timesteps, n_assets) - historical price data
	    risk_aversion: float - risk aversion parameter (higher = more risk-averse, typically 1-10)

	Returns:
	    1D array of shape (n_particles,) - negative risk-adjusted utility for each particle
	    (negative because optimizers typically minimize, but we want to maximize utility)

	Example:
	    >>> import numpy as np
	    >>> prices = np.array([[100, 200], [101, 198], [102, 201]])  # 3 timesteps, 2 assets
	    >>> weights = np.array([[0.6, 0.4], [0.3, 0.7]])  # 2 particles
	    >>> objectives = price_based_objective_function(weights, prices, risk_aversion=2.0)
	    >>> best_particle = np.argmin(objectives)  # Lowest objective = highest utility
	"""
	# Ensure weights is 2D
	if weights.ndim == 1:
		weights = weights.reshape(1, -1)

	n_particles, n_assets = weights.shape

	# Calculate returns from prices
	returns = np.diff(prices, axis=0) / prices[:-1]  # Shape: (n_timesteps-1, n_assets)

	# Calculate expected returns (mean of historical returns)
	mu = np.mean(returns, axis=0)  # Shape: (n_assets,)

	# Calculate covariance matrix
	cov = np.cov(returns, rowvar=False)  # Shape: (n_assets, n_assets)

	# Portfolio expected returns for all particles using matrix multiplication
	# weights @ mu.T -> (n_particles, n_assets) @ (n_assets,) -> (n_particles,)
	portfolio_returns = weights @ mu

	# Portfolio variances for all particles using matrix multiplication
	# For each particle i: weights[i] @ cov @ weights[i].T
	# Vectorized: (weights @ cov) * weights -> element-wise product, then sum along assets
	portfolio_variances = np.sum((weights @ cov) * weights, axis=1)  # Shape: (n_particles,)

	# Risk-adjusted returns using mean-variance utility
	# Utility = E[R] - (risk_aversion / 2) * Var[R]
	risk_adjusted_returns = portfolio_returns - (risk_aversion / 2) * portfolio_variances

	# Return negative because optimizers typically minimize (we want to maximize utility)
	return -risk_adjusted_returns


def sortino_ratio_objective(weights: np.ndarray, prices: np.ndarray, target_return: float = 0.0) -> np.ndarray:
	"""
	Calculate Sortino ratio for multiple portfolios - focuses only on downside risk.

	The Sortino ratio is a distribution-free risk measure that only penalizes returns
	below a target threshold, making it more suitable for asymmetric return distributions.

	Sortino Ratio = (E[R] - target_return) / Downside_Deviation
	where Downside_Deviation = sqrt(E[min(R - target_return, 0)^2])

	Args:
	    weights: 2D array of shape (n_particles, n_assets) - portfolio weights
	    prices: 2D array of shape (n_timesteps, n_assets) - historical price data
	    target_return: float - minimum acceptable return (default 0.0 for zero threshold)

	Returns:
	    1D array of shape (n_particles,) - negative Sortino ratios
	"""
	if weights.ndim == 1:
		weights = weights.reshape(1, -1)

	# Calculate returns
	returns = np.diff(prices, axis=0) / prices[:-1]

	# Calculate portfolio returns for each particle
	portfolio_returns = returns @ weights.T  # Shape: (n_timesteps-1, n_particles)

	# Calculate expected returns
	expected_returns = np.mean(portfolio_returns, axis=0)  # Shape: (n_particles,)

	# Calculate downside deviations (only negative excess returns)
	excess_returns = portfolio_returns - target_return  # Shape: (n_timesteps-1, n_particles)
	downside_returns = np.minimum(excess_returns, 0)  # Only negative values
	downside_variance = np.mean(downside_returns**2, axis=0)  # Shape: (n_particles,)
	downside_deviation = np.sqrt(downside_variance)

	# Avoid division by zero
	downside_deviation = np.maximum(downside_deviation, 1e-10)

	# Calculate Sortino ratio
	sortino_ratios = (expected_returns - target_return) / downside_deviation

	# Return negative for minimization
	return -sortino_ratios


def conditional_value_at_risk_objective(weights: np.ndarray, prices: np.ndarray, alpha: float = 0.05) -> np.ndarray:
	"""
	Calculate Conditional Value at Risk (CVaR) based objective function.

	CVaR is a coherent risk measure that considers the expected loss beyond the VaR threshold.
	It's distribution-free and provides a more conservative risk assessment than VaR alone.

	The objective maximizes return while minimizing CVaR:
	Objective = E[R] - CVaR_alpha

	Args:
	    weights: 2D array of shape (n_particles, n_assets) - portfolio weights
	    prices: 2D array of shape (n_timesteps, n_assets) - historical price data
	    alpha: float - confidence level for CVaR (default 0.05 for 95% CVaR)

	Returns:
	    1D array of shape (n_particles,) - negative risk-adjusted returns using CVaR
	"""
	if weights.ndim == 1:
		weights = weights.reshape(1, -1)

	# Calculate returns
	returns = np.diff(prices, axis=0) / prices[:-1]

	# Calculate portfolio returns for each particle
	portfolio_returns = returns @ weights.T  # Shape: (n_timesteps-1, n_particles)

	# Calculate expected returns
	expected_returns = np.mean(portfolio_returns, axis=0)  # Shape: (n_particles,)

	# Calculate CVaR for each portfolio
	n_particles = weights.shape[0]
	cvar_values = np.zeros(n_particles)

	for i in range(n_particles):
		port_ret = portfolio_returns[:, i]
		# Sort returns in ascending order (worst first)
		sorted_returns = np.sort(port_ret)
		# Find VaR threshold
		var_index = int(np.floor(alpha * len(sorted_returns)))
		var_index = max(0, var_index - 1)  # Ensure valid index
		# CVaR is the mean of returns below VaR
		cvar_values[i] = np.mean(sorted_returns[: var_index + 1]) if var_index >= 0 else sorted_returns[0]

	# Risk-adjusted return: maximize return while minimizing CVaR (note: CVaR is negative for losses)
	risk_adjusted_returns = expected_returns - np.abs(cvar_values)

	# Return negative for minimization
	return -risk_adjusted_returns


def maximum_drawdown_objective(weights: np.ndarray, prices: np.ndarray, return_penalty: float = 1.0) -> np.ndarray:
	"""
	Calculate maximum drawdown based objective function.

	Maximum drawdown measures the largest peak-to-trough decline in portfolio value,
	providing a distribution-free measure of downside risk that captures the worst
	historical loss period.

	Objective = return_penalty * E[R] - Maximum_Drawdown

	Args:
	    weights: 2D array of shape (n_particles, n_assets) - portfolio weights
	    prices: 2D array of shape (n_timesteps, n_assets) - historical price data
	    return_penalty: float - weight for expected return vs drawdown (default 1.0)

	Returns:
	    1D array of shape (n_particles,) - negative risk-adjusted returns using max drawdown
	"""
	if weights.ndim == 1:
		weights = weights.reshape(1, -1)

	# Calculate returns
	returns = np.diff(prices, axis=0) / prices[:-1]

	# Calculate portfolio returns for each particle
	portfolio_returns = returns @ weights.T  # Shape: (n_timesteps-1, n_particles)

	# Calculate expected returns
	expected_returns = np.mean(portfolio_returns, axis=0)  # Shape: (n_particles,)

	# Calculate maximum drawdown for each portfolio
	n_particles = weights.shape[0]
	max_drawdowns = np.zeros(n_particles)

	for i in range(n_particles):
		port_ret = portfolio_returns[:, i]
		# Calculate cumulative returns (portfolio value over time)
		cumulative_returns = np.cumprod(1 + port_ret)
		# Calculate running maximum (peak values)
		running_max = np.maximum.accumulate(cumulative_returns)
		# Calculate drawdowns (current value / peak - 1)
		drawdowns = cumulative_returns / running_max - 1
		# Maximum drawdown is the worst (most negative) drawdown
		max_drawdowns[i] = np.min(drawdowns)

	# Risk-adjusted return: maximize return while minimizing drawdown
	# Note: max_drawdown is negative, so we subtract it (making it positive penalty)
	risk_adjusted_returns = return_penalty * expected_returns - np.abs(max_drawdowns)

	# Return negative for minimization
	return -risk_adjusted_returns


def robust_sharpe_objective(weights: np.ndarray, prices: np.ndarray, mad_multiplier: float = 1.4826) -> np.ndarray:
	"""
	Calculate robust Sharpe ratio using Median Absolute Deviation (MAD) instead of standard deviation.

	The robust Sharpe ratio replaces standard deviation with MAD, making it less sensitive
	to outliers and not assuming normal distribution. MAD is multiplied by 1.4826 to make
	it consistent with standard deviation under normality.

	Robust Sharpe = (Median[R] - Median[Risk_Free_Rate]) / (MAD[R] * mad_multiplier)

	Args:
	    weights: 2D array of shape (n_particles, n_assets) - portfolio weights
	    prices: 2D array of shape (n_timesteps, n_assets) - historical price data
	    mad_multiplier: float - scaling factor for MAD (1.4826 for normal equivalence)

	Returns:
	    1D array of shape (n_particles,) - negative robust Sharpe ratios
	"""
	if weights.ndim == 1:
		weights = weights.reshape(1, -1)

	# Calculate returns
	returns = np.diff(prices, axis=0) / prices[:-1]

	# Calculate portfolio returns for each particle
	portfolio_returns = returns @ weights.T  # Shape: (n_timesteps-1, n_particles)

	# Calculate median returns (more robust than mean)
	median_returns = np.median(portfolio_returns, axis=0)  # Shape: (n_particles,)

	# Calculate MAD for each portfolio
	n_particles = weights.shape[0]
	mad_values = np.zeros(n_particles)

	for i in range(n_particles):
		port_ret = portfolio_returns[:, i]
		# MAD = median(|returns - median(returns)|)
		mad_values[i] = np.median(np.abs(port_ret - np.median(port_ret)))

	# Avoid division by zero
	mad_values = np.maximum(mad_values, 1e-10)

	# Calculate robust Sharpe ratios
	# Assuming risk-free rate is 0 for simplicity
	robust_sharpe_ratios = median_returns / (mad_values * mad_multiplier)

	# Return negative for minimization
	return -robust_sharpe_ratios


def risk_adjusted_returns_objective(
	x: np.ndarray,
	enable_l_moments: bool,
	l_moments: LMoments,
	risk_aversion: float,
	mu: np.ndarray,
	cov: np.ndarray,
) -> np.ndarray:
	scale = x[:, 0:1]  # (n_particles, 1)
	raw_weights = x[:, 1:]  # (n_particles, n_assets)

	scale = _adjust_scaling(scale)

	# Normalize raw weights to sum to 1, then scale
	raw_weight_sums = np.sum(raw_weights, axis=1, keepdims=True)
	# Avoid division by zero
	raw_weight_sums = np.maximum(raw_weight_sums, 1e-10)
	normalized_weights = raw_weights / raw_weight_sums
	final_weights = scale * normalized_weights

	if enable_l_moments:
		cost = -1 * expected_return_moments(
			weights=final_weights,
			l_moments=l_moments,
			risk_aversion=risk_aversion,
			normalize_weights=False,
		)
	else:
		cost = -1 * expected_return_classical(
			weights=final_weights,
			mu=mu,
			cov=cov,
			risk_aversion=risk_aversion,
			normalize_weights=False,
		)

	return cost


def _adjust_scaling(scale: float) -> float:
	# make scaling more gradual - if almost fully invested, do fully invest, if almost zero, do zero

	scale = np.clip(scale, 0.0, 1.0)

	x_points = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
	y_points = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])

	scale = np.interp(scale, x_points, y_points)

	return scale


if __name__ == "__main__":
	# Test the distribution-free objective functions with synthetic data
	np.random.seed(42)

	# Create synthetic price data with different distribution characteristics
	n_timesteps = 252  # One year of daily data
	n_assets = 3

	# Generate non-normal returns with skewness and fat tails
	base_returns = np.random.normal(0, 0.02, (n_timesteps, n_assets))

	# Add skewness: occasional large positive returns for asset 0
	skew_events = np.random.random(n_timesteps) < SKEW_EVENT_PROBABILITY  # 5% chance
	base_returns[skew_events, 0] += np.random.exponential(SKEW_EVENT_MAGNITUDE, np.sum(skew_events))

	# Add fat tails: occasional large losses for asset 1
	tail_events = np.random.random(n_timesteps) < TAIL_EVENT_PROBABILITY  # 3% chance
	base_returns[tail_events, 1] -= np.random.exponential(0.08, np.sum(tail_events))

	# Convert returns to prices
	initial_prices = np.array([100.0, 100.0, 100.0])
	prices = np.zeros((n_timesteps + 1, n_assets))
	prices[0] = initial_prices

	for t in range(n_timesteps):
		prices[t + 1] = prices[t] * (1 + base_returns[t])

	# Test portfolios
	test_weights = np.array(
		[
			[0.33, 0.33, 0.34],  # Equal weight
			[0.50, 0.30, 0.20],  # Concentrated in asset 0 (positive skew)
			[0.20, 0.60, 0.20],  # Concentrated in asset 1 (fat tails)
			[0.10, 0.10, 0.80],  # Concentrated in asset 2 (normal)
		]
	)

	print("=== Comparing Distribution-Free Risk Metrics ===")
	print(f"Price data shape: {prices.shape}")
	print(f"Test portfolios: {test_weights.shape[0]}")
	print()

	# 1. Mean-Variance (assumes normality)
	mv_scores = price_based_objective_function(test_weights, prices, risk_aversion=2.0)
	print("1. Mean-Variance Utility (assumes normal distribution):")
	for i, score in enumerate(-mv_scores):  # Convert back to positive utility
		print(f"   Portfolio {i+1}: {score:.6f}")
	print()

	# 2. Sortino Ratio (downside risk only)
	sortino_scores = sortino_ratio_objective(test_weights, prices, target_return=0.0)
	print("2. Sortino Ratio (downside risk only, distribution-free):")
	for i, score in enumerate(-sortino_scores):
		print(f"   Portfolio {i+1}: {score:.6f}")
	print()

	# 3. CVaR Objective (tail risk focus)
	cvar_scores = conditional_value_at_risk_objective(test_weights, prices, alpha=0.05)
	print("3. CVaR Objective (tail risk, distribution-free):")
	for i, score in enumerate(-cvar_scores):
		print(f"   Portfolio {i+1}: {score:.6f}")
	print()

	# 4. Maximum Drawdown (worst loss period)
	mdd_scores = maximum_drawdown_objective(test_weights, prices, return_penalty=10.0)
	print("4. Maximum Drawdown Objective (worst loss period, distribution-free):")
	for i, score in enumerate(-mdd_scores):
		print(f"   Portfolio {i+1}: {score:.6f}")
	print()

	# 5. Robust Sharpe (MAD instead of std dev)
	robust_scores = robust_sharpe_objective(test_weights, prices)
	print("5. Robust Sharpe Ratio (MAD-based, distribution-free):")
	for i, score in enumerate(-robust_scores):
		print(f"   Portfolio {i+1}: {score:.6f}")
	print()

	# Show ranking differences
	print("=== Ranking Comparison ===")
	mv_ranks = np.argsort(-mv_scores)  # Best to worst
	sortino_ranks = np.argsort(-sortino_scores)
	cvar_ranks = np.argsort(-cvar_scores)
	mdd_ranks = np.argsort(-mdd_scores)
	robust_ranks = np.argsort(-robust_scores)

	print("Rankings (1=best, 4=worst):")
	print("Portfolio | Mean-Var | Sortino | CVaR | MaxDD | Robust")
	print("----------|----------|---------|------|-------|-------")
	for i in range(len(test_weights)):
		mv_rank = np.where(mv_ranks == i)[0][0] + 1
		sortino_rank = np.where(sortino_ranks == i)[0][0] + 1
		cvar_rank = np.where(cvar_ranks == i)[0][0] + 1
		mdd_rank = np.where(mdd_ranks == i)[0][0] + 1
		robust_rank = np.where(robust_ranks == i)[0][0] + 1

		print(
			f"    {i+1:1d}     |    {mv_rank:1d}     |    {sortino_rank:1d}    |  {cvar_rank:1d}   | "
			f"  {mdd_rank:1d}   |   {robust_rank:1d}"
		)

	print("\nDistribution-free metrics often give different rankings than mean-variance,")
	print("especially when returns are not normally distributed (have skewness or fat tails).")
