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


def risk_adjusted_returns_objective(  # noqa: PLR0913
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
