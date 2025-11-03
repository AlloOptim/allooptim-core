"""
Allocation Metrics Module
"""

import logging
from dataclasses import dataclass
from typing import Optional

import lmo
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Constants for numerical stability and defaults
EPSILON_REGULARIZATION = 1e-6
EPSILON_DIVISION = 1e-10
EPSILON_MIN_STD = 1e-8
DEFAULT_RISK_AVERSION = 4.0
DEFAULT_TRIM = 0.02
MIN_OBSERVATIONS = 10

# Constants for tensor dimensions in higher moment calculations
SKEWNESS_TENSOR_DIMENSION = 3
KURTOSIS_TENSOR_DIMENSION = 4


def make_positive_definite(
	cov: np.ndarray,
	epsilon: float = EPSILON_REGULARIZATION,
) -> np.ndarray:
	"""Make a covariance matrix positive definite by adding regularization if needed."""
	try:
		# Try Cholesky decomposition to check if matrix is positive definite
		np.linalg.cholesky(cov)
		return cov
	except np.linalg.LinAlgError:
		# Matrix is not positive definite, add diagonal regularization
		# This is a simpler and more robust approach
		n = cov.shape[0]
		regularization = epsilon * np.eye(n)
		cov_regularized = cov + regularization

		# Verify the regularization worked
		try:
			np.linalg.cholesky(cov_regularized)
			return cov_regularized
		except np.linalg.LinAlgError:
			# If still not positive definite, use a larger regularization
			return cov + (epsilon * 10) * np.eye(n)


@dataclass
class LMoments:
	lt_comoment_1: np.ndarray
	lt_comoment_2: np.ndarray
	lt_comoment_3: np.ndarray
	lt_comoment_4: np.ndarray


def validate_no_nan(data: np.ndarray, name: str) -> None:
	"""Validate array contains no NaN values"""
	if np.any(np.isnan(data)):
		raise ValueError(f"{name} contains NaN values: {data}")


def expected_return_moments(
	weights: np.ndarray,
	l_moments: LMoments,
	risk_aversion: float = DEFAULT_RISK_AVERSION,
	normalize_weights: bool = False,
) -> np.ndarray:
	if normalize_weights:
		weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

	# Handle both single weight vector and batch of weight vectors
	if weights.ndim == 1:
		# Single weight vector case: (n_assets,)
		weights = weights.reshape(1, -1)

	# Batch matrix multiplication for portfolio L-comoments
	# weights shape: (n_particles, n_assets)
	# l_moments matrices shape: (n_assets, n_assets)
	n_particles = weights.shape[0]
	port_lt_comom_1 = np.zeros(n_particles)
	port_lt_comom_2 = np.zeros(n_particles)
	port_lt_comom_3 = np.zeros(n_particles)
	port_lt_comom_4 = np.zeros(n_particles)

	for k in range(n_particles):
		w = weights[k, :]
		port_lt_comom_1[k] = w @ l_moments.lt_comoment_1 @ w
		port_lt_comom_2[k] = w @ l_moments.lt_comoment_2 @ w
		port_lt_comom_3[k] = w @ l_moments.lt_comoment_3 @ w
		port_lt_comom_4[k] = w @ l_moments.lt_comoment_4 @ w

	# Compute ratios for portfolio using the appropriate L-comoment scale
	# Handle division by zero element-wise
	port_lt_coskew = np.where(np.abs(port_lt_comom_2) > EPSILON_DIVISION, port_lt_comom_3 / port_lt_comom_2, 0.0)
	port_lt_cokurt = np.where(np.abs(port_lt_comom_2) > EPSILON_DIVISION, port_lt_comom_4 / port_lt_comom_2, 0.0)

	# Expected returns using L-comoments
	expected_return_lt_co_mv = port_lt_comom_1 - 0.5 * risk_aversion * port_lt_comom_2**2
	expected_return_lt_co_mvs = (
		expected_return_lt_co_mv + (1 / 6) * risk_aversion * (risk_aversion + 1) * port_lt_coskew * port_lt_comom_2**3
	)
	expected_return_lt_co_mvsk = (
		expected_return_lt_co_mvs
		+ (1 / 24) * risk_aversion * (risk_aversion + 1) * (risk_aversion + 2) * port_lt_cokurt * port_lt_comom_2**4
	)

	return expected_return_lt_co_mvsk


def expected_return_classical(
	weights: np.ndarray,
	mu: np.ndarray,
	cov: np.ndarray,
	skew: Optional[np.ndarray] = None,
	kurt: Optional[np.ndarray] = None,
	risk_aversion: float = DEFAULT_RISK_AVERSION,
	normalize_weights: bool = False,
) -> np.ndarray:
	"""
	Computes the portfolio geometric mean return analytically assuming normal returns.

	Args:
	    weights: 2D array of weight candidates (n_candidates, n_allocators)
	    mu: Expected returns for each allocator (n_allocators,)
	    cov: Covariance matrix (n_allocators, n_allocators)

	Returns:
	    Array of geometric mean returns (n_candidates,)
	"""
	if normalize_weights:
		weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

	portfolio_mean = np.dot(weights, mu)

	portfolio_variance = 0.5 * np.sum((weights @ cov) * weights, axis=1)

	if skew is not None and skew.ndim == SKEWNESS_TENSOR_DIMENSION:
		skew_component = (1 / 6) * np.einsum("pi,pj,pk,ijk", weights, weights, weights, skew)

	elif skew is not None and skew.ndim == 1:
		skew_component = (1 / 6) * np.sum(weights**3 * skew, axis=1)

	else:
		skew_component = 0.0

	if kurt is not None and kurt.ndim == KURTOSIS_TENSOR_DIMENSION:
		kurt_component = (1 / 24) * np.einsum("pi,pj,pk,pl,ijkl", weights, weights, weights, weights, kurt)
	elif kurt is not None and kurt.ndim == 1:
		kurt_component = (1 / 24) * np.sum(weights**4 * kurt, axis=1)

	else:
		kurt_component = 0.0

	geometric_mean = (
		portfolio_mean
		- risk_aversion * portfolio_variance
		+ risk_aversion**1.5 * skew_component
		- risk_aversion**2 * kurt_component
	)

	return geometric_mean


def estimate_linear_moments(
	returns: np.ndarray,
	trim: Optional[float] = DEFAULT_TRIM,
) -> LMoments:
	"""
	Estimate L-comoments from return samples.
	"""

	# Compute L-comoments
	l_comom_1 = lmo.l_comoment(returns, 1, rowvar=False, trim=trim)
	l_comom_2 = lmo.l_comoment(returns, 2, rowvar=False, trim=trim)
	l_comom_3 = lmo.l_comoment(returns, 3, rowvar=False, trim=trim)
	l_comom_4 = lmo.l_comoment(returns, 4, rowvar=False, trim=trim)

	l_moments = LMoments(
		lt_comoment_1=l_comom_1,
		lt_comoment_2=l_comom_2,
		lt_comoment_3=l_comom_3,
		lt_comoment_4=l_comom_4,
	)

	return l_moments


def estimate_classical_moments(
	returns: np.ndarray,
	trim: Optional[int] = None,
	estimate_vectors: bool = True,
	min_points: int = MIN_OBSERVATIONS,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Estimate co-skewness and co-kurtosis tensors or vectors from return samples using classical moments.

	Args:
	    returns: array of shape (n_observations, n_assets)
	    trim: Optional trimming parameter (not used in classical approach)
	    estimate_vectors: bool, if True, return only diagonal vectors (skew_iii, kurt_iiii)
	                      if False, return full tensors
	    min_points: minimum observations required

	Returns:
	    skew: np.ndarray (n_assets, n_assets, n_assets) or (n_assets,) if estimate_vectors
	    kurt: np.ndarray (n_assets, n_assets, n_assets, n_assets) or (n_assets,) if estimate_vectors
	"""
	n_obs, n_assets = returns.shape

	# Need at least min_points observations for moment estimation
	if n_obs < min_points:
		logger.warning(f"Not enough observations ({n_obs}) to estimate moments, returning zeros")
		if estimate_vectors:
			return np.zeros(n_assets), np.zeros(n_assets)
		else:
			return np.zeros((n_assets, n_assets, n_assets)), np.zeros((n_assets, n_assets, n_assets, n_assets))

	try:
		# Standardize returns (zero mean, unit variance)
		mean_returns = np.mean(returns, axis=0)
		std_returns = np.std(returns, axis=0, ddof=1)

		# Avoid division by zero
		std_returns = np.maximum(std_returns, EPSILON_MIN_STD)
		standardized = (returns - mean_returns) / std_returns

		if estimate_vectors:
			# Only compute diagonal elements (univariate skew and kurtosis for each asset)

			skew = np.array([stats.skew(returns[:, i]) for i in range(n_assets)])
			kurt = np.array([stats.kurtosis(returns[:, i]) for i in range(n_assets)])
			return skew, kurt
		else:
			# Compute full co-moment tensors
			skew_tensor = np.zeros((n_assets, n_assets, n_assets))
			kurt_tensor = np.zeros((n_assets, n_assets, n_assets, n_assets))

			# Co-skewness: E[(R_i - μ_i)(R_j - μ_j)(R_k - μ_k)] / (σ_i * σ_j * σ_k)
			for i in range(n_assets):
				for j in range(n_assets):
					for k in range(n_assets):
						coskew = np.mean(standardized[:, i] * standardized[:, j] * standardized[:, k])
						skew_tensor[i, j, k] = coskew

			# Co-kurtosis: E[(R_i - μ_i)(R_j - μ_j)(R_k - μ_k)(R_l - μ_l)] / (σ_i * σ_j * σ_k * σ_l)
			for i in range(n_assets):
				for j in range(n_assets):
					for k in range(n_assets):
						for l in range(n_assets):
							cokurt = np.mean(
								standardized[:, i] * standardized[:, j] * standardized[:, k] * standardized[:, l]
							)
							kurt_tensor[i, j, k, l] = cokurt

			return skew_tensor, kurt_tensor

	except Exception as e:
		logger.error(f"Failed to estimate classical moments: {e}")
		if estimate_vectors:
			return np.zeros(n_assets), np.zeros(n_assets)
		else:
			return np.zeros((n_assets, n_assets, n_assets)), np.zeros((n_assets, n_assets, n_assets, n_assets))
