#!/usr/bin/env python3
"""
Test script for the new CMA-ES optimizer integration
"""

import numpy as np
import pandas as pd

from allo_optim.optimizer.covariance_matrix_adaption.cma_optimizer import MeanVarianceCMAOptimizer

# Constants for test tolerances
WEIGHT_SUM_TEST_TOLERANCE = 0.01


def create_test_data():
	"""Create sample test data for optimization"""
	# Create sample expected returns
	assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
	mu = pd.Series([0.12, 0.15, 0.10, 0.14, 0.18], index=assets)

	# Create sample covariance matrix
	np.random.seed(42)  # For reproducible results
	cov_data = np.random.randn(5, 5) * 0.1
	cov_data = cov_data @ cov_data.T  # Make positive semi-definite
	cov = pd.DataFrame(cov_data, index=assets, columns=assets)

	return mu, cov
