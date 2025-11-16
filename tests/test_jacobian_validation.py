"""Unit tests for analytical Jacobian/Hessian implementations."""

import numpy as np
from scipy.optimize import check_grad, approx_fprime

from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import RobustMeanVarianceOptimizer
from allooptim.optimizer.sequential_quadratic_programming.higher_moments_optimizer import HigherMomentOptimizer
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import (
            RobustMeanVarianceOptimizerConfig,
        )
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import (
            RobustMeanVarianceOptimizerConfig,
        )
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import (
            RobustMeanVarianceOptimizerConfig,
        )       
from allooptim.optimizer.sequential_quadratic_programming.adjusted_return_optimizer import (
            LMomentsAdjustedReturnsOptimizer,
            MeanVarianceAdjustedReturnsOptimizerConfig,
        )        
from allooptim.optimizer.sequential_quadratic_programming.monte_carlo_robust_optimizer import (
            MonteCarloMaxDiversificationOptimizer,
        )
class TestRiskParityJacobian:
    """Test Risk Parity optimizer Jacobian."""

    def test_gradient_accuracy(self):
        """Verify analytical gradient matches numerical gradient."""
        optimizer = RiskParityOptimizer()

        # Setup test data
        n = 4
        x0 = np.array([0.25, 0.25, 0.25, 0.25])
        cov = np.array([
            [0.04, 0.01, 0.00, 0.00],
            [0.01, 0.09, 0.00, 0.00],
            [0.00, 0.00, 0.16, 0.02],
            [0.00, 0.00, 0.02, 0.25]
        ])

        optimizer._cov = cov
        optimizer._target_risk = np.ones(n) / n

        # Test using scipy's check_grad (more robust than approx_fprime)
        error = check_grad(
            optimizer._risk_budget_objective,
            optimizer._risk_budget_jacobian,
            x0
        )

        assert error < 1e-3, f"Gradient check failed with error: {error}"

    def test_gradient_at_multiple_points(self):
        """Test gradient accuracy at various weight configurations."""
        optimizer = RiskParityOptimizer()

        cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])

        optimizer._cov = cov
        optimizer._target_risk = np.ones(3) / 3

        # Test at multiple points
        test_points = [
            np.array([0.33, 0.33, 0.34]),  # Equal weights
            np.array([0.5, 0.3, 0.2]),      # Skewed
            np.array([0.6, 0.2, 0.2]),      # Heavy on one asset
        ]

        for x0 in test_points:
            error = check_grad(
                optimizer._risk_budget_objective,
                optimizer._risk_budget_jacobian,
                x0
            )
            assert error < 1e-3, f"Gradient check failed at {x0} with error: {error}"


class TestRobustMeanVarianceDerivatives:
    """Test Robust MV optimizer Jacobian and Hessian."""

    def test_jacobian_accuracy(self):
        """Verify analytical Jacobian matches numerical."""

        config = RobustMeanVarianceOptimizerConfig(
            risk_aversion=1.0,
            mu_uncertainty_level=0.1,
            cov_uncertainty_level=0.05,
        )
        optimizer = RobustMeanVarianceOptimizer(config)

        # Setup
        optimizer._mu_array = np.array([0.10, 0.12, 0.08])
        optimizer._cov_robust = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])
        optimizer._eps_mu = 0.1

        x0 = np.array([0.33, 0.33, 0.34])

        # Test using scipy's check_grad
        error = check_grad(
            optimizer._objective_function,
            optimizer._objective_jacobian,
            x0
        )

        assert error < 1e-5, f"Gradient check failed with error: {error}"

    def test_hessian_accuracy(self):
        """Verify analytical Hessian matches numerical."""

        config = RobustMeanVarianceOptimizerConfig(risk_aversion=1.0)
        optimizer = RobustMeanVarianceOptimizer(config)

        optimizer._mu_array = np.array([0.10, 0.12, 0.08])
        optimizer._cov_robust = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])
        optimizer._eps_mu = 0.1

        x0 = np.array([0.33, 0.33, 0.34])

        # Numerical Hessian (approximate via finite differences of Jacobian)
        num_hess = approx_fprime(x0, optimizer._objective_jacobian, epsilon=1e-8)

        # Analytical Hessian
        ana_hess = optimizer._objective_hessian(x0)

        assert num_hess.shape == ana_hess.shape, "Hessian shape mismatch"
        assert np.allclose(num_hess, ana_hess, rtol=1e-3, atol=1e-5), \
            f"Hessian mismatch: max diff = {np.max(np.abs(num_hess - ana_hess))}"

    def test_jacobian_near_zero(self):
        """Test gradient behavior near zero weights (edge case)."""

        config = RobustMeanVarianceOptimizerConfig()
        optimizer = RobustMeanVarianceOptimizer(config)

        optimizer._mu_array = np.array([0.10, 0.12, 0.08])
        optimizer._cov_robust = np.eye(3) * 0.04
        optimizer._eps_mu = 0.1

        # Test near origin (where L2 norm gradient is undefined)
        x_near_zero = np.array([1e-10, 1e-10, 1e-10])

        # Should not raise error
        grad = optimizer._objective_jacobian(x_near_zero)
        assert grad.shape == (3,), "Gradient shape incorrect"
        assert np.all(np.isfinite(grad)), "Gradient contains non-finite values"


class TestAdjustedReturnsDerivatives:
    """Test Adjusted Returns optimizer derivatives."""

    def test_mean_variance_jacobian(self):
        """Test MV jacobian for non-L-moments case."""


        config = MeanVarianceAdjustedReturnsOptimizerConfig(risk_aversion=2.0)
        optimizer = LMomentsAdjustedReturnsOptimizer(config)

        # Disable L-moments to test analytical derivatives
        optimizer.enable_l_moments = False

        optimizer._mu = np.array([0.08, 0.10, 0.12])
        optimizer._cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])

        x0 = np.array([0.33, 0.33, 0.34])

        error = check_grad(
            optimizer._objective_function,
            optimizer._objective_jacobian,
            x0
        )

        assert error < 1e-6, f"MV Gradient check failed: {error}"

    def test_hessian_is_constant(self):
        """Verify Hessian is constant (doesn't depend on weights)."""
        from allooptim.optimizer.sequential_quadratic_programming.adjusted_return_optimizer import (
            LMomentsAdjustedReturnsOptimizer,
            MeanVarianceAdjustedReturnsOptimizerConfig,
        )

        config = MeanVarianceAdjustedReturnsOptimizerConfig(risk_aversion=2.0)
        optimizer = LMomentsAdjustedReturnsOptimizer(config)
        optimizer.enable_l_moments = False

        optimizer._cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])

        # Hessian should be same at any point
        x1 = np.array([0.33, 0.33, 0.34])
        x2 = np.array([0.5, 0.3, 0.2])

        H1 = optimizer._objective_hessian(x1)
        H2 = optimizer._objective_hessian(x2)

        assert np.allclose(H1, H2), "Hessian should be constant"
        assert np.allclose(H1, H1.T), "Hessian should be symmetric"


class TestMonteCarloDerivatives:
    """Test Monte Carlo optimizer derivatives."""

    def test_min_variance_jacobian(self):
        """Test MIN_VARIANCE jacobian."""
        # Create simple test case
        cov_matrix = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])

        w0 = np.array([0.33, 0.33, 0.34])

        # Objective
        def obj(w):
            return w.T @ cov_matrix @ w

        # Analytical jacobian
        def jac(w):
            return 2 * cov_matrix @ w

        # Numerical
        num_jac = approx_fprime(w0, obj, epsilon=1e-8)
        ana_jac = jac(w0)

        assert np.allclose(num_jac, ana_jac, rtol=1e-6), \
            f"Variance Jacobian mismatch: num={num_jac}, ana={ana_jac}"

    def test_diversification_jacobian(self):
        """Test MAX_DIVERSIFICATION jacobian."""


        optimizer = MonteCarloMaxDiversificationOptimizer()

        cov_matrix = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])

        w0 = np.array([0.33, 0.33, 0.34])

        # Wrap objective and jacobian
        def obj(w):
            return optimizer._diversification_objective(w, cov_matrix)

        def jac(w):
            return optimizer._diversification_jacobian(w, cov_matrix)

        # Numerical
        num_jac = approx_fprime(w0, obj, epsilon=1e-8)
        ana_jac = jac(w0)

        assert np.allclose(num_jac, ana_jac, rtol=1e-4, atol=1e-6), \
            f"Diversification Jacobian mismatch: num={num_jac}, ana={ana_jac}"


class TestHigherMomentsDerivatives:
    """Test Higher Moments optimizer partial Jacobian."""

    def test_mean_variance_jacobian(self):
        """Test partial MV jacobian (L-moments gradient not implemented)."""
        from allooptim.optimizer.sequential_quadratic_programming.higher_moments_optimizer import (
            HigherMomentsOptimizerConfig,
        )

        config = HigherMomentsOptimizerConfig(risk_aversion=2.0, use_l_moments=False)
        optimizer = HigherMomentOptimizer(config)

        optimizer._mu = np.array([0.08, 0.10, 0.12])
        optimizer._cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])
        optimizer._l_moments = None  # Disable L-moments
        optimizer._skew_vec = np.array([0.1, 0.2, 0.3])  # Set dummy skew values
        optimizer._kurt_vec = np.array([3.1, 3.2, 3.3])  # Set dummy kurtosis values

        x0 = np.array([0.33, 0.33, 0.34])

        # Numerical gradient of objective
        num_grad = approx_fprime(x0, optimizer._objective, epsilon=1e-8)
        ana_grad = optimizer._objective_jacobian(x0)

        assert np.allclose(num_grad, ana_grad, rtol=1e-4, atol=1e-6), \
            f"Higher Moments Jacobian mismatch: num={num_grad}, ana={ana_grad}"

