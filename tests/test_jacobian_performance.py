"""Performance benchmarks for Jacobian/Hessian implementations."""

import time

import numpy as np
import pandas as pd
import pytest

from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import (
    RobustMeanVarianceOptimizer,
    RobustMeanVarianceOptimizerConfig,
)


class TestPerformanceBenchmarks:
    """Benchmark optimization speed with/without analytical derivatives."""

    @pytest.mark.parametrize("n_assets", [10, 30, 50])
    def test_risk_parity_speedup(self, n_assets):
        """Measure speedup from analytical Jacobian."""
        # Generate test data
        np.random.seed(42)
        assets = [f"Asset_{i}" for i in range(n_assets)]
        mu = pd.Series(np.random.randn(n_assets) * 0.1 + 0.08, index=assets)

        # Random positive semi-definite covariance
        A = np.random.randn(n_assets, n_assets)
        cov_array = A @ A.T / n_assets + np.eye(n_assets) * 0.01
        cov = pd.DataFrame(cov_array, index=assets, columns=assets)

        optimizer = RiskParityOptimizer()

        # Warm-up
        _ = optimizer.allocate(mu, cov)

        # Benchmark with analytical Jacobian (current implementation)
        start = time.time()
        weights_with_jac = optimizer.allocate(mu, cov)
        time_with_jac = time.time() - start

        # TODO: Benchmark without Jacobian (would need to disable it)
        # For now, just measure and report
        print(f"\n{n_assets} assets:")
        print(f"  Time with Jacobian: {time_with_jac:.4f}s")
        print(f"  Weights sum: {weights_with_jac.sum():.6f}")

        assert time_with_jac < 5.0, f"Optimization too slow: {time_with_jac}s"

    def test_robust_mv_convergence(self):
        """Test convergence properties with analytical derivatives."""
        config = RobustMeanVarianceOptimizerConfig(
            risk_aversion=1.0,
            mu_uncertainty_level=0.1,
            cov_uncertainty_level=0.05,
        )

        n_assets = 20
        assets = [f"Asset_{i}" for i in range(n_assets)]

        np.random.seed(42)
        mu = pd.Series(np.random.randn(n_assets) * 0.1 + 0.08, index=assets)
        A = np.random.randn(n_assets, n_assets)
        cov_array = A @ A.T / n_assets + np.eye(n_assets) * 0.01
        cov = pd.DataFrame(cov_array, index=assets, columns=assets)

        optimizer = RobustMeanVarianceOptimizer(config)

        start = time.time()
        weights = optimizer.allocate(mu, cov)
        elapsed = time.time() - start

        print(f"\nRobust MV with {n_assets} assets:")
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Weights sum: {weights.sum():.6f}")
        print(f"  Min weight: {weights.min():.6f}")
        print(f"  Max weight: {weights.max():.6f}")

        assert elapsed < 2.0, "Optimization too slow"
        assert 0.0 <= weights.sum() <= 1.01, "Invalid weight sum"


def benchmark_all_optimizers():
    """Comprehensive benchmark comparing all implementations."""
    print("\n" + "=" * 60)
    print("JACOBIAN/HESSIAN PERFORMANCE BENCHMARK")
    print("=" * 60)

    results = []

    # Test configuration
    n_assets = 30
    n_trials = 5

    # Generate test data
    np.random.seed(42)
    assets = [f"Asset_{i}" for i in range(n_assets)]
    mu = pd.Series(np.random.randn(n_assets) * 0.1 + 0.08, index=assets)
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T / n_assets + np.eye(n_assets) * 0.01
    cov = pd.DataFrame(cov_array, index=assets, columns=assets)

    # Test Risk Parity
    optimizer = RiskParityOptimizer()
    times = []
    for _ in range(n_trials):
        start = time.time()
        _ = optimizer.allocate(mu, cov)
        times.append(time.time() - start)

    results.append(
        {
            "Optimizer": "RiskParity",
            "Has Jacobian": "Yes",
            "Has Hessian": "No",
            "Mean Time": f"{np.mean(times):.4f}s",
            "Std Time": f"{np.std(times):.4f}s",
        }
    )

    # Test Robust MV
    optimizer = RobustMeanVarianceOptimizer()
    times = []
    for _ in range(n_trials):
        start = time.time()
        _ = optimizer.allocate(mu, cov)
        times.append(time.time() - start)

    results.append(
        {
            "Optimizer": "RobustMeanVariance",
            "Has Jacobian": "Yes",
            "Has Hessian": "Yes",
            "Mean Time": f"{np.mean(times):.4f}s",
            "Std Time": f"{np.std(times):.4f}s",
        }
    )

    # Print results
    print(f"\nBenchmark Results ({n_assets} assets, {n_trials} trials):")
    print("-" * 80)
    for r in results:
        print(
            f"{r['Optimizer']:25s} | Jac: {r['Has Jacobian']:3s} | Hess: {r['Has Hessian']:3s} | "
            f"Time: {r['Mean Time']} ± {r['Std Time']}"
        )
    print("-" * 80)

    return results


if __name__ == "__main__":
    benchmark_all_optimizers()
