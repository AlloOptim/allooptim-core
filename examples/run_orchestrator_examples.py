"""
Example demonstrating AllocationOrchestrator usage with all three orchestration types.

This shows how to use the orchestrator in standalone scripts for:
1. Equal weighting (simple average)
2. Optimized allocation (MCOS + PSO)
3. Wikipedia pipeline (pre-selection + optimization)
"""

import time

import numpy as np
import pandas as pd

from allo_optim.allocation_to_allocators.allocation_orchestrator import (
    AllocationOrchestrator,
    AllocationOrchestratorConfig,
    OrchestrationType,
)
from allo_optim.config.stock_universe import list_of_dax_stocks


def main():
    """Run allocation using all three orchestration types."""

    # Create sample price data
    np.random.seed(42)
    all_stocks = list_of_dax_stocks()[:10]  # Use 10 stocks
    assets = [stock.symbol for stock in all_stocks]

    # Generate 6 months of daily price data
    dates = pd.date_range("2023-01-01", periods=126, freq="D")
    n_assets = len(assets)

    # Generate correlated price movements
    returns = np.random.multivariate_normal(
        mean=np.full(n_assets, 0.0005),  # 0.05% daily return
        cov=np.eye(n_assets) * 0.0004 + np.full((n_assets, n_assets), 0.0001),  # Some correlation
        size=len(dates),
    )

    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=assets)

    print("=" * 80)
    print("AllocationOrchestrator Examples")
    print("=" * 80)
    print(f"\nTest data: {len(assets)} assets, {len(dates)} days")
    print(f"Assets: {', '.join(assets[:5])}...")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}\n")

    # Configure orchestrator with 3 optimizers and 1 transformer
    optimizer_names = ["NaiveOptimizer", "MomentumOptimizer", "RiskParityOptimizer"]
    transformer_names = ["OracleCovarianceTransformer"]

    orchestration_types = [
        (OrchestrationType.EQUAL, "Equal Weighting"),
        (OrchestrationType.OPTIMIZED, "Optimized (MCOS + PSO)"),
        (OrchestrationType.WIKIPEDIA_PIPELINE, "Wikipedia Pipeline"),
    ]

    for orch_type, description in orchestration_types:
        print(f"\n{'=' * 80}")
        print(f"Testing: {description}")
        print(f"{'=' * 80}")

        # Create configuration
        config = AllocationOrchestratorConfig(
            orchestration_type=orch_type,
            n_data_observations=20 if orch_type == OrchestrationType.OPTIMIZED else 50,
            n_particle_swarm_iterations=100,
            n_particles=50,
            use_wiki_database=False,
            n_historical_days=30,
        )

        # Create orchestrator
        orchestrator = AllocationOrchestrator(
            optimizer_names=optimizer_names,
            transformer_names=transformer_names,
            config=config,
        )

        # Run allocation
        start_time = time.time()
        result = orchestrator.allocate(
            all_stocks=all_stocks,
            time_today=dates[-1],
            df_prices=prices,
        )
        elapsed = time.time() - start_time

        # Display results
        print(f"\nSuccess: {result.success}")
        print(f"Computation time: {elapsed:.2f}s")

        if result.success:
            print("\nAsset Weights (top 5):")
            sorted_weights = sorted(result.asset_weights.items(), key=lambda x: x[1], reverse=True)
            for asset, weight in sorted_weights[:5]:
                print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")

            total_weight = sum(result.asset_weights.values())
            print(f"\nTotal weight: {total_weight:.6f}")

            # Show df_allocation if available
            if result.df_allocation is not None:
                print("\nOptimizer Allocations (df_allocation):")
                print(f"  Shape: {result.df_allocation.shape} (optimizers x assets)")
                print(f"  Optimizers: {list(result.df_allocation.index)}")
                print("\n  Sample allocation (first 3 assets):")
                print(result.df_allocation.iloc[:, :3].to_string())

            # Show statistics if available
            if hasattr(result.statistics, "asset_returns"):
                print("\nPerformance Statistics:")
                print(f"  Returns: {result.statistics.asset_returns}")
                print(f"  Runtime: {result.statistics.algo_runtime}")
        else:
            print(f"Error: {result.error_message}")

    print(f"\n{'=' * 80}")
    print("All orchestration types demonstrated successfully!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
