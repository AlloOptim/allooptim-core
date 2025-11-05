#!/usr/bin/env python3
"""
Comprehensive Allocation Algorithm Backtest

This script performs a comprehensive backtest of all allocation algorithms:
- 13 individual optimizers from OPTIMIZER_LIST
- A2A ensemble (average of all optimizer weights)
- S&P 500 benchmark

Backtest Configuration:
- Period: 2014-12-31 to 2024-12-31 (10 years)
- Rebalancing: Every 5 trading days
- Lookback: 90 days for optimizer estimation
- Universe: ~400 assets from Alpaca universe
- Execution: Perfect execution (target = actual)

Performance Metrics:
- Sharpe ratio, max drawdown, time underwater, CAGR
- Risk-adjusted return, portfolio turnover, daily returns statistics
- Computation time and memory usage
- Optimizer clustering analysis
"""

import logging
import traceback
import warnings
from datetime import datetime

import pandas as pd

from allo_optim.allocation_to_allocators.a2a_config import A2AConfig
from allo_optim.backtest.backtest_config import BacktestConfig
from allo_optim.backtest.backtest_engine import BacktestEngine
from allo_optim.backtest.backtest_report import generate_report
from allo_optim.backtest.backtest_visualizer import create_visualizations
from allo_optim.backtest.cluster_analyzer import ClusterAnalyzer
from allo_optim.config.stock_universe import everything_in_alpaca, extract_symbols_from_list

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""

    logger.info("Starting comprehensive allocation algorithm backtest")

    try:
        symbols = extract_symbols_from_list(everything_in_alpaca())

        config_backtest = BacktestConfig(
            start_date=datetime(2014, 12, 31),
            end_date=datetime(2024, 12, 31),
            rebalance_frequency=10,
            lookback_days=90,
            quick_test=False,
            log_returns=True,
            benchmark="SPY",
            symbols=symbols,
            optimizer_names=[
                "MeanVarianceCMAOptimizer",
                "LMomentsCMAOptimizer",
                "SortinoCMAOptimizer",
                "MaxDrawdownCMAOptimizer",
                "RobustSharpeCMAOptimizer",
                "CVARCMAOptimizer",
                "MeanVarianceParticleSwarmOptimizer",
                "LMomentsParticleSwarmOptimizer",
                "HRPOptimizer",
                "NCOSharpeOptimizer",
                "NaiveOptimizer",
                "MomentumOptimizer",
                "RiskParityOptimizer",
                "MeanVarianceAdjustedReturnsOptimizer",
                "EMAAdjustedReturnsOptimizer",
                "LMomentsAdjustedReturnsOptimizer",
                "SemiVarianceAdjustedReturnsOptimizer",
                "HigherMomentOptimizer",
                "MaxSharpeOptimizer",
                "EfficientReturnOptimizer",
                "EfficientRiskOptimizer",
                "AugmentedLightGBMOptimizer",
                "RobustMeanVarianceOptimizer",
            ],
            transformer_names=["OracleCovarianceTransformer"],
            orchestration_type="equal",
        )

        config_a2a = A2AConfig()

        # Initialize backtest engine with config
        backtest_engine = BacktestEngine(config_backtest, config_a2a)

        # Run backtest
        results = backtest_engine.run_backtest()

        if not results:
            logger.error("No results generated from backtest")
            return

        # Perform clustering analysis
        cluster_analyzer = ClusterAnalyzer(results)
        clustering_results = cluster_analyzer.analyze_clusters()

        # Create visualizations
        create_visualizations(results, clustering_results, config_backtest.results_dir)

        # Generate report
        report = generate_report(results, clustering_results, config_backtest)

        # Save report
        report_path = config_backtest.results_dir / "comprehensive_backtest_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        # Save detailed results to CSV
        csv_data = []
        for name, data in results.items():
            row = {"optimizer": name}
            row.update(data["metrics"])
            csv_data.append(row)

        results_df = pd.DataFrame(csv_data)
        results_df.to_csv(config_backtest.results_dir / "backtest_results.csv", index=False)

        # Save Euclidean distance analysis to CSV
        if "euclidean_distance" in clustering_results and "closest_pairs" in clustering_results["euclidean_distance"]:
            distance_data = clustering_results["euclidean_distance"]["closest_pairs"]
            if distance_data:
                distance_df = pd.DataFrame(distance_data)
                distance_df.to_csv(config_backtest.results_dir / "optimizer_distances.csv", index=False)
                logger.info("Optimizer distance analysis saved to optimizer_distances.csv")

        logger.info(f"Backtest completed successfully. Results saved to {config_backtest.results_dir}")
        logger.info(f"Report available at: {report_path}")

        # Print summary
        print(f"\n{'='*80}")
        print("BACKTEST COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Period: {config_backtest.get_report_date_range()[0]} to {config_backtest.get_report_date_range()[1]}")
        print(f"Optimizers tested: {len(results)}")
        print(f"Results directory: {config_backtest.results_dir}")
        print(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
