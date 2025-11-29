"""Comprehensive Allocation Algorithm Backtest.

Runs a 14 year backtest (2014-2024) across multiple portfolio allocation algorithms,
generates performance reports, visualizations, and clustering analysis of optimizer behaviors.

THE RUNTIME OF THIS SCRIPT MAY BE LONG (SEVERAL HOURS) DUE TO THE EXTENSIVE BACKTEST PERIOD AND NUMBER OF OPTIMIZERS TESTED.

Enable or disable 'quick_test' mode in BacktestConfig to shorten runtime for testing purposes.
"""

import logging
import traceback
import warnings
from datetime import datetime
from typing import Optional
import pandas as pd

from allooptim.config.a2a_manager_config import A2AManagerConfig
from allooptim.config.backtest_config import BacktestConfig
from allooptim.optimizer.optimizer_config import OptimizerConfig
from allooptim.backtest.backtest_engine import BacktestEngine
from allooptim.backtest.backtest_quantstats import create_quantstats_reports
from allooptim.backtest.backtest_report import generate_report
from allooptim.backtest.backtest_visualizer import create_visualizations
from allooptim.backtest.cluster_analyzer import ClusterAnalyzer
from allooptim.allocation_to_allocators.orchestrator_factory import OrchestratorType
from allooptim.config.stock_universe import extract_symbols_from_list, large_stock_universe

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OPTIMIZER_CONFIG =[
                # # Example with default config
                # "CMAMeanVariance",
                # # Example with custom config
                # OptimizerConfig(name="CMALMoments", config={"budget": 2000}),
                # # More optimizers with defaults
                # OptimizerConfig(name="MeanVarianceParticleSwarmOptimizer"),
                # OptimizerConfig(name="LMomentsParticleSwarmOptimizer"),
                # OptimizerConfig(name="NCOSharpeOptimizer"),
                OptimizerConfig(name="NaiveOptimizer"),
                OptimizerConfig(name="MomentumOptimizer"),
                # OptimizerConfig(name="MeanVarianceAdjustedReturnsOptimizer"),
                # OptimizerConfig(name="SemiVarianceAdjustedReturnsOptimizer"),
                # OptimizerConfig(name="HigherMomentOptimizer"),
                # OptimizerConfig(name="EfficientReturnOptimizer"),
                # OptimizerConfig(name="EfficientRiskOptimizer"),
                # OptimizerConfig(name="MaxSharpeOptimizer"),
            ]


def main_backtest(
    config_backtest: Optional[BacktestConfig] = None,
    a2a_manager_config: Optional[A2AManagerConfig] = None,
) -> None:
    """Main execution function."""
    logger.info("Starting comprehensive allocation algorithm backtest")

    # Use default configs if none provided
    if config_backtest is None:
        config_backtest = BacktestConfig(
            start_date=datetime(2015, 1, 1),
            end_date=datetime(2025, 1, 1),
            rebalance_frequency=10,
            quick_test=True,
            quick_start_date=datetime(2022, 1, 1),  # Extended period for A2A to work
            quick_end_date=datetime(2023, 12, 31),   # Extended period for A2A to work
        )

    if a2a_manager_config is None:
        symbols = extract_symbols_from_list(large_stock_universe())

        a2a_manager_config = A2AManagerConfig(
            symbols=symbols,
            optimizer_configs=DEFAULT_OPTIMIZER_CONFIG,
            transformer_names=["OracleCovarianceTransformer"],
            orchestration_type=OrchestratorType.VOLATILITY_ADJUSTED,
            lookback_days=90,
        )

    try:

        # Initialize backtest engine with config
        backtest_engine = BacktestEngine(config_backtest, a2a_manager_config)

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

        # Generate QuantStats reports
        create_quantstats_reports(
            results=results,
            output_dir=config_backtest.results_dir,
            generate_individual=config_backtest.quantstats_individual,
            generate_top_n=config_backtest.quantstats_top_n,
            benchmark=a2a_manager_config.benchmark,
        )

        # Generate report
        report = generate_report(results, clustering_results, config_backtest, a2a_manager_config)

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
        results_df.to_csv(
            config_backtest.results_dir / "backtest_results.csv", index=False
        )

        # Save Euclidean distance analysis to CSV
        if "euclidean_distance" in clustering_results and "closest_pairs" in clustering_results["euclidean_distance"]:
            distance_data = clustering_results["euclidean_distance"]["closest_pairs"]
            if distance_data:
                distance_df = pd.DataFrame(distance_data)
                distance_df.to_csv(
                    config_backtest.results_dir / "optimizer_distances.csv", index=False
                )
                logger.info("Optimizer distance analysis saved to optimizer_distances.csv")

        logger.info(f"Backtest completed successfully. Results saved to {config_backtest.results_dir}")
        logger.info(f"Report available at: {report_path}")

        # Check QuantStats reports
        logger.info(f"QuantStats reports generated in {config_backtest.results_dir}")
        logger.info("Open HTML files in a web browser for interactive analysis")

        logger.info(f"\n{'='*80}")
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(
            f"Period: {config_backtest.get_report_date_range()[0]} to {config_backtest.get_report_date_range()[1]}"
        )
        logger.info(f"Optimizers tested: {len(results)}")
        logger.info(f"Results directory: {config_backtest.results_dir}")
        logger.info(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main_backtest()
