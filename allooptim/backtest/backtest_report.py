"""Backtest report generation and formatting utilities.

This module provides comprehensive reporting functionality for portfolio
backtesting results. It generates detailed markdown reports with performance
metrics, visualizations, and statistical analysis for comparing different
optimization strategies.

Key features:
- Comprehensive markdown report generation
- Performance metrics summary and analysis
- Clustering analysis visualization
- Statistical significance testing
- Benchmark comparisons
- Automated report formatting and structure
- Configurable reporting periods and metrics
"""

import logging
from datetime import datetime
from typing import Optional

from allooptim.allocation_to_allocators.a2a_manager_config import A2AManagerConfig
from allooptim.config.backtest_config import BacktestConfig

logger = logging.getLogger(__name__)


def generate_report(results: dict, clustering_results: dict, config_backtest: Optional[BacktestConfig] = None, a2a_manager_config: Optional[A2AManagerConfig] = None) -> str:
    """Generate comprehensive markdown report."""
    if config_backtest is None:
        config_backtest = BacktestConfig()
    if a2a_manager_config is None:
        a2a_manager_config = A2AManagerConfig()

    logger.info("Generating comprehensive report")

    # Get date range for report
    start_date, end_date = config_backtest.get_report_date_range()

    # Calculate number of individual optimizers (excluding ensemble and benchmark)
    individual_optimizers = [name for name in results if name not in ["A2AEnsemble", "SPY"]]
    n_individual = len(individual_optimizers)

    report = f"""# Comprehensive Allocation Algorithm Backtest Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}  
**Rebalancing Frequency:** Every {config_backtest.rebalance_frequency} trading days  
**Lookback Window:** {a2a_manager_config.lookback_days} days  
**Number of Individual Optimizers Tested:** {n_individual}
**Fallback Strategy:** {'Equal Weights' if a2a_manager_config.use_equal_weights_fallback else 'Zero Weights'}  

## Executive Summary

### Key Findings

"""

    # Find best performers
    if results:
        best_sharpe = max(results.items(), key=lambda x: x[1]["metrics"].get("sharpe_ratio", -999))
        best_return = max(results.items(), key=lambda x: x[1]["metrics"].get("cagr", -999))
        lowest_drawdown = min(results.items(), key=lambda x: x[1]["metrics"].get("max_drawdown", 999))

        report += f"""
**Best Sharpe Ratio:** {best_sharpe[0]} ({best_sharpe[1]['metrics'].get('sharpe_ratio', 0):.3f})
**Best CAGR:** {best_return[0]} ({best_return[1]['metrics'].get('cagr', 0)*100:.2f}%)
**Lowest Max Drawdown:** {lowest_drawdown[0]} ({lowest_drawdown[1]['metrics'].get('max_drawdown', 0)*100:.2f}%)
"""

    report += """
## Performance Metrics

### Summary Statistics

| Optimizer | Sharpe Ratio | CAGR | Max Drawdown | Annual Vol | Risk-Adj Return | Total Return |
|-----------|--------------|------|--------------|------------|-----------------|--------------|
"""

    # Add performance table
    for name, data in results.items():
        metrics = data["metrics"]
        report += (
            f"| {name} | {metrics.get('sharpe_ratio', 0):.3f} | "
            f"{metrics.get('cagr', 0)*100:.2f}% | "
            f"{metrics.get('max_drawdown', 0)*100:.2f}% | "
            f"{metrics.get('annual_volatility', 0)*100:.2f}% | "
            f"{metrics.get('risk_adjusted_return', 0)*100:.2f}% | "
            f"{metrics.get('total_return', 0)*100:.2f}% |\n"
        )

    report += """
### Detailed Performance Analysis

#### Returns Distribution Statistics

| Optimizer | Mean Daily Return | Volatility | Skewness | Kurtosis | Min Return | Max Return |
|-----------|-------------------|------------|----------|----------|------------|------------|
"""

    for name, data in results.items():
        metrics = data["metrics"]
        report += (
            f"| {name} | {metrics.get('returns_mean', 0)*100:.4f}% | "
            f"{metrics.get('returns_std', 0)*100:.4f}% | "
            f"{metrics.get('returns_skew', 0):.3f} | "
            f"{metrics.get('returns_kurtosis', 0):.3f} | "
            f"{metrics.get('returns_min', 0)*100:.3f}% | "
            f"{metrics.get('returns_max', 0)*100:.3f}% |\n"
        )

    report += """
#### Portfolio Turnover Analysis

| Optimizer | Mean Turnover | Turnover Std | Min Turnover | Max Turnover | Median Turnover |
|-----------|---------------|--------------|--------------|--------------|-----------------|
"""

    for name, data in results.items():
        metrics = data["metrics"]
        report += (
            f"| {name} | {metrics.get('turnover_mean', 0)*100:.2f}% | "
            f"{metrics.get('turnover_std', 0)*100:.2f}% | "
            f"{metrics.get('turnover_min', 0)*100:.2f}% | "
            f"{metrics.get('turnover_max', 0)*100:.2f}% | "
            f"{metrics.get('turnover_q50', 0)*100:.2f}% |\n"
        )

    report += """
#### Portfolio Change Rate Analysis

| Optimizer | Mean Change Rate | Change Rate Std | Min Change Rate | Max Change Rate | Median Change Rate |
|-----------|------------------|-----------------|-----------------|-----------------|-------------------|
"""

    for name, data in results.items():
        metrics = data["metrics"]
        report += (
            f"| {name} | {metrics.get('change_rate_mean', 0)*100:.2f}% | "
            f"{metrics.get('change_rate_std', 0)*100:.2f}% | "
            f"{metrics.get('change_rate_min', 0)*100:.2f}% | "
            f"{metrics.get('change_rate_max', 0)*100:.2f}% | "
            f"{metrics.get('change_rate_q50', 0)*100:.2f}% |\n"
        )

    report += """
#### Portfolio Diversification Metrics

##### Assets Above Threshold (Mean Count)

| Optimizer | 5% Above Equal Weight | 10% Above Equal Weight | 50% Above Equal Weight | 100% Above Equal Weight |
|-----------|----------------------|------------------------|------------------------|-------------------------|
"""

    for name, data in results.items():
        metrics = data["metrics"]
        report += (
            f"| {name} | "
            f"{metrics.get('invested_5_abs_mean', 0):.1f} "
            f"({metrics.get('invested_5_pct_mean', 0)*100:.1f}%) | "
            f"{metrics.get('invested_10_abs_mean', 0):.1f} "
            f"({metrics.get('invested_10_pct_mean', 0)*100:.1f}%) | "
            f"{metrics.get('invested_50_abs_mean', 0):.1f} "
            f"({metrics.get('invested_50_pct_mean', 0)*100:.1f}%) | "
            f"{metrics.get('invested_100_abs_mean', 0):.1f} "
            f"({metrics.get('invested_100_pct_mean', 0)*100:.1f}%) |\n"
        )

    report += """
##### Top N Assets Weight Concentration

| Optimizer | Top 5 Assets Weight | Top 10 Assets Weight | Top 50 Assets Weight |
|-----------|--------------------|--------------------|---------------------|
"""

    for name, data in results.items():
        metrics = data["metrics"]
        report += (
            f"| {name} | {metrics.get('invested_top_5_mean', 0)*100:.1f}% | "
            f"{metrics.get('invested_top_10_mean', 0)*100:.1f}% | "
            f"{metrics.get('invested_top_50_mean', 0)*100:.1f}% |\n"
        )

    report += """
#### Computational Performance (Aggregated Across Backtest)

| Optimizer | Avg Computation Time (s) | Max Computation Time (s) | Avg Memory Usage (MB) | Max Memory Usage (MB) |
|-----------|---------------------------|---------------------------|------------------------|------------------------|
"""

    for name, data in results.items():
        metrics = data["metrics"]
        report += (
            f"| {name} | {metrics.get('avg_computation_time', 0):.4f} | "
            f"{metrics.get('max_computation_time', 0):.4f} | "
            f"{metrics.get('avg_memory_usage_mb', 0):.2f} | "
            f"{metrics.get('max_memory_usage_mb', 0):.2f} |\n"
        )

    report += """

**Note:** These values represent averages and maxima across all backtest rebalancing periods.
For detailed per-run analysis, see the Detailed Computational Analysis section below.

"""

    report += "### Per-Optimizer Runtime and Memory Usage\n\n"
    report += "This section provides detailed computational performance metrics for each optimizer\n"
    report += "across all backtest rebalancing periods, showing actual runtime and memory consumption\n"
    report += "measured during each allocation computation.\n\n"

    # We need to access the raw A2AResult data to get per-optimizer metrics
    # For now, we'll show the aggregated statistics with additional context
    if results:
        report += "#### Computational Performance Summary (Aggregated)\n\n"
        report += "| Optimizer | Avg Runtime (s) | Max Runtime (s) | Avg Memory (MB) | Max Memory (MB) | Total Runs |\n"
        report += "|-----------|-----------------|-----------------|-----------------|-----------------|------------|\n"

        for name, data in results.items():
            if name == "A2AEnsemble":  # Skip ensemble results
                continue
            metrics = data["metrics"]
            # Count the number of rebalancing periods from weights history
            n_runs = len(data.get("weights_history", []))
            report += (
                f"| {name} | {metrics.get('avg_computation_time', 0):.4f} | "
                f"{metrics.get('max_computation_time', 0):.4f} | "
                f"{metrics.get('avg_memory_usage_mb', 0):.2f} | "
                f"{metrics.get('max_memory_usage_mb', 0):.2f} | {n_runs} |\n"
            )

        report += "\n**Notes:**\n"
        report += "- Runtime and memory measurements are taken during actual optimizer execution\n"
        report += "- Values represent averages/maxima across all backtest rebalancing periods\n"
        report += "- Memory usage shows peak consumption during allocation computation\n"
        report += "- Failed optimizer runs are excluded from averages but counted in total runs\n\n"

    report += "## Optimizer Clustering Analysis\n\n"
    report += "The clustering analysis groups optimizers based on their performance characteristics,\n"
    report += "portfolio similarities, and return patterns to identify which algorithms behave similarly.\n\n"

    # Add clustering results
    for cluster_type, cluster_data in clustering_results.items():
        if isinstance(cluster_data, dict) and "clusters" in cluster_data:
            report += f"### {cluster_type.replace('_', ' ').title()} Clustering\n\n"
            report += f"**Method:** {cluster_data.get('method', 'Unknown')}\n"
            report += f"**Number of Clusters:** {cluster_data.get('n_clusters', 0)}\n\n"
            for cluster_id, optimizers in cluster_data["clusters"].items():
                report += f"**Cluster {cluster_id}:** {', '.join(optimizers)}\n\n"

    # Add Euclidean distance analysis
    if "euclidean_distance" in clustering_results:
        euclidean_data = clustering_results["euclidean_distance"]

        report += "### Euclidean Distance Analysis\n\n"
        report += "This analysis computes the mean Euclidean distance between optimizer portfolio weights\n"
        report += "across all timesteps, revealing which optimizers make the most similar allocation decisions.\n\n"

        # Add closest pairs table
        if "closest_pairs" in euclidean_data and euclidean_data["closest_pairs"]:
            report += "#### Most Similar Optimizer Pairs (Shortest Distances)\n\n"
            report += "| Rank | Optimizer A | Optimizer B | Mean Euclidean Distance |\n"
            report += "|------|-------------|-------------|-------------------------|\n"

            for i, pair in enumerate(euclidean_data["closest_pairs"][:10], 1):
                report += (
                    f"| {i} | {pair['optimizer_a']} | {pair['optimizer_b']} | {pair['mean_euclidean_distance']:.4f} |\n"
                )

        # Add clustering results based on distances
        if "clustering" in euclidean_data and "clusters" in euclidean_data["clustering"]:
            report += "#### Distance-Based Groupings\n\n"
            report += "Using hierarchical clustering on Euclidean distances, optimizers are grouped into\n"
            report += f"{euclidean_data['clustering'].get('n_clusters', 0)} clusters:\n\n"
            for cluster_id, optimizers in euclidean_data["clustering"]["clusters"].items():
                report += f"**Distance Cluster {cluster_id}:** {', '.join(optimizers)}\n\n"

        report += "**Key Insights:**\n"
        report += "- Optimizers with small Euclidean distances make very similar allocation decisions\n"
        report += "- Distance-based clusters reveal functional similarity beyond theoretical groupings\n"
        report += "- The closest pairs often represent variations of the same underlying approach\n"
        report += "- Large distances indicate fundamentally different allocation strategies\n\n"


    report += "## Key Insights and Recommendations\n\n"
    report += "### Performance Insights\n"

    if results:
        # Calculate some insights
        spy_performance = results.get("SPY", {}).get("metrics", {})
        a2a_performance = results.get("A2AEnsemble", {}).get("metrics", {})

        if spy_performance and a2a_performance:
            outperformed = a2a_performance.get("sharpe_ratio", 0) > spy_performance.get("sharpe_ratio", 0)
            report += "1. **Benchmark Comparison**: The S&P 500 benchmark achieved a Sharpe ratio of\n"
            report += f"{spy_performance.get('sharpe_ratio', 0):.3f} vs A2A ensemble of {a2a_performance.get('sharpe_ratio', 0):.3f}\n"
            report += (
                f"2. **Ensemble Effect**: The A2A ensemble {'outperformed' if outperformed else 'underperformed'}\n"
            )
            report += "the S&P 500 benchmark\n"

        # Add diversification insights
        most_concentrated = min(results.items(), key=lambda x: x[1]["metrics"].get("invested_top_5_mean", 1))
        most_diversified = max(results.items(), key=lambda x: x[1]["metrics"].get("invested_5_abs_mean", 0))

        report += f"3. **Concentration Analysis**: {most_concentrated[0]} is most concentrated\n"
        report += f"(top 5 assets: {most_concentrated[1]['metrics'].get('invested_top_5_mean', 0)*100:.1f}%)\n"
        report += f"4. **Diversification Leader**: {most_diversified[0]} uses most assets above 5% threshold\n"
        report += f"(avg: {most_diversified[1]['metrics'].get('invested_5_abs_mean', 0):.1f} assets)\n"

        # Calculate clustering insights
        cluster_analysis = {}
        if clustering_results:
            # Find the main clustering result (usually performance or correlation based)
            for _, cluster_data in clustering_results.items():
                if isinstance(cluster_data, dict) and "n_clusters" in cluster_data:
                    cluster_analysis = cluster_data
                    break

            if cluster_analysis:
                total_optimizers = sum(len(optimizers) for optimizers in cluster_analysis.get("clusters", {}).values())
                avg_cluster_size = (
                    total_optimizers / cluster_analysis.get("n_clusters", 1)
                    if cluster_analysis.get("n_clusters", 0) > 0
                    else 0
                )
                cluster_analysis["avg_cluster_size"] = avg_cluster_size

        sharpe_ratio = best_sharpe[1]["metrics"].get("sharpe_ratio", 0)

        report += f"5. **Clustering Analysis**: {cluster_analysis.get('n_clusters', 0)} clusters identified\n"
        report += f"(avg cluster size: {cluster_analysis.get('avg_cluster_size', 0):.1f} assets)\n"
        report += f"6. **Risk-Return Profile**: {best_sharpe[0]} leads with Sharpe ratio\n"
        report += f"{sharpe_ratio:.2f} (avg return: {best_sharpe[1]['metrics'].get('avg_return', 0)*100:.2f}%)\n"

    return report
