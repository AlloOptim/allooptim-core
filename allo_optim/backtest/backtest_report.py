import logging
from datetime import datetime
from typing import Optional

from allo_optim.backtest.backtest_config import BacktestConfig

logger = logging.getLogger(__name__)


def generate_report(results: dict, clustering_results: dict, config: Optional[BacktestConfig] = None) -> str:
    """Generate comprehensive markdown report."""

    if config is None:
        config = BacktestConfig()

    logger.info("Generating comprehensive report")

    # Get date range for report
    start_date, end_date = config.get_report_date_range()

    report = f"""# Comprehensive Allocation Algorithm Backtest Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}  
**Rebalancing Frequency:** Every {config.rebalance_frequency} trading days  
**Lookback Window:** {config.lookback_days} days  
**Fallback Strategy:** {'Equal Weights' if config.use_equal_weights_fallback else 'Zero Weights'}  

## Executive Summary

This report presents a comprehensive backtest of {len(results)} allocation algorithms including:
- 13 individual optimizers from the enhanced allocation framework
- A2A ensemble optimizer (simple average of all individual optimizers)  
- S&P 500 benchmark (100% SPY allocation)

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
#### Computational Performance

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
## Optimizer Clustering Analysis

The clustering analysis groups optimizers based on their performance characteristics,
portfolio similarities, and return patterns to identify which algorithms behave similarly.

"""

    # Add clustering results
    for cluster_type, cluster_data in clustering_results.items():
        if isinstance(cluster_data, dict) and "clusters" in cluster_data:
            report += f"""
### {cluster_type.replace('_', ' ').title()} Clustering

**Method:** {cluster_data.get('method', 'Unknown')}  
**Number of Clusters:** {cluster_data.get('n_clusters', 0)}  

"""
            for cluster_id, optimizers in cluster_data["clusters"].items():
                report += f"**Cluster {cluster_id}:** {', '.join(optimizers)}\n\n"

    # Add Euclidean distance analysis
    if "euclidean_distance" in clustering_results:
        euclidean_data = clustering_results["euclidean_distance"]

        report += """
### Euclidean Distance Analysis

This analysis computes the mean Euclidean distance between optimizer portfolio weights
across all timesteps, revealing which optimizers make the most similar allocation decisions.

"""

        # Add closest pairs table
        if "closest_pairs" in euclidean_data and euclidean_data["closest_pairs"]:
            report += """
#### Most Similar Optimizer Pairs (Shortest Distances)

| Rank | Optimizer A | Optimizer B | Mean Euclidean Distance |
|------|-------------|-------------|-------------------------|
"""

            for i, pair in enumerate(euclidean_data["closest_pairs"][:10], 1):
                report += (
                    f"| {i} | {pair['optimizer_a']} | {pair['optimizer_b']} | {pair['mean_euclidean_distance']:.4f} |\n"
                )

        # Add clustering results based on distances
        if "clustering" in euclidean_data and "clusters" in euclidean_data["clustering"]:
            report += f"""

#### Distance-Based Groupings

Using hierarchical clustering on Euclidean distances, optimizers are grouped into
{euclidean_data['clustering'].get('n_clusters', 0)} clusters:

"""
            for cluster_id, optimizers in euclidean_data["clustering"]["clusters"].items():
                report += f"**Distance Cluster {cluster_id}:** {', '.join(optimizers)}\n\n"

        report += """
**Key Insights:**
- Optimizers with small Euclidean distances make very similar allocation decisions
- Distance-based clusters reveal functional similarity beyond theoretical groupings
- The closest pairs often represent variations of the same underlying approach
- Large distances indicate fundamentally different allocation strategies

"""

    report += """
## Theoretical Optimizer Groupings

Based on the underlying optimization approaches, we can group the algorithms theoretically:

### Mean Reversion & Risk Parity Group
- **RiskParityOptimizer:** Equal risk contribution
- **NaiveOptimizer:** Equal weight allocation
- **EfficientRiskOptimizer:** Risk-based allocation

### Modern Portfolio Theory Group  
- **MeanVarianceParticalSwarmOptimizer:** PSO with mean-variance optimization
- **MeanVarianceAdjustedReturnsOptimizer:** Classical mean-variance with adjusted returns
- **MaxSharpeOptimizer:** Maximum Sharpe ratio optimization

### Alternative Risk Models Group
- **LMomentsParticleSwarmOptimizer:** PSO with L-moments
- **LMomentsAdjustedReturnsOptimizer:** L-moments based allocation
- **HRPOptimizer:** Hierarchical risk parity

### Advanced Optimization Group
- **NCOOptimizer:** Nested clustered optimization
- **MomentumOptimizer:** Momentum-based allocation
- **CongressSenateOptimizer:** Congress trading patterns

### Market-Based Group
- **MarketCapOptimizer:** Market capitalization weighted
- **SPY_Benchmark:** S&P 500 benchmark

### Ensemble Group
- **A2A_Ensemble:** Average of all individual optimizers

## Key Insights and Recommendations

### Performance Insights
"""

    if results:
        # Calculate some insights
        spy_performance = results.get("SPY_Benchmark", {}).get("metrics", {})
        a2a_performance = results.get("A2A_Ensemble", {}).get("metrics", {})

        if spy_performance and a2a_performance:
            outperformed = a2a_performance.get("sharpe_ratio", 0) > spy_performance.get("sharpe_ratio", 0)
            report += f"""
1. **Benchmark Comparison**: The S&P 500 benchmark achieved a Sharpe ratio of
{spy_performance.get('sharpe_ratio', 0):.3f} vs A2A ensemble of {a2a_performance.get('sharpe_ratio', 0):.3f}
2. **Ensemble Effect**: The A2A ensemble {'outperformed' if outperformed else 'underperformed'}
the S&P 500 benchmark
"""

        # Add diversification insights
        most_concentrated = min(results.items(), key=lambda x: x[1]["metrics"].get("invested_top_5_mean", 1))
        most_diversified = max(results.items(), key=lambda x: x[1]["metrics"].get("invested_5_abs_mean", 0))

        report += f"""
3. **Concentration Analysis**: {most_concentrated[0]} is most concentrated
(top 5 assets: {most_concentrated[1]['metrics'].get('invested_top_5_mean', 0)*100:.1f}%)
4. **Diversification Leader**: {most_diversified[0]} uses most assets above 5% threshold
(avg: {most_diversified[1]['metrics'].get('invested_5_abs_mean', 0):.1f} assets)
"""

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

        report += f"""
5. **Clustering Analysis**: {cluster_analysis.get('n_clusters', 0)} clusters identified
(avg cluster size: {cluster_analysis.get('avg_cluster_size', 0):.1f} assets)
6. **Risk-Return Profile**: {best_sharpe[0]} leads with Sharpe ratio
{sharpe_ratio:.2f} (avg return: {best_sharpe[1]['metrics'].get('avg_return', 0)*100:.2f}%)
"""

    report += f"""
### Algorithm Clustering Insights

1. **Performance Clustering**: Identifies optimizers with similar risk-return profiles
2. **Portfolio Correlation**: Groups algorithms that tend to select similar assets
3. **Returns Pattern**: Clusters based on return distribution characteristics

### Recommendations

1. **Diversification Strategy**: Use optimizers from different theoretical and experimental clusters
2. **Ensemble Optimization**: The A2A approach shows promise for risk diversification
3. **Computational Efficiency**: Consider computation time vs performance trade-offs
4. **Market Regime Sensitivity**: Monitor performance across different market conditions

## Technical Details

### Data Quality
- **Universe Size**: Approximately 400 assets from Alpaca-available universe
- **Data Source**: Yahoo Finance via yfinance library
- **Missing Data Handling**: Forward fill with 80% completeness threshold

### Methodology
- **Rebalancing**: Portfolio weights updated every 5 trading days
- **Lookback Window**: 90 days of historical data for each optimization
- **Execution**: Perfect execution assumed (no slippage, transaction costs, or liquidity constraints)
- **Fallback Strategy**: Equal weights used when optimizers fail

### Risk Considerations
- **Survivorship Bias**: Only includes currently available assets
- **Look-Ahead Bias**: Avoided by using only historical data at each rebalancing point
- **Transaction Costs**: Not included in performance calculations
- **Market Impact**: Not considered due to perfect execution assumption

## Appendix

### Configuration Parameters
- **Start Date**: {start_date.strftime('%Y-%m-%d')}
- **End Date**: {end_date.strftime('%Y-%m-%d')}
- **Rebalancing Frequency**: {config.rebalance_frequency} trading days
- **Lookback Period**: {config.lookback_days} days
- **Fallback Strategy**: {'Equal Weights' if config.use_equal_weights_fallback else 'Zero Weights'}
- **Results Directory**: {config.results_dir}

### Generated Files
- `performance_comparison.png`: Performance metrics bar charts
- `portfolio_evolution.png`: Portfolio value time series  
- `risk_return_scatter.png`: Risk-return scatter plot
- `clustering_dendrogram.png`: Hierarchical clustering visualization
- `backtest_results.csv`: Detailed results in CSV format
- `optimizer_distances.csv`: Pairwise Euclidean distances between optimizers

---
*This report was generated automatically by the comprehensive backtest framework.*
"""

    return report
