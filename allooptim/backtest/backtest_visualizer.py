"""Visualization utilities for backtest results and analysis.

This module provides comprehensive plotting and visualization capabilities
for portfolio backtesting results. It creates charts for performance comparison,
clustering analysis, and statistical insights using matplotlib and seaborn.

Key visualizations:
- Portfolio performance time series plots
- Risk-return scatter plots
- Clustering dendrograms and heatmaps
- Performance metric comparisons
- Benchmark-relative performance charts
- Statistical distribution plots
- Automated chart saving and formatting
"""

import logging
from pathlib import Path
from typing import Optional

from scipy.cluster.hierarchy import dendrogram

from allooptim.backtest.backtest_config import BacktestConfig

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    visualization_dependencies_available = True
except ImportError:
    logger.info("Visualization dependencies are not available. Skipping visualization methods.")
    visualization_dependencies_available = False


def create_visualizations(results: dict, clustering_results: dict, results_dir: Optional[Path] = None) -> None:
    """Create comprehensive visualizations of results."""

    if not visualization_dependencies_available:
        logger.warning("Visualization dependencies are not available. Skipping visualization.")
        return None

    if results_dir is None:
        config = BacktestConfig()
        results_dir = config.results_dir

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating visualizations")

    # Set up plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Performance comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Sharpe ratios
    sharpe_ratios = {name: data["metrics"].get("sharpe_ratio", 0) for name, data in results.items()}
    axes[0, 0].bar(range(len(sharpe_ratios)), list(sharpe_ratios.values()))
    axes[0, 0].set_title("Sharpe Ratio by Optimizer")
    axes[0, 0].set_xticks(range(len(sharpe_ratios)))
    axes[0, 0].set_xticklabels(list(sharpe_ratios.keys()), rotation=45, ha="right")

    # Max drawdowns
    max_drawdowns = {name: data["metrics"].get("max_drawdown", 0) for name, data in results.items()}
    axes[0, 1].bar(range(len(max_drawdowns)), list(max_drawdowns.values()))
    axes[0, 1].set_title("Maximum Drawdown by Optimizer")
    axes[0, 1].set_xticks(range(len(max_drawdowns)))
    axes[0, 1].set_xticklabels(list(max_drawdowns.keys()), rotation=45, ha="right")

    # CAGR
    cagrs = {name: data["metrics"].get("cagr", 0) for name, data in results.items()}
    axes[1, 0].bar(range(len(cagrs)), list(cagrs.values()))
    axes[1, 0].set_title("CAGR by Optimizer")
    axes[1, 0].set_xticks(range(len(cagrs)))
    axes[1, 0].set_xticklabels(list(cagrs.keys()), rotation=45, ha="right")

    # Risk-adjusted returns
    risk_adj_returns = {name: data["metrics"].get("risk_adjusted_return", 0) for name, data in results.items()}
    axes[1, 1].bar(range(len(risk_adj_returns)), list(risk_adj_returns.values()))
    axes[1, 1].set_title("Risk-Adjusted Return by Optimizer")
    axes[1, 1].set_xticks(range(len(risk_adj_returns)))
    axes[1, 1].set_xticklabels(list(risk_adj_returns.keys()), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(results_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Portfolio value evolution
    plt.figure(figsize=(14, 8))
    for name, data in results.items():
        if "portfolio_values" in data and not data["portfolio_values"].empty:
            portfolio_values = data["portfolio_values"]
            normalized_values = portfolio_values / portfolio_values.iloc[0] * 100
            plt.plot(portfolio_values.index, normalized_values, label=name, alpha=0.8)

    plt.title("Portfolio Value Evolution (Normalized to 100)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "portfolio_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Risk-return scatter plot
    plt.figure(figsize=(12, 8))
    returns = [data["metrics"].get("annual_return", 0) for data in results.values()]
    volatilities = [data["metrics"].get("annual_volatility", 0) for data in results.values()]
    names = list(results.keys())

    plt.scatter(volatilities, returns, s=100, alpha=0.7)

    for i, name in enumerate(names):
        plt.annotate(name, (volatilities[i], returns[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    plt.title("Risk-Return Profile")
    plt.xlabel("Annual Volatility")
    plt.ylabel("Annual Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "risk_return_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Clustering dendrogram (if hierarchical clustering available)
    if "performance" in clustering_results and "linkage_matrix" in clustering_results["performance"]:
        plt.figure(figsize=(12, 6))
        dendrogram(
            clustering_results["performance"]["linkage_matrix"],
            labels=clustering_results["performance"]["optimizer_names"],
            leaf_rotation=45,
        )
        plt.title("Hierarchical Clustering of Optimizers (Performance-based)")
    plt.tight_layout()
    plt.savefig(results_dir / "clustering_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Visualizations saved to results directory")
