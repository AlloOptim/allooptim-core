"""
Utility functions for AlloOptim Jupyter notebooks.

This module provides helper functions specifically designed for use in Jupyter notebooks,
making it easier to work with AlloOptim results, visualizations, and data analysis.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_optimizer_comparison(results: Dict[str, Any], metrics: List[str] = None, top_n: int = 10) -> pd.DataFrame:
    """
    Display a comparison table of optimizer performance metrics.

    Args:
        results: Dictionary of backtest results from BacktestEngine
        metrics: List of metrics to display (default: common metrics)
        top_n: Number of top performers to show

    Returns:
        DataFrame with comparison data
    """
    if metrics is None:
        metrics = ["sharpe_ratio", "cagr", "max_drawdown", "total_return", "volatility"]

    # Extract data
    data = []
    for name, result in results.items():
        if "metrics" in result:
            row = {"optimizer": name}
            row.update(result["metrics"])
            data.append(row)

    df = pd.DataFrame(data)

    # Sort by Sharpe ratio and show top N
    df_sorted = df.sort_values("sharpe_ratio", ascending=False).head(top_n)

    # Format percentages
    for col in ["cagr", "max_drawdown", "total_return", "volatility"]:
        if col in df_sorted.columns:
            df_sorted[col] = (df_sorted[col] * 100).round(2).astype(str) + "%"

    # Format Sharpe ratio
    if "sharpe_ratio" in df_sorted.columns:
        df_sorted["sharpe_ratio"] = df_sorted["sharpe_ratio"].round(3)

    return df_sorted


def plot_returns_distribution(
    results: Dict[str, Any], optimizers: List[str] = None, figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot the distribution of daily returns for selected optimizers.

    Args:
        results: Dictionary of backtest results
        optimizers: List of optimizer names to plot (default: top 5)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if optimizers is None:
        # Get top 5 by Sharpe ratio
        perf_data = []
        for name, result in results.items():
            if "metrics" in result:
                perf_data.append({"name": name, "sharpe": result["metrics"].get("sharpe_ratio", 0)})

        df_perf = pd.DataFrame(perf_data).sort_values("sharpe", ascending=False)
        optimizers = df_perf["name"].head(5).tolist()

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, opt_name in enumerate(optimizers[:6]):  # Max 6 plots
        if opt_name in results and "returns" in results[opt_name]:
            returns = results[opt_name]["returns"]
            axes[i].hist(returns, bins=50, alpha=0.7, density=True)
            axes[i].set_title(f"{opt_name}\nDaily Returns Distribution")
            axes[i].set_xlabel("Daily Return")
            axes[i].set_ylabel("Density")
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(optimizers), 6):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def create_performance_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a comprehensive performance summary DataFrame.

    Args:
        results: Dictionary of backtest results

    Returns:
        DataFrame with all performance metrics
    """
    data = []
    for name, result in results.items():
        if "metrics" in result:
            row = {"optimizer": name}
            row.update(result["metrics"])
            data.append(row)

    df = pd.DataFrame(data)

    # Add ranking columns
    df["sharpe_rank"] = df["sharpe_ratio"].rank(ascending=False).astype(int)
    df["cagr_rank"] = df["cagr"].rank(ascending=False).astype(int)
    df["drawdown_rank"] = df["max_drawdown"].rank(ascending=True).astype(int)  # Lower is better

    # Sort by Sharpe ratio
    df = df.sort_values("sharpe_ratio", ascending=False)

    return df


def plot_cumulative_returns(
    results: Dict[str, Any], optimizers: List[str] = None, benchmark: str = "S&P 500", figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Plot cumulative returns for selected optimizers.

    Args:
        results: Dictionary of backtest results
        optimizers: List of optimizer names to plot
        benchmark: Name of benchmark optimizer
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if optimizers is None:
        # Get top 5 and benchmark
        perf_data = []
        for name, result in results.items():
            if "metrics" in result:
                perf_data.append({"name": name, "sharpe": result["metrics"].get("sharpe_ratio", 0)})

        df_perf = pd.DataFrame(perf_data).sort_values("sharpe", ascending=False)
        optimizers = df_perf["name"].head(5).tolist()

        # Add benchmark if available
        if benchmark in results and benchmark not in optimizers:
            optimizers.append(benchmark)

    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))

    for i, opt_name in enumerate(optimizers):
        if opt_name in results and "cumulative_returns" in results[opt_name]:
            cum_returns = results[opt_name]["cumulative_returns"]
            ax.plot(cum_returns.index, cum_returns.values, label=opt_name, color=colors[i], linewidth=2)

    ax.set_title("Cumulative Returns Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    return fig


def save_notebook_results(
    results: Dict[str, Any], clustering_results: Dict[str, Any] = None, output_dir: str = "notebook_results"
) -> Path:
    """
    Save notebook results to CSV files for further analysis.

    Args:
        results: Dictionary of backtest results
        clustering_results: Dictionary of clustering analysis results
        output_dir: Output directory path

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save performance metrics
    perf_data = []
    for name, result in results.items():
        if "metrics" in result:
            row = {"optimizer": name}
            row.update(result["metrics"])
            perf_data.append(row)

    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        df_perf.to_csv(output_path / "performance_metrics.csv", index=False)

    # Save clustering results if available
    if clustering_results and "euclidean_distance" in clustering_results and "closest_pairs" in clustering_results["euclidean_distance"]:
            distance_data = clustering_results["euclidean_distance"]["closest_pairs"]
            if distance_data:
                df_dist = pd.DataFrame(distance_data)
                df_dist.to_csv(output_path / "optimizer_distances.csv", index=False)

    print(f"Results saved to: {output_path}")
    return output_path


def print_backtest_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of backtest results.

    Args:
        results: Dictionary of backtest results
    """
    print("=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    # Extract metrics
    perf_data = []
    for name, result in results.items():
        if "metrics" in result:
            perf_data.append({"name": name, **result["metrics"]})

    if not perf_data:
        print("No performance data available")
        return

    df = pd.DataFrame(perf_data)

    # Overall statistics
    print(f"Total optimizers tested: {len(df)}")
    print("Test period: 2014-12-31 to 2024-12-31 (10 years)")
    print("Rebalancing: Every 5 trading days")
    print()

    # Top performers
    best_sharpe = df.loc[df["sharpe_ratio"].idxmax()]
    best_cagr = df.loc[df["cagr"].idxmax()]
    best_total_return = df.loc[df["total_return"].idxmax()]

    print("TOP PERFORMERS:")
    print(f"  Best Sharpe Ratio: {best_sharpe['name']} ({best_sharpe['sharpe_ratio']:.3f})")
    print(f"  Best CAGR: {best_cagr['name']} ({best_cagr['cagr']:.1%})")
    print(f"  Best Total Return: {best_total_return['name']} ({best_total_return['total_return']:.1%})")
    print()

    # Risk metrics
    print("RISK METRICS:")
    print(".3f")
    print(".3f")
    print(".3f")
    print()

    # Distribution statistics
    print("PERFORMANCE DISTRIBUTION:")
    print(f"  Sharpe Ratio - Mean: {df['sharpe_ratio'].mean():.3f}, Std: {df['sharpe_ratio'].std():.3f}")
    print(f"  CAGR - Mean: {df['cagr'].mean():.1%}, Std: {df['cagr'].std():.1%}")
    print(f"  Max Drawdown - Mean: {df['max_drawdown'].mean():.1%}, Std: {df['max_drawdown'].std():.1%}")

    print("=" * 80)
