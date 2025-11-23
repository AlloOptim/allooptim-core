# Backtesting Guide

This guide covers comprehensive backtesting of portfolio optimization strategies using AlloOptim's backtesting framework.

## Overview

AlloOptim provides a complete backtesting framework supporting:

- Walk-forward analysis with configurable rebalancing
- Multiple performance metrics
- Benchmark comparisons
- QuantStats integration for professional reporting
- Clustering analysis of optimizer behaviors
- Visualization tools

## Quick Start

```python
from allooptim.config.backtest_config import BacktestConfig
from allooptim.backtest.backtest_engine import BacktestEngine
from datetime import datetime

# Configure backtest
config = BacktestConfig(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1),
    rebalance_frequency=21,  # Monthly
    optimizer_configs=[
        "MaxSharpeOptimizer",
        "HRPOptimizer",
        "RiskParityOptimizer"
    ]
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run_backtest()

print(f"Tested {len(results)} optimizers")
```

## Configuration

### BacktestConfig

The main configuration class for backtesting scenarios.

```python
from allooptim.config.backtest_config import BacktestConfig
from allooptim.stock_universe import large_stock_universe, extract_symbols_from_list

config = BacktestConfig(
    # Time period
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1),

    # Rebalancing
    rebalance_frequency=21,  # Days between rebalancing
    lookback_days=252,       # Historical lookback window

    # Assets
    symbols=extract_symbols_from_list(large_stock_universe()),

    # Strategies
    optimizer_configs=[
        "MaxSharpeOptimizer",
        "HRPOptimizer",
        "RiskParityOptimizer",
        "NaiveOptimizer"
    ],

    # Covariance processing
    transformer_names=["OracleCovarianceTransformer"],

    # Benchmark
    benchmark="SPY",

    # Performance options
    log_returns=True,
    quick_test=False,

    # Reporting
    quantstats_individual=True,
    quantstats_top_n=5
)
```

### Key Parameters

#### Time Configuration
- `start_date`, `end_date`: Backtest period
- `rebalance_frequency`: Days between portfolio rebalancing
- `lookback_days`: Historical data window for optimization

#### Asset Universe
- `symbols`: List of ticker symbols to include
- `benchmark`: Benchmark ticker (e.g., "SPY", "^GSPC")

#### Strategy Selection
- `optimizer_configs`: List of optimizer names or configurations
- `transformer_names`: Covariance matrix transformers

#### Performance Options
- `log_returns`: Use log returns vs. arithmetic returns
- `quick_test`: Reduced dataset for faster testing

## Running Backtests

### Basic Backtest

```python
from allooptim.backtest.backtest_engine import BacktestEngine

engine = BacktestEngine(config)
results = engine.run_backtest()
```

### With A2A Configuration

```python
from allooptim.config.a2a_config import A2AConfig

a2a_config = A2AConfig(
    ensemble_method="equal_weight",
    risk_parity=True
)

engine = BacktestEngine(config, a2a_config)
results = engine.run_backtest()
```

## Results Analysis

### Results Structure

The backtest returns a dictionary where each key is an optimizer name and values contain:

```python
{
    'MaxSharpeOptimizer': {
        'metrics': {
            'total_return': 0.85,
            'annualized_return': 0.12,
            'annualized_volatility': 0.18,
            'sharpe_ratio': 0.67,
            'sortino_ratio': 1.02,
            'max_drawdown': -0.25,
            'calmar_ratio': 0.48,
            # ... more metrics
        },
        'weights': pd.DataFrame(...),  # Time series of weights
        'returns': pd.Series(...),     # Portfolio returns
        'benchmark_returns': pd.Series(...)
    }
}
```

### Performance Metrics

AlloOptim calculates comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return, alpha, beta
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio, CVaR
- **Drawdown Metrics**: Maximum drawdown, Calmar ratio, recovery time
- **Benchmark-relative**: Tracking error, information ratio

## Reporting

### QuantStats Reports

Generate professional HTML tearsheets:

```python
from allooptim.backtest.backtest_quantstats import create_quantstats_reports

create_quantstats_reports(
    results=results,
    output_dir="./reports",
    generate_individual=True,    # Individual optimizer reports
    generate_top_n=5,           # Top N performers report
    benchmark=config.benchmark
)
```

This creates:
- `quantstats_report.html`: Complete analysis
- `individual/`: Reports for each optimizer
- `top_5_report.html`: Top performers comparison

### Markdown Reports

Generate comprehensive markdown reports:

```python
from allooptim.backtest.backtest_report import generate_report

report = generate_report(results, clustering_results, config)
with open("backtest_report.md", "w") as f:
    f.write(report)
```

### CSV Export

Export results to CSV for further analysis:

```python
import pandas as pd

# Performance metrics
metrics_data = []
for name, data in results.items():
    row = {"optimizer": name}
    row.update(data["metrics"])
    metrics_data.append(row)

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv("performance_metrics.csv", index=False)

# Portfolio weights (example for one optimizer)
weights_df = results["MaxSharpeOptimizer"]["weights"]
weights_df.to_csv("maxsharpe_weights.csv")
```

## Clustering Analysis

Analyze optimizer behavior patterns:

```python
from allooptim.backtest.cluster_analyzer import ClusterAnalyzer

cluster_analyzer = ClusterAnalyzer(results)
clustering_results = cluster_analyzer.analyze_clusters()
```

### Clustering Results

```python
{
    'correlation_matrix': pd.DataFrame(...),
    'clusters': {
        'cluster_0': ['MaxSharpeOptimizer', 'EfficientReturnOptimizer'],
        'cluster_1': ['HRPOptimizer', 'RiskParityOptimizer'],
        # ...
    },
    'euclidean_distance': {
        'closest_pairs': [
            {'pair': ['OptA', 'OptB'], 'distance': 0.15},
            # ...
        ]
    }
}
```

## Visualization

Create comprehensive visualizations:

```python
from allooptim.backtest.backtest_visualizer import create_visualizations

create_visualizations(
    results=results,
    clustering_results=clustering_results,
    output_dir="./reports"
)
```

Creates:
- Performance comparison charts
- Risk-return scatter plots
- Drawdown analysis
- Weight evolution over time
- Clustering dendrograms

## Advanced Features

### Custom Optimizers

Test your own optimizers:

```python
from allooptim.optimizer.optimizer_interface import AbstractOptimizer

class CustomOptimizer(AbstractOptimizer):
    @property
    def name(self) -> str:
        return "CustomOptimizer"

    def allocate(self, mu, cov, **kwargs):
        # Your logic here
        return weights

# Add to config
config.optimizer_configs.append("CustomOptimizer")
```

### Transaction Costs

Account for transaction costs:

```python
# In development - coming soon
config.transaction_costs = 0.001  # 10bps per trade
```

### Risk Management

Add risk constraints:

```python
# In development - coming soon
config.max_weight = 0.2  # Maximum 20% per asset
config.min_weight = 0.0
```

## Best Practices

### 1. Start Small

Begin with quick tests:

```python
config = BacktestConfig(
    start_date=datetime(2022, 1, 1),  # Shorter period
    end_date=datetime(2024, 1, 1),
    rebalance_frequency=63,  # Quarterly rebalancing
    quick_test=True,         # Faster execution
    symbols=["AAPL", "MSFT", "GOOGL"]  # Fewer assets
)
```

### 2. Validate Data

Check data quality before backtesting:

```python
from allooptim.backtest.data_validator import validate_backtest_data

is_valid, issues = validate_backtest_data(config)
if not is_valid:
    print("Data issues found:", issues)
```

### 3. Compare Strategies

Always include benchmarks:

```python
config.optimizer_configs = [
    "MaxSharpeOptimizer",
    "HRPOptimizer",
    "RiskParityOptimizer",
    "NaiveOptimizer"  # Equal weight benchmark
]
```

### 4. Analyze Robustness

Test across different market conditions:

```python
# Bull market period
bull_config = BacktestConfig(
    start_date=datetime(2020, 4, 1),
    end_date=datetime(2021, 12, 31),
    # ... other settings
)

# Bear market period
bear_config = BacktestConfig(
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 10, 31),
    # ... other settings
)
```

### 5. Monitor Performance

Track key metrics:

```python
def analyze_results(results):
    metrics = ['sharpe_ratio', 'max_drawdown', 'total_return']

    for optimizer, data in results.items():
        print(f"\n{optimizer}:")
        for metric in metrics:
            value = data['metrics'][metric]
            print(f"  {metric}: {value:.3f}")

analyze_results(results)
```

## Performance Optimization

### Memory Management

For large backtests:

```python
# Use fewer assets
config.symbols = config.symbols[:50]

# Reduce lookback window
config.lookback_days = 126  # 6 months

# Less frequent rebalancing
config.rebalance_frequency = 63  # Quarterly
```

### Parallel Processing

Enable parallel optimization (future feature):

```python
# In development
config.parallel_optimization = True
config.n_jobs = -1  # Use all cores
```

## Troubleshooting

### Common Issues

1. **No results generated**
   - Check date range validity
   - Verify symbols have price data
   - Ensure optimizers are properly configured

2. **Poor performance**
   - Validate input data quality
   - Check for look-ahead bias
   - Consider transaction costs

3. **Memory errors**
   - Reduce number of assets
   - Use shorter lookback periods
   - Enable quick_test mode

4. **Optimization failures**
   - Check covariance matrix conditioning
   - Validate expected returns
   - Review optimizer constraints

### Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run backtest with logging
engine = BacktestEngine(config)
results = engine.run_backtest()
```

## Example: Complete Analysis

```python
from allooptim.config.backtest_config import BacktestConfig
from allooptim.backtest.backtest_engine import BacktestEngine
from allooptim.backtest.backtest_quantstats import create_quantstats_reports
from allooptim.backtest.backtest_report import generate_report
from allooptim.backtest.backtest_visualizer import create_visualizations
from allooptim.backtest.cluster_analyzer import ClusterAnalyzer
from datetime import datetime

def run_complete_analysis():
    # Configuration
    config = BacktestConfig(
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2024, 1, 1),
        rebalance_frequency=21,
        optimizer_configs=[
            "MaxSharpeOptimizer", "HRPOptimizer", "RiskParityOptimizer",
            "NaiveOptimizer", "BlackLittermanOptimizer"
        ],
        benchmark="SPY"
    )

    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest()

    # Clustering analysis
    cluster_analyzer = ClusterAnalyzer(results)
    clustering_results = cluster_analyzer.analyze_clusters()

    # Generate reports
    create_quantstats_reports(results, "./reports")
    create_visualizations(results, clustering_results, "./reports")

    report = generate_report(results, clustering_results, config)
    with open("./reports/backtest_report.md", "w") as f:
        f.write(report)

    print("Analysis complete! Check ./reports/ for results.")

if __name__ == "__main__":
    run_complete_analysis()
```