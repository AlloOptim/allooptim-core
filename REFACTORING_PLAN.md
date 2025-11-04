# Orchestrator-Only Backtest Refactoring Plan

## Overview
Remove adapter layer and legacy optimizer loops. Backtest directly uses AllocationOrchestrator, extracting per-optimizer results from df_allocation.

## Architecture Changes

### Before
```
BacktestEngine
  ├─ _setup_optimizers() → AllocationOrchestratorAdapter OR individual optimizers
  └─ run_backtest()
      ├─ Loop over individual_optimizers
      ├─ Loop over ensemble_optimizers
      └─ Track weights/memory/time per optimizer
```

### After
```
BacktestEngine
  ├─ __init__() → Create AllocationOrchestrator directly
  └─ run_backtest()
      ├─ orchestrator.run_allocation() → AllocationResult
      ├─ Compute SPY benchmark separately
      └─ Store AllocationResults (one per timestep)
  
  └─ _calculate_portfolio_performance()
      ├─ Extract df_allocation from AllocationResults
      ├─ Reconstruct weights_history per optimizer
      └─ Calculate performance for each optimizer + SPY
```

## Implementation Phases

### Phase 1: Data Models (Tasks 1-3)
**Files**: `backtest_config.py`, `allocation_dataclasses.py`

- Remove `use_orchestrator` boolean from BacktestConfig
- Make `orchestration_type` required field (no default)
- Add `optimizer_memory_usage: Optional[dict[str, float]]` to AllocationResult
- Add `optimizer_computation_time: Optional[dict[str, float]]` to AllocationResult
- Add `algo_memory_usage: dict[str, float]` to A2AStatistics
- Add `algo_computation_time: dict[str, float]` to A2AStatistics

### Phase 2: Orchestrator Enhancements (Tasks 4-5)
**File**: `allocation_orchestrator.py`

- In `_compute_final_asset_weights()`:
  - Wrap each `optimizer.allocate()` with `tracemalloc`
  - Track `memory_usage[optimizer.name]` (peak MB)
  - Track `computation_time[optimizer.name]` (seconds)
  - Return `(asset_weights_dict, statistics_dict, df_allocation, memory_usage, computation_time)`

- Update all three `run_allocation()` methods:
  - `_run_equal_allocation_to_allocators()`
  - `_run_optimized_allocation_to_allocators()`
  - `_run_wikipedia_and_equal_allocation()`
  - Pass `optimizer_memory_usage` and `optimizer_computation_time` to AllocationResult
  - Pass to A2AStatistics constructor

### Phase 3: Remove Adapter (Task 6)
**Files**: `orchestrator_adapter.py`, `backtest_engine.py`

- Delete `allo_optim/allocation_to_allocators/orchestrator_adapter.py`
- Remove import from `backtest_engine.py`
- Remove imports from any test files

### Phase 4: Simplify BacktestEngine (Tasks 7-9)
**File**: `backtest_engine.py`

**Task 7 - __init__**:
```python
def __init__(self, config_instance: BacktestConfig = None):
    self.config = config_instance or config
    self.data_loader = DataLoader()
    
    # Create orchestrator directly
    self.orchestrator = AllocationOrchestrator(
        optimizer_names=self.config.optimizer_names,
        transformer_names=self.config.transformer_names,
        config=AllocationOrchestratorConfig(
            orchestration_type=self.config.orchestration_type
        )
    )
    
    self.results = {}
```

**Task 8 - run_backtest main loop**:
```python
# Initialize tracking
allocation_results = []  # Store AllocationResult per timestep
benchmark_weights_history = []  # SPY benchmark

for rebalance_date in rebalance_dates:
    # Get data...
    
    # Single orchestrator call
    allocation_result = self.orchestrator.run_allocation(
        all_stocks=all_stocks,
        time_today=rebalance_date,
        df_prices=clean_data
    )
    allocation_results.append(allocation_result)
    
    # SPY benchmark (Task 9)
    spy_weights = pd.Series(0.0, index=mu.index)
    if 'SPY' in spy_weights.index:
        spy_weights['SPY'] = 1.0
    benchmark_weights_history.append(spy_weights)

# Calculate performance (Task 10)
self.results = self._calculate_portfolio_performance(
    price_data, allocation_results, benchmark_weights_history, rebalance_dates
)
```

### Phase 5: Results Processing (Tasks 10-11)
**File**: `backtest_engine.py`

**Task 10 - _calculate_portfolio_performance**:
```python
def _calculate_portfolio_performance(
    self,
    price_data: pd.DataFrame,
    allocation_results: list[AllocationResult],
    benchmark_weights_history: list[pd.Series],
    rebalance_dates: list[datetime],
) -> dict:
    # Extract per-optimizer weights_history from df_allocation
    optimizer_names = set()
    for result in allocation_results:
        if result.df_allocation is not None:
            optimizer_names.update(result.df_allocation.index)
    
    # Build weights_history dict: {optimizer_name: list[pd.Series]}
    weights_history = {name: [] for name in optimizer_names}
    for result in allocation_results:
        if result.df_allocation is not None:
            for opt_name in optimizer_names:
                if opt_name in result.df_allocation.index:
                    weights_history[opt_name].append(result.df_allocation.loc[opt_name])
    
    # Accumulate memory/time per optimizer
    memory_stats = {name: [] for name in optimizer_names}
    time_stats = {name: [] for name in optimizer_names}
    for result in allocation_results:
        if result.optimizer_memory_usage:
            for name, mem in result.optimizer_memory_usage.items():
                memory_stats[name].append(mem)
        if result.optimizer_computation_time:
            for name, time in result.optimizer_computation_time.items():
                time_stats[name].append(time)
    
    # Calculate performance for each optimizer
    results = {}
    for optimizer_name in optimizer_names:
        portfolio_values = self._simulate_portfolio(
            price_data, weights_history[optimizer_name], rebalance_dates
        )
        # Calculate metrics...
        results[optimizer_name] = {...}
    
    # Add SPY benchmark
    weights_history['SPY_Benchmark'] = benchmark_weights_history
    portfolio_values = self._simulate_portfolio(price_data, benchmark_weights_history, rebalance_dates)
    results['SPY_Benchmark'] = {...}
    
    return results
```

**Task 11**: `_simulate_portfolio` stays unchanged

### Phase 6: Validation (Tasks 12-13)

**Task 12**: Verify `backtest_report.py` and `backtest_visualizer.py` work with new results dict

**Task 13**: 
- Delete `tests/test_orchestrator_adapter.py`
- Delete `tests/test_orchestrator_backtest.py`
- Create new comprehensive test verifying:
  - AllocationResult has df_allocation
  - Memory/time metrics populated
  - Reports show all optimizers + SPY
  - Clustering analysis works

## Expected Benefits

✅ **Simpler**: No adapter, no optimizer loops  
✅ **Cleaner**: 1 call per timestep instead of N  
✅ **Complete**: All optimizer data in df_allocation  
✅ **Efficient**: Orchestrator handles optimization strategy  
✅ **Maintainable**: Single source of truth for allocations  

## Breaking Changes

⚠️ `use_orchestrator` config removed - always uses orchestrator  
⚠️ No direct optimizer instantiation in BacktestEngine  
⚠️ Adapter pattern eliminated  

---
*Total: 13 tasks across 6 phases*
