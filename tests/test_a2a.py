"""Tests for A2A (Allocation-to-Allocators) orchestrators."""

import numpy as np
import pandas as pd
import pytest

from allooptim.allocation_to_allocators.a2a_orchestrator import AbstractA2AOrchestrator
from allooptim.allocation_to_allocators.a2a_result import A2AResult
from allooptim.allocation_to_allocators.observation_simulator import (
    MuCovPartialObservationSimulator,
)
from allooptim.allocation_to_allocators.orchestrator_factory import (
    OrchestratorType,
    create_orchestrator,
)
from allooptim.config.a2a_config import A2AConfig
from allooptim.config.cash_config import AllowCashOption, CashConfig
from allooptim.config.stock_universe import list_of_dax_stocks
from allooptim.covariance_transformer.transformer_list import TRANSFORMER_LIST
from allooptim.optimizer.optimizer_config import OptimizerConfig
from allooptim.optimizer.optimizer_list import OPTIMIZER_LIST
from tests.conftest import (
    FAST_TEST_ITERATIONS,
    FAST_TEST_OBSERVATIONS,
    FAST_TEST_PARTICLES,
)


@pytest.mark.parametrize("optimizer_names", [["NaiveOptimizer"], ["NaiveOptimizer", "MomentumOptimizer"]])
@pytest.mark.parametrize("orchestrator_type", OrchestratorType)
def test_a2a(orchestrator_type, optimizer_names):
    """Test that all A2A allocators work correctly."""
    # Convert optimizer names to OptimizerConfig objects
    optimizer_configs = [OptimizerConfig(name=name) for name in optimizer_names]

    # Create sample price data for optimizers that need it
    all_stocks = list_of_dax_stocks()[:5]
    assets = [stock.symbol for stock in all_stocks]
    dates = pd.date_range("2025-09-05", periods=50, freq="D")
    price_data = np.random.randn(50, len(assets)).cumsum(axis=0) + 100
    prices = pd.DataFrame(price_data, index=dates, columns=assets)

    if len(optimizer_names) > 1:
        custom_a2a_weights = {"NaiveOptimizer": 0.7, "MomentumOptimizer": 0.3}
    else:
        custom_a2a_weights = {"NaiveOptimizer": 1.0}

    fast_a2a_config = A2AConfig(
        n_simulations=FAST_TEST_OBSERVATIONS,
        n_particles=FAST_TEST_PARTICLES,
        n_pso_iterations=FAST_TEST_ITERATIONS,
        custom_a2a_weights=custom_a2a_weights,
        cash_config=CashConfig(allow_cash_option=AllowCashOption.GLOBAL_FORBID_CASH),
    )

    # For Wikipedia pipeline, add specific kwargs
    kwargs = {}
    if orchestrator_type == OrchestratorType.WIKIPEDIA_PIPELINE:
        kwargs = {
            "n_historical_days": 30,
            "use_wiki_database": False,
        }

    orchestrator = create_orchestrator(
        orchestrator_type=orchestrator_type,
        optimizer_configs=optimizer_configs,
        transformer_names=["OracleCovarianceTransformer"],
        a2a_config=fast_a2a_config,
        **kwargs,
    )

    # Create data provider for backtest context
    data_provider = MuCovPartialObservationSimulator(prices, n_observations=fast_a2a_config.n_simulations)

    result = orchestrator.allocate(
        data_provider=data_provider,
        time_today=prices.index[-1],
        all_stocks=all_stocks,
    )

    WEIGHT_SUM_TOLERANCE = 1e-6

    assert isinstance(result, A2AResult)
    assert len(result.final_allocation) == len(assets)
    assert all(0 <= w <= 1.0 for w in result.final_allocation.values)
    assert abs(sum(result.final_allocation.values) - 1.0) < WEIGHT_SUM_TOLERANCE  # Should sum to 1.0

    # Verify df_allocation is populated correctly
    df_allocation = result.to_dataframe()
    assert df_allocation is not None
    assert isinstance(df_allocation, pd.DataFrame)
    assert df_allocation.shape[0] == len(assets)  # rows = assets
    assert df_allocation.shape[1] == len(optimizer_names)  # cols = optimizers
    assert list(df_allocation.index) == assets  # same asset order


@pytest.mark.parametrize("orchestrator_type", OrchestratorType)
def test_create_orchestrator_vary_a2a(orchestrator_type):
    """Test creating orchestrator with different orchestrator types."""
    optimizer_configs = [
        OptimizerConfig(name=OPTIMIZER_LIST[0]().name),
        OptimizerConfig(name=OPTIMIZER_LIST[1]().name),
    ]

    transformer_list = [TRANSFORMER_LIST[0]().name, TRANSFORMER_LIST[1]().name]

    # Create custom A2A weights for CUSTOM_WEIGHT orchestrator type
    custom_a2a_weights = None
    if orchestrator_type == OrchestratorType.CUSTOM_WEIGHT:
        custom_a2a_weights = {
            OPTIMIZER_LIST[0]().name: 0.6,
            OPTIMIZER_LIST[1]().name: 0.4,
        }

    a2a_config = A2AConfig(custom_a2a_weights=custom_a2a_weights) if custom_a2a_weights else None

    orchestrator = create_orchestrator(
        orchestrator_type=orchestrator_type,
        optimizer_configs=optimizer_configs,
        transformer_names=transformer_list,
        a2a_config=a2a_config,
    )

    assert isinstance(orchestrator, AbstractA2AOrchestrator)


@pytest.mark.parametrize("optimizer", OPTIMIZER_LIST)
def test_create_orchestrator_vary_optimizer(optimizer):
    """Test creating orchestrator with different optimizers."""
    optimizer_configs = [
        OptimizerConfig(name=optimizer().name),
        OptimizerConfig(name=OPTIMIZER_LIST[0]().name),
    ]

    transformer_list = [TRANSFORMER_LIST[0]().name, TRANSFORMER_LIST[1]().name]

    orchestrator = create_orchestrator(
        orchestrator_type=OrchestratorType.EQUAL_WEIGHT,
        optimizer_configs=optimizer_configs,
        transformer_names=transformer_list,
    )

    assert isinstance(orchestrator, AbstractA2AOrchestrator)


@pytest.mark.parametrize("transformer", TRANSFORMER_LIST)
def test_create_orchestrator_vary_transformer(transformer):
    """Test creating orchestrator with different transformers."""
    optimizer_configs = [
        OptimizerConfig(name=OPTIMIZER_LIST[0]().name),
        OptimizerConfig(name=OPTIMIZER_LIST[1]().name),
    ]

    transformer_list = [transformer().name, TRANSFORMER_LIST[0]().name]

    orchestrator = create_orchestrator(
        orchestrator_type=OrchestratorType.EQUAL_WEIGHT,
        optimizer_configs=optimizer_configs,
        transformer_names=transformer_list,
    )

    assert isinstance(orchestrator, AbstractA2AOrchestrator)
