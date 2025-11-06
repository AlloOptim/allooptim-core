"""Enhanced Allocation Workflow

Clean orchestration of the complete allocation-to-allocators process.
Integrates MCOS simulation with allocation optimization.
"""

import logging

from allooptim.optimizer.covariance_matrix_adaption.cma_optimizer import (
    CVARCMAOptimizer,
    LMomentsCMAOptimizer,
    MaxDrawdownCMAOptimizer,
    MeanVarianceCMAOptimizer,
    RobustSharpeCMAOptimizer,
    SortinoCMAOptimizer,
)
from allooptim.optimizer.deep_learning.deep_learning_optimizer import (
    LSTMOptimizer,
    MAMBAOptimizer,
    TCNOptimizer,
)
from allooptim.optimizer.efficient_frontier.black_litterman_optimizer import (
    BlackLittermanOptimizer,
)
from allooptim.optimizer.efficient_frontier.efficient_frontier_optimizer import (
    EfficientReturnOptimizer,
    EfficientRiskOptimizer,
    MaxSharpeOptimizer,
)
from allooptim.optimizer.fundamental.fundamental_optimizer import (
    BalancedFundamentalOptimizer,
    MarketCapFundamentalOptimizer,
    QualityGrowthFundamentalOptimizer,
    ValueInvestingFundamentalOptimizer,
)
from allooptim.optimizer.hierarchical_risk_parity.hrp_optimizer import HRPOptimizer
from allooptim.optimizer.kelly_criterion.kelly_criterion_optimizer import (
    KellyCriterionOptimizer,
)
from allooptim.optimizer.light_gbm.light_gbm_optimizer import AugmentedLightGBMOptimizer, LightGBMOptimizer
from allooptim.optimizer.naive.momentum_optimizer import (
    EMAMomentumOptimizer,
    MomentumOptimizer,
)
from allooptim.optimizer.naive.naive_optimizer import (
    NaiveOptimizer,
)
from allooptim.optimizer.nested_cluster.nco_optimizer import NCOSharpeOptimizer
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from allooptim.optimizer.particle_swarm.pso_optimizer import (
    LMomentsParticleSwarmOptimizer,
    MeanVarianceParticleSwarmOptimizer,
)
from allooptim.optimizer.sequential_quadratic_programming.adjusted_return_optimizer import (
    EMAAdjustedReturnsOptimizer,
    LMomentsAdjustedReturnsOptimizer,
    MeanVarianceAdjustedReturnsOptimizer,
    SemiVarianceAdjustedReturnsOptimizer,
)
from allooptim.optimizer.sequential_quadratic_programming.higher_moments_optimizer import (
    HigherMomentOptimizer,
)
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import (
    RiskParityOptimizer,
)
from allooptim.optimizer.sequential_quadratic_programming.robust_mean_variance_optimizer import (
    RobustMeanVarianceOptimizer,
)
from allooptim.optimizer.wikipedia.wikipedia_optimizer import WikipediaOptimizer

logger = logging.getLogger(__name__)


OPTIMIZER_LIST: list[type[AbstractOptimizer]] = [
    MeanVarianceCMAOptimizer,
    LMomentsCMAOptimizer,
    SortinoCMAOptimizer,
    MaxDrawdownCMAOptimizer,
    RobustSharpeCMAOptimizer,
    CVARCMAOptimizer,
    MeanVarianceParticleSwarmOptimizer,
    LMomentsParticleSwarmOptimizer,
    HRPOptimizer,
    NCOSharpeOptimizer,
    NaiveOptimizer,
    MomentumOptimizer,
    EMAMomentumOptimizer,
    RiskParityOptimizer,
    MeanVarianceAdjustedReturnsOptimizer,
    EMAAdjustedReturnsOptimizer,
    LMomentsAdjustedReturnsOptimizer,
    SemiVarianceAdjustedReturnsOptimizer,
    HigherMomentOptimizer,
    MarketCapFundamentalOptimizer,
    BalancedFundamentalOptimizer,
    QualityGrowthFundamentalOptimizer,
    ValueInvestingFundamentalOptimizer,
    MaxSharpeOptimizer,
    EfficientReturnOptimizer,
    EfficientRiskOptimizer,
    WikipediaOptimizer,
    LightGBMOptimizer,
    AugmentedLightGBMOptimizer,
    KellyCriterionOptimizer,
    RobustMeanVarianceOptimizer,
    BlackLittermanOptimizer,
    MAMBAOptimizer,
    LSTMOptimizer,
    TCNOptimizer,
]


def get_all_optimizer_names() -> list[str]:
    """Get the list of all available optimizer names."""
    return [optimizer().name for optimizer in OPTIMIZER_LIST]


def get_all_optimizers() -> list[AbstractOptimizer]:
    """Get instances of all available optimizers."""
    return [optimizer() for optimizer in OPTIMIZER_LIST]


def get_optimizer_by_names(names: list[str]) -> list[AbstractOptimizer]:
    """Retrieve optimizer instances by their names."""
    all_optimizers = get_all_optimizers()
    name_to_optimizer = {optimizer.name: optimizer for optimizer in all_optimizers}

    for name in names:
        if name not in name_to_optimizer:
            logger.warning(
                f"Optimizer '{name}' is not recognized. Available optimizers: {list(name_to_optimizer.keys())}"
            )

    selected_optimizers = [optimizer for name, optimizer in name_to_optimizer.items() if name in names]

    if len(selected_optimizers) == 0:
        logger.error("No valid optimizers found for the provided names.")

    return selected_optimizers
