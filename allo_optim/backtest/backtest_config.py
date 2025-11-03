import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# Configuration
class BacktestConfig:
    """Backtest configuration parameters."""

    RERAISE_ALLOCATOR_EXCEPTIONS = False
    LOG_RETURNS = True

    # Time periods
    START_DATE = datetime(2014, 12, 31)
    END_DATE = datetime(2024, 12, 31)
    DEBUG_START_DATE = datetime(2022, 12, 31)
    DEBUG_END_DATE = datetime(2023, 2, 28)

    # Quick test mode for debugging
    QUICK_TEST = True

    # Rebalancing and lookback
    REBALANCE_FREQUENCY = 10
    LOOKBACK_DAYS = 60

    # Fallback behavior
    USE_EQUAL_WEIGHTS_FALLBACK = True  # True for equal weights, False for zero weights
    
    OPTIMIZER_NAMES = ["CMA_MEAN_VARIANCE",
    "CMA_L_MOMENTS",
    "CMA_SORTINO",
    "CMA_MAX_DRAWDOWN",
    "CMA_ROBUST_SHARPE",
    "CMA_CVAR",
    "PSO_MeanVariance",
    "PSO_LMoments",
    "HRP",
    "NCOSharpeOptimizer",
    "Naive",
    "CappedMomentum",
    "RiskParity",
    "AdjustedReturns_MeanVariance",
    "AdjustedReturns_EMA",
    "MaxSharpe",
    "EfficientReturn",
    "EfficientRisk",
    ]
    
    TRANSFORMER_NAMES = [
        "OracleCovarianceTransformer",
    ]

    # Output
    RESULTS_DIR = Path("backtest_results") / datetime.now().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def get_report_date_range(cls) -> tuple[datetime, datetime]:
        """Get start and end dates based on debug mode."""

        if cls.QUICK_TEST:
            return cls.DEBUG_START_DATE, cls.DEBUG_END_DATE
        return cls.START_DATE, cls.END_DATE

    @classmethod
    def get_data_date_range(cls) -> tuple[datetime, datetime]:
        """Get start and end dates based on debug mode."""
        previous_days = timedelta(days=cls.LOOKBACK_DAYS)

        if cls.QUICK_TEST:
            return cls.DEBUG_START_DATE - previous_days, cls.DEBUG_END_DATE
        return cls.START_DATE - previous_days, cls.END_DATE
