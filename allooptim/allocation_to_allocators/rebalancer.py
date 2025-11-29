import math
from typing import Optional
import pandas as pd
from allooptim.config.rebalancer_config import RebalancerConfig


class PortfolioRebalancer:
    """
    Multi-strategy portfolio rebalancing engine that minimizes transaction costs
    by combining EMA smoothing, threshold-based rebalancing, and trade filtering.

    The rebalancer applies strategies in sequence:
    1. EMA smoothing on target weights to reduce oscillations
    2. Threshold-based rebalancing with no-trade regions
    3. Minimum trade size filtering
    4. Optional priority ranking to limit number of trades

    Parameters
    ----------
    half_lifetime_days : int
        Half-life of the smoothing filter in days. The time for the filter response
        to decay to half its initial value. Used to calculate the filter constant.

    rebalancing_days : int
        Number of days between rebalancing periods. Used to calculate the filter constant.

    absolute_threshold : float
        Absolute deviation threshold for rebalancing (e.g., 0.03 = 3%).
        Only trade if |current - target| > threshold.

    relative_threshold : float
        Relative deviation threshold as fraction of target weight (e.g., 0.20 = 20%).
        Provides adaptive thresholds for different position sizes.

    min_trade_pct : Optional[float]
        Minimum trade size as a fraction of portfolio value for a trade to be executed.
        Trades smaller than this fraction are filtered out.

    max_trades_per_day : Optional[int]
        Maximum number of trades to execute per rebalancing period.
        If set, only the highest priority trades are executed.
        None means no limit.

    trade_to_band_edge : bool
        If True, trade to the edge of the no-trade region rather than to target.
        This reduces future rebalancing frequency.

    Attributes
    ----------
    _smoothed_weights : Optional[pd.Series]
        Current smoothed target weights maintained across rebalancing periods.

    _actual_weights : pd.Series
        Actual portfolio weights after applying rebalancing logic.
        Tracks the true portfolio state across periods.
    """

    def __init__(
        self,
        rebalancing_days: int,
        config: Optional[RebalancerConfig] = None,
    ) -> None:

        self.config = config or RebalancerConfig()
        self.rebalancing_days = rebalancing_days
        
        # Calculate filter constant for first-order low-pass filter
        # α = exp(-Δt/τ) where τ = lifetime_days
        # This gives the correct smoothing constant for the desired life
        if self.config.filter_lifetime_days > 1000:
            self.filter_constant = 0.0  # No smoothing
        else:
            self.filter_constant = math.exp(
                -rebalancing_days / self.config.filter_lifetime_days
            )

        self._smoothed_weights: Optional[pd.Series] = None
        self._actual_weights: pd.Series = pd.Series(dtype=float)

    def rebalance(self, target_weights: pd.Series) -> pd.Series:
        """
        Determine optimal trades to rebalance portfolio while minimizing transaction costs.

        The method maintains internal state of actual portfolio weights and applies
        smoothing and threshold logic to determine when trades are necessary.

        Parameters
        ----------
        target_weights : pd.Series
            Target portfolio weights indexed by asset identifier.
            Must sum to approximately 1.0.

        Returns
        -------
        pd.Series
            New portfolio weights after applying cost-reduction strategies.
            This represents the actual portfolio weights after trading.
            Use this as the new current_weights for position tracking.
        """
        smoothed_target = self._apply_ema_smoothing(target_weights)

        effective_threshold = self._calculate_effective_threshold(smoothed_target)

        trades = self._identify_threshold_trades(smoothed_target, effective_threshold)

        filtered_trades = self._apply_minimum_trade_filter(trades)

        if self.config.max_trades_per_day is not None:
            final_trades = self._apply_priority_ranking(filtered_trades)
        else:
            final_trades = filtered_trades

        self._update_actual_weights(final_trades, smoothed_target)

        return self._actual_weights.copy()

    def _apply_ema_smoothing(self, target_weights: pd.Series) -> pd.Series:
        """
        Apply first-order low-pass filter smoothing to reduce weight oscillations.
        Maintains state across rebalancing periods.
        """
        if self._smoothed_weights is None:
            self._smoothed_weights = target_weights.copy()
            return self._smoothed_weights

        all_assets = target_weights.index.union(self._smoothed_weights.index)
        target_aligned = target_weights.reindex(all_assets, fill_value=0.0)
        smoothed_aligned = self._smoothed_weights.reindex(all_assets, fill_value=0.0)

        # First-order low-pass filter: y[n] = α * y[n-1] + (1-α) * x[n]
        # where α = filter_constant = exp(-Δt/τ)
        self._smoothed_weights = (
            self.filter_constant * smoothed_aligned + (1 - self.filter_constant) * target_aligned
        )

        total = self._smoothed_weights.sum()
        if total > 0:
            self._smoothed_weights = self._smoothed_weights / total

        return self._smoothed_weights

    def _calculate_effective_threshold(self, target_weights: pd.Series) -> pd.Series:
        """
        Calculate effective threshold for each asset as maximum of absolute and relative thresholds.
        Provides adaptive thresholds that scale with position size.
        """

        relative_band = target_weights * self.config.relative_threshold
        absolute_band = pd.Series(self.config.absolute_threshold, index=target_weights.index)

        return pd.concat([relative_band, absolute_band], axis=1).max(axis=1)

    def _identify_threshold_trades(
        self, target_weights: pd.Series, thresholds: pd.Series
    ) -> pd.Series:
        """
        Identify trades where actual weight deviates beyond threshold band from target.
        Optionally trades to band edge rather than target to reduce future rebalancing.
        """
        all_assets = target_weights.index.union(self._actual_weights.index)
        actual_aligned = self._actual_weights.reindex(all_assets, fill_value=0.0)
        target_aligned = target_weights.reindex(all_assets, fill_value=0.0)
        threshold_aligned = thresholds.reindex(
            all_assets, fill_value=self.config.absolute_threshold
        )

        deviation = (actual_aligned - target_aligned).abs()
        needs_rebalancing = deviation > threshold_aligned

        trades = pd.Series(dtype=float)

        for asset in all_assets[needs_rebalancing]:
            actual = actual_aligned[asset]
            target = target_aligned[asset]
            threshold = threshold_aligned[asset]

            if self.config.trade_to_band_edge:
                if actual > target:
                    new_weight = target + threshold
                else:
                    new_weight = max(0.0, target - threshold)
            else:
                new_weight = target

            trades[asset] = new_weight

        return trades

    def _apply_minimum_trade_filter(
        self, trades: pd.Series
    ) -> pd.Series:
        """
        Filter out trades below minimum dollar threshold to avoid uneconomical executions.
        """
        if len(trades) == 0:
            return trades

        if self.config.min_trade_pct is None:
            return trades

        trade_sizes = trades.abs()
        significant_trades = trade_sizes >= self.config.min_trade_pct

        return trades[significant_trades]

    def _apply_priority_ranking(self, trades: pd.Series) -> pd.Series:
        """
        Rank trades by priority and execute only the top N most important ones.
        Priority is calculated as deviation weighted by position importance.
        """
        if len(trades) == 0 or self.config.max_trades_per_day is None:
            return trades

        actual_aligned = self._actual_weights.reindex(trades.index, fill_value=0.0)

        deviations = (trades - actual_aligned).abs()
        importance = pd.concat([trades, actual_aligned], axis=1).max(axis=1)
        priority = deviations * importance

        top_trades = priority.nlargest(self.config.max_trades_per_day)

        return trades[top_trades.index]

    def _update_actual_weights(
        self, trades: pd.Series, smoothed_target: pd.Series
    ) -> None:
        """
        Update internal tracking of actual portfolio weights after executing trades.
        Assets not traded maintain their previous weights.
        """
        all_assets = smoothed_target.index.union(self._actual_weights.index)

        if len(self._actual_weights) == 0:
            self._actual_weights = trades.copy()
        else:
            self._actual_weights = self._actual_weights.reindex(
                all_assets, fill_value=0.0
            )
            for asset in trades.index:
                self._actual_weights[asset] = trades[asset]

        self._actual_weights = self._actual_weights[self._actual_weights > 1e-10]

        total = self._actual_weights.sum()
        if total > 0:
            self._actual_weights = self._actual_weights / total

    def reset_smoothing(self) -> None:
        """
        Reset the EMA smoothing state and actual weights. Call this when starting
        a new backtest or when discontinuity in target weights occurs.
        """
        self._smoothed_weights = None
        self._actual_weights = pd.Series(dtype=float)

    def get_actual_weights(self) -> pd.Series:
        """
        Get current actual portfolio weights being tracked internally.

        Returns
        -------
        pd.Series
            Current actual portfolio weights after all rebalancing decisions.
        """
        return self._actual_weights.copy()
