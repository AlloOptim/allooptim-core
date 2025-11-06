import numpy as np
import pandas as pd

# Constants for performance metrics
MIN_DATA_POINTS = 2
DOUBLE_COUNTING_AVOIDANCE_FACTOR = 2


class PerformanceMetrics:
    """Calculate comprehensive performance metrics."""

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate daily returns from price series."""
        return prices.pct_change().dropna()

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns / returns.std()

    @staticmethod
    def max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    @staticmethod
    def time_underwater(prices: pd.Series) -> float:
        """Calculate time underwater as fraction of total time."""
        cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        is_underwater = cumulative < running_max
        return is_underwater.sum() / len(is_underwater)

    @staticmethod
    def cagr(prices: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(prices) < MIN_DATA_POINTS:
            return 0.0

        start_value = prices.iloc[0]
        end_value = prices.iloc[-1]
        years = len(prices) / 252  # Assuming 252 trading days per year

        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0.0

        return (end_value / start_value) ** (1 / years) - 1

    @staticmethod
    def risk_adjusted_return(returns: pd.Series) -> float:
        """Calculate return minus volatility."""
        return returns.mean() * 252 - returns.std() * np.sqrt(252)

    @staticmethod
    def portfolio_turnover(weights_history: pd.DataFrame) -> pd.Series:
        """Calculate portfolio turnover for each period."""
        if len(weights_history) < MIN_DATA_POINTS:
            return pd.Series([], dtype=float)

        turnover = []
        for i in range(1, len(weights_history)):
            prev_weights = weights_history.iloc[i - 1]
            curr_weights = weights_history.iloc[i]

            # Calculate turnover as sum of absolute weight changes
            weight_changes = abs(curr_weights - prev_weights)
            turnover.append(
                weight_changes.sum() / DOUBLE_COUNTING_AVOIDANCE_FACTOR
            )  # Divide by 2 to avoid double counting

        return pd.Series(turnover, index=weights_history.index[1:])

    @staticmethod
    def portfolio_changerate(weights_history: pd.DataFrame) -> pd.Series:
        """Calculate portfolio change rate for each period."""
        if len(weights_history) < MIN_DATA_POINTS:
            return pd.Series([], dtype=float)

        change_rate = []
        for i in range(1, len(weights_history)):
            prev_weights = weights_history.iloc[i - 1]
            curr_weights = weights_history.iloc[i]

            # Calculate change rate as sum of absolute weight changes
            weight_changes = (curr_weights - prev_weights) / prev_weights
            change_rate.append(weight_changes.mean())

        return pd.Series(change_rate, index=weights_history.index[1:])

    @staticmethod
    def portfolio_invested_assets(
        weights_history: pd.DataFrame,
        rel_threshold: float,
    ) -> pd.Series:
        """Calculate portfolio invested assets for each period.

        Count all assets with more than 5% of equal weight portfolio weights.
        """
        if len(weights_history) < MIN_DATA_POINTS:
            return pd.Series([], dtype=float)

        n_all_assets = weights_history.shape[1]
        abs_threshold = rel_threshold / n_all_assets

        invested_assets = []
        for i in range(len(weights_history)):
            curr_weights = weights_history.iloc[i]
            count = (curr_weights > abs_threshold).sum()
            invested_assets.append(count)

        return pd.Series(invested_assets, index=weights_history.index)

    @staticmethod
    def portfolio_invested_top_n(
        weights_history: pd.DataFrame,
        top_n: int,
    ) -> pd.Series:
        """Calculate portfolio invested assets for each period.

        count the combined weight of the top n assets.
        """
        if len(weights_history) < MIN_DATA_POINTS:
            return pd.Series([], dtype=float)

        invested_assets = []
        for i in range(len(weights_history)):
            curr_weights = weights_history.iloc[i]
            top_n_weights = curr_weights.nlargest(top_n)
            invested_assets.append(top_n_weights.sum())

        return pd.Series(invested_assets, index=weights_history.index)
