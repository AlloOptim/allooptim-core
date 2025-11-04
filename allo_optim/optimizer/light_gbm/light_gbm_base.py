import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Constants for model training thresholds
MIN_FEATURES_FOR_MODEL_TRAINING = 50


class FastPortfolioOptimizer:
    """
    State-of-the-art portfolio optimizer using:
    1. LightGBM for return prediction (fast training)
    2. Online covariance estimation with exponential weighting
    3. Risk-aware optimization with transaction costs
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int = 252,
        decay: float = 0.94,
        risk_aversion: float = 2.0,
        transaction_cost: float = 0.001,
    ) -> None:
        """
        Args:
            n_assets: Number of assets in portfolio
            lookback: Historical window for feature engineering
            decay: Exponential decay for covariance (0.94 = ~20 day half-life)
            risk_aversion: Risk aversion parameter (higher = more conservative)
        """
        self.n_assets = n_assets
        self.lookback = lookback
        self.decay = decay
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost

        # Model for each asset's returns
        self.models = []
        self.scalers = []
        self.cov_estimator = LedoitWolf()
        self.trained = False  # Track if models have been successfully trained

        # Online statistics
        self.ewm_cov = None
        self.last_weights = np.ones(n_assets) / n_assets

        # Training data buffers for incremental learning
        self.feature_buffer = []
        self.target_buffer = []
        self.max_buffer_size = 1000

    def _engineer_features(self, prices: np.ndarray) -> list[np.ndarray]:
        """Create features from price data"""
        features = []

        for i in range(self.n_assets):
            asset_prices = prices[:, i]

            # Returns at multiple horizons
            returns_1d = np.diff(np.log(asset_prices))
            returns_5d = (np.log(asset_prices[5:]) - np.log(asset_prices[:-5])) / 5
            returns_20d = (np.log(asset_prices[20:]) - np.log(asset_prices[:-20])) / 20

            # Volatility features
            vol_20d = pd.Series(returns_1d).rolling(20).std().values

            # Momentum features
            mom_20d = asset_prices[20:] / asset_prices[:-20] - 1
            mom_60d = asset_prices[60:] / asset_prices[:-60] - 1

            # RSI-like features
            gains = np.maximum(returns_1d, 0)
            losses = np.maximum(-returns_1d, 0)
            avg_gain = pd.Series(gains).rolling(14).mean().values
            avg_loss = pd.Series(losses).rolling(14).mean().values
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Combine features (align to shortest length)
            min_len = min(
                len(returns_1d), len(returns_5d), len(returns_20d), len(vol_20d), len(mom_20d), len(mom_60d), len(rsi)
            )

            if min_len == 0:
                asset_features = np.array([]).reshape(0, 7)
            else:
                asset_features = np.column_stack(
                    [
                        returns_1d[-min_len:],
                        returns_5d[-min_len:],
                        returns_20d[-min_len:],
                        vol_20d[-min_len:],
                        mom_20d[-min_len:],
                        mom_60d[-min_len:],
                        rsi[-min_len:],
                    ]
                )

            features.append(asset_features)

        return features

    def _update_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Update covariance matrix using exponential weighting"""
        if self.ewm_cov is None:
            self.ewm_cov = np.cov(returns.T)
        else:
            new_cov = np.outer(returns[-1], returns[-1])
            self.ewm_cov = self.decay * self.ewm_cov + (1 - self.decay) * new_cov

        # Shrinkage for stability
        target = np.eye(self.n_assets) * np.trace(self.ewm_cov) / self.n_assets
        shrinkage = 0.1
        return (1 - shrinkage) * self.ewm_cov + shrinkage * target

    def train(self, prices: np.ndarray, returns: np.ndarray = None) -> None:
        """
        Fast training using LightGBM with early stopping

        Args:
            prices: (T, n_assets) historical prices
            returns: (T-1, n_assets) historical returns (optional, calculated from prices if not provided)
        """
        # Check minimum data requirement
        min_required_periods = 61  # Need at least 61 periods for 60-day momentum
        if len(prices) < min_required_periods:
            logger.warning(
                f"Insufficient data for LightGBM training: {len(prices)} periods provided, "
                f"need at least {min_required_periods}. Using equal weights fallback."
            )
            # Initialize with equal weights fallback
            self.trained = False
            return

        # Estimate returns from prices
        if returns is None:
            returns = np.diff(np.log(prices), axis=0)

        logger.debug("Engineering features...")
        features = self._engineer_features(prices)

        logger.debug("Training models...")
        for i in range(self.n_assets):
            X = features[i][:-1]  # All but last
            y = returns[-len(X) :, i]  # Corresponding future returns

            # Check if we have any samples after feature engineering
            if len(X) == 0:
                logger.error(
                    f"No samples available for asset {i} after feature engineering. "
                    f"Price data shape: {prices.shape}, Features shape: {features[i].shape}"
                )
                # Initialize with equal weights fallback
                self.trained = False
                return

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # LightGBM with fast parameters
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

            feature_names = [f"feature_{j}" for j in range(X_scaled.shape[1])]
            X_df = pd.DataFrame(X_scaled, columns=feature_names)
            model.fit(X_df, y)

            self.models.append(model)
            self.scalers.append(scaler)

        # Update covariance
        self.ewm_cov = self._update_covariance(returns[-min(len(returns), 252) :])

        self.trained = True
        logger.debug("Training complete! Models ready for daily updates.")

    def incremental_update(self, new_prices: np.ndarray, new_returns: np.ndarray = None) -> None:
        """
        Fast incremental update with new data (runs in <1 second)

        Args:
            new_prices: (lookback+1, n_assets) recent prices including new day
            new_returns: (1, n_assets) latest returns (optional, calculated from prices if not provided)
        """
        # Estimate returns from prices
        if new_returns is None:
            returns_window = np.diff(np.log(new_prices), axis=0)
            new_returns = returns_window[-1:]  # Last return (today's return)
        else:
            returns_window = np.diff(np.log(new_prices), axis=0)

        # Update covariance (very fast)
        self.ewm_cov = self._update_covariance(returns_window)

        # Add to buffer for periodic retraining
        features = self._engineer_features(new_prices)
        for i in range(self.n_assets):
            if len(features[i]) > 0:
                self.feature_buffer.append((i, features[i][-1]))
                self.target_buffer.append(new_returns[0, i])

        # Periodic full retrain (e.g., every 20 days)
        if len(self.feature_buffer) > self.max_buffer_size:
            self._periodic_retrain()

    def _periodic_retrain(self) -> None:
        """Retrain models with buffered data"""
        for i in range(self.n_assets):
            asset_features = [f for idx, f in self.feature_buffer if idx == i]
            asset_targets = [self.target_buffer[j] for j, (idx, _) in enumerate(self.feature_buffer) if idx == i]

            if len(asset_features) > MIN_FEATURES_FOR_MODEL_TRAINING:
                X = np.array(asset_features)
                y = np.array(asset_targets)
                X_scaled = self.scalers[i].transform(X)

                # Warm start from existing model
                self.models[i].fit(X_scaled, y)

        self.feature_buffer = []
        self.target_buffer = []

    def predict(self, current_prices: np.ndarray) -> np.ndarray:
        """
        Predict returns and optimize portfolio (runs in milliseconds)

        Args:
            current_prices: (lookback, n_assets) recent prices

        Returns:
            optimal_weights: (n_assets,) portfolio weights
        """
        # If not trained, return equal weights
        if not self.trained:
            logger.warning("Model not trained, returning equal weights")
            return np.ones(self.n_assets) / self.n_assets

        # Generate features for prediction
        features = self._engineer_features(current_prices)

        # Predict returns
        predicted_returns = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if len(features[i]) > 0:
                X = features[i][-1].reshape(1, -1)
                X_scaled = self.scalers[i].transform(X)

                # Convert to DataFrame with same feature names used during training
                import pandas as pd

                feature_names = [f"feature_{j}" for j in range(X_scaled.shape[1])]
                X_df = pd.DataFrame(X_scaled, columns=feature_names)
                predicted_returns[i] = self.models[i].predict(X_df)[0]

        # Optimize with transaction costs
        optimal_weights = self._optimize_portfolio(predicted_returns, self.ewm_cov)

        # expected_return = np.dot(optimal_weights, predicted_returns)

        return optimal_weights

    def _optimize_portfolio(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Mean-variance optimization with transaction costs
        """
        n = len(mu)

        def objective(w):
            port_return = np.dot(w, mu)
            port_risk = np.sqrt(np.dot(w, np.dot(cov, w)))
            turnover = np.sum(np.abs(w - self.last_weights))
            return -(port_return - self.risk_aversion * port_risk - self.transaction_cost * turnover)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Fully invested
        ]

        bounds = tuple((0, 1) for _ in range(n))  # Long only

        # Warm start from current weights
        result = minimize(
            objective,
            x0=self.last_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )

        self.last_weights = result.x
        return result.x
