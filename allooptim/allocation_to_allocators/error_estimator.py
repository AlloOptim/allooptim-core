"""Error Estimator Registry

Extensible registry for MCOS error estimation metrics.
Provides a plugin architecture for different error estimation methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class AbstractErrorEstimator(ABC):
    """Abstract base class for error estimators.

    Error estimators compute metrics that quantify the difference between
    predicted and actual values in MCOS simulations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier for this error estimator."""
        pass

    @abstractmethod
    def estimate_error(self, predicted: np.ndarray, actual: np.ndarray, **kwargs) -> float:
        """Estimate the error between predicted and actual values.

        Args:
            predicted: Predicted values (e.g., estimated covariance matrix)
            actual: Actual/ground truth values
            **kwargs: Additional estimator-specific parameters

        Returns:
            Error metric value (lower is better)
        """
        pass


class MeanSquaredErrorEstimator(AbstractErrorEstimator):
    """Mean Squared Error estimator for matrix-valued predictions."""

    @property
    def name(self) -> str:
        """Get the estimator name identifier."""
        return "mse"

    def estimate_error(self, predicted: np.ndarray, actual: np.ndarray, **kwargs) -> float:
        """Compute Mean Squared Error between predicted and actual matrices.

        Args:
            predicted: Predicted matrix
            actual: Actual matrix
            **kwargs: Unused

        Returns:
            MSE value
        """
        return np.mean((predicted - actual) ** 2)


class FrobeniusNormEstimator(AbstractErrorEstimator):
    """Frobenius norm estimator for matrix error."""

    @property
    def name(self) -> str:
        """Get the estimator name identifier."""
        return "frobenius"

    def estimate_error(self, predicted: np.ndarray, actual: np.ndarray, **kwargs) -> float:
        """Compute Frobenius norm of the difference matrix.

        Args:
            predicted: Predicted matrix
            actual: Actual matrix
            **kwargs: Unused

        Returns:
            Frobenius norm value
        """
        diff = predicted - actual
        return np.sqrt(np.sum(diff**2))


class RelativeErrorEstimator(AbstractErrorEstimator):
    """Relative error estimator normalized by ground truth magnitude."""

    @property
    def name(self) -> str:
        """Get the estimator name identifier."""
        return "relative"

    def estimate_error(self, predicted: np.ndarray, actual: np.ndarray, **kwargs) -> float:
        """Compute relative error normalized by actual values.

        Args:
            predicted: Predicted matrix
            actual: Actual matrix
            epsilon: Small value to avoid division by zero (default: 1e-8)

        Returns:
            Relative error value
        """
        epsilon = kwargs.get("epsilon", 1e-8)
        return np.mean(np.abs(predicted - actual) / (np.abs(actual) + epsilon))


class KullbackLeiblerEstimator(AbstractErrorEstimator):
    """Kullback-Leibler divergence for covariance matrices."""

    @property
    def name(self) -> str:
        """Get the estimator name identifier."""
        return "kl_divergence"

    def estimate_error(self, predicted: np.ndarray, actual: np.ndarray, **kwargs) -> float:
        """Compute KL divergence between two covariance matrices.

        Assumes predicted and actual are positive definite covariance matrices.

        Args:
            predicted: Predicted covariance matrix
            actual: Actual covariance matrix
            **kwargs: Unused

        Returns:
            KL divergence value
        """
        # Add small regularization to ensure positive definiteness
        epsilon = kwargs.get("epsilon", 1e-8)
        pred_reg = predicted + epsilon * np.eye(predicted.shape[0])
        actual_reg = actual + epsilon * np.eye(actual.shape[0])

        try:
            pred_inv = np.linalg.inv(pred_reg)
            actual_inv = np.linalg.inv(actual_reg)

            # KL divergence: 0.5 * (trace(actual_inv @ predicted) + log(det(actual)/det(predicted)) - n)
            trace_term = np.trace(actual_inv @ pred_reg)
            log_det_term = np.log(np.linalg.det(actual_reg) / np.linalg.det(pred_reg))
            n = predicted.shape[0]

            return 0.5 * (trace_term + log_det_term - n)
        except np.linalg.LinAlgError:
            # Fallback to Frobenius norm if matrices are singular
            return np.sum((predicted - actual) ** 2)


class ErrorEstimatorRegistry:
    """Registry for error estimators.

    Provides a centralized way to register and access different error
    estimation methods for MCOS simulations.
    """

    def __init__(self):
        """Initialize registry with default estimators."""
        self._estimators: Dict[str, AbstractErrorEstimator] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register the default error estimators."""
        self.register_estimator(MeanSquaredErrorEstimator())
        self.register_estimator(FrobeniusNormEstimator())
        self.register_estimator(RelativeErrorEstimator())
        self.register_estimator(KullbackLeiblerEstimator())

    def register_estimator(self, estimator: AbstractErrorEstimator):
        """Register a new error estimator.

        Args:
            estimator: Error estimator instance to register
        """
        self._estimators[estimator.name] = estimator

    def get_estimator(self, name: str) -> AbstractErrorEstimator:
        """Get an error estimator by name.

        Args:
            name: Name of the estimator

        Returns:
            Error estimator instance

        Raises:
            ValueError: If estimator is not registered
        """
        if name not in self._estimators:
            available = list(self._estimators.keys())
            raise ValueError(f"Unknown error estimator '{name}'. Available: {available}")
        return self._estimators[name]

    def list_estimators(self) -> List[str]:
        """List all registered error estimator names.

        Returns:
            List of estimator names
        """
        return list(self._estimators.keys())

    def estimate_errors(
        self, predicted: np.ndarray, actual: np.ndarray, estimator_names: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, float]:
        """Estimate errors using multiple estimators.

        Args:
            predicted: Predicted values
            actual: Actual values
            estimator_names: Names of estimators to use (all if None)
            **kwargs: Additional parameters passed to estimators

        Returns:
            Dictionary mapping estimator names to error values
        """
        if estimator_names is None:
            estimator_names = self.list_estimators()

        results = {}
        for name in estimator_names:
            estimator = self.get_estimator(name)
            results[name] = estimator.estimate_error(predicted, actual, **kwargs)

        return results


# Global registry instance
error_estimator_registry = ErrorEstimatorRegistry()
