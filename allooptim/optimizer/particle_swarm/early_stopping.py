import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class EarlyStopObjective:
    """Objective function wrapper with early stopping capability.

    Monitors optimization progress and stops when no significant improvement
    is observed for a specified number of iterations, preventing wasted computation.
    """

    def __init__(
        self,
        objective_function: Callable,
        tolerance: float = 1e-6,
        stagnation_limit: int = 50,
    ) -> None:
        """Initialize early stopping objective wrapper.

        Args:
            objective_function: The objective function to wrap
            tolerance: Minimum improvement threshold to reset stagnation counter
            stagnation_limit: Number of iterations without improvement before stopping
        """
        self.best_value = np.inf
        self.no_improve_count = 0
        self.tolerance = tolerance
        self.stagnation_limit = stagnation_limit
        self.converged = False
        self.objective_function = objective_function

    def __call__(self, x):
        """Evaluate objective function with early stopping logic."""
        if self.converged:
            # Return best known value once converged to stop further meaningful updates
            return self.best_value

        val = self.objective_function(x)

        # Handle array values by taking the mean
        val_scalar = np.mean(val) if isinstance(val, np.ndarray) else val

        improvement = self.best_value - val_scalar

        if improvement > self.tolerance:
            self.best_value = val_scalar
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.stagnation_limit:
            logger.debug(
                f"Early stopping triggered inside objective after no improvement: {self.no_improve_count} iterations."
            )
            self.converged = True

        return val
