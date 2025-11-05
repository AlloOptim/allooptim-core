import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from allooptim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG
from allooptim.optimizer.allocation_metric import LMoments
from allooptim.optimizer.asset_name_utils import create_weights_series, validate_asset_names
from allooptim.optimizer.optimizer_interface import AbstractOptimizer
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

logger = logging.getLogger(__name__)

# Constants for clustering and optimization thresholds
MIN_ASSETS_FOR_CLUSTERING = 4
MIN_FEASIBLE_CLUSTERS = 2
CLUSTER_IMPROVEMENT_THRESHOLD = 0.5
WEIGHT_DIFFERENCE_TOLERANCE = 0.1
SPEEDUP_FACTOR_THRESHOLD = 2.0
LARGE_ASSET_COUNT_THRESHOLD = 100
SMALL_ASSET_COUNT_THRESHOLD = 20


@dataclass
class ClusterResult:
    age: int
    score: float
    kmeans: Optional[KMeans]
    success: bool


class ObjectiveType(Enum):
    """Optimization objectives"""

    SHARPE = "sharpe"
    VARIANCE = "variance"


def compute_corr(cov: np.ndarray) -> np.ndarray:
    """
    normalize covariance matrix into a correlation matrix

    args:
        cov: covariance matrix of daily returns

    returns:
        correlation of returns
    """
    cov = pd.DataFrame(cov)
    std = np.sqrt(np.diag(cov))

    return cov / np.outer(std, std)


def convex_opt(cov: np.ndarray, mu: np.ndarray) -> None:
    """ """
    # pseudoinverse for numerically unstable matrices
    try:
        inv = np.linalg.inv(cov)

    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)

    ones = np.ones(shape=(len(inv), 1))

    w = np.dot(inv, mu)
    if np.sum(w) == 0:
        logger.error("Sum of weights is zero in convex optimization, falling back to equal weights")
        return ones.flatten() / len(ones)

    return w / np.dot(ones.T, w)


class NCOOptimizerConfig(BaseModel):
    model_config = DEFAULT_PYDANTIC_CONFIG

    long_only: bool = True
    enable_warm_start: bool = True
    cluster_age_limit: int = 10
    top_n: int = 10


class NCOSharpeOptimizer(AbstractOptimizer):
    """
    optimal portfolio allocation using Nested Clustered Optimization algorithm
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961)

    attributes:
        _returns: daily returns
        _cov: covariance matrix of daily returns
        _num_assets: number of assets in investment universe
        _W_init: initial portfolio allocation for optimization task, takes
            values of equal weight strategy
        _long_only: True if long positions only, false otherwise
        _corr: correlation matrix of daily returns
    """

    objective_type = ObjectiveType.SHARPE

    def __init__(self, config: Optional[NCOOptimizerConfig] = None) -> None:
        self.config = config or NCOOptimizerConfig()

        self._previous_weights = None
        self._cluster_results: dict[int, ClusterResult] = {}
        self._previous_best_score: float = -np.inf
        self._previous_top_n_indices: list[int] = []

    def allocate(
        self,
        ds_mu: pd.Series,
        df_cov: pd.DataFrame,
        df_prices: Optional[pd.DataFrame] = None,
        time: Optional[datetime] = None,
        l_moments: Optional[LMoments] = None,
    ) -> pd.Series:
        # Validate asset names consistency
        validate_asset_names(ds_mu, df_cov)
        asset_names = ds_mu.index.tolist()

        # Ensure inputs are numpy arrays, not DataFrames
        mu_array = np.asarray(ds_mu.values).flatten()
        cov_array = np.asarray(df_cov.values)

        weights = self._optimal_weights(mu_array, cov_array)

        # Ensure weights are valid and sum to 1
        weights = np.asarray(weights).flatten()
        weights = weights - np.min(weights)

        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            logger.error("NCO resulted in zero weights, falling back to equal weight strategy")
            weights = np.ones(len(asset_names)) / len(asset_names)

        return create_weights_series(weights, asset_names)

    def _optimal_weights(
        self,
        returns: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """
        computes the optimal weights using the NCO algorithm

        args:
            objective: "sharpe" for maximum sharpe ratio, and "variance"
                for minimum variance

        returns:
            optimal weight allocation
        """
        self._initialize(returns, cov)

        self._cluster_assets()

        # optimization parameter
        intra_weights = np.zeros((self._n_assets, self._n_clusters))

        curr_mu = np.ones(shape=(self._n_assets, 1))

        # within-cluster weights
        for idx, cluster in self._clusters.items():
            try:
                curr_cov = self._cov[cluster][:, cluster]

                if self.objective_type == ObjectiveType.SHARPE:
                    curr_mu = self._returns[cluster].reshape(-1, 1)

                cluster_weights = convex_opt(curr_cov, curr_mu).flatten()
                intra_weights[cluster, idx] = cluster_weights

            except (np.linalg.LinAlgError, ValueError) as e:
                logger.error(f"Clustering failed for cluster {idx}, using equal weights: {e}")
                cluster_size = len(cluster)
                intra_weights[cluster, idx] = np.ones(cluster_size) / cluster_size

        # cluster weights
        try:
            cluster_cov = intra_weights.T @ self._cov @ intra_weights

            if self.objective_type == ObjectiveType.SHARPE:
                cluster_mu = intra_weights.T @ self._returns
            else:
                cluster_mu = np.ones(shape=(self._n_clusters, 1))

            inter_weights = convex_opt(cluster_cov, cluster_mu).flatten()

            new_weights = np.multiply(intra_weights, inter_weights).sum(axis=1)

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.error(f"Inter-cluster optimization failed, using equal weights: {e}")
            # Fall back to equal weights if inter-cluster optimization fails
            new_weights = np.ones(self._n_assets) / self._n_assets

        # Ensure weights sum to 1 and are non-negative if long_only
        if self.config.long_only:
            new_weights = np.maximum(new_weights, 0)

        weight_sum = np.sum(new_weights)
        if weight_sum > 0:
            new_weights = new_weights / weight_sum
        else:
            logger.error("NCO resulted in zero weights, falling back to equal weight strategy")
            new_weights = np.ones(self._n_assets) / self._n_assets

        self._previous_weights = new_weights

        return new_weights

    def _initialize(
        self,
        returns: np.ndarray,
        cov: np.ndarray,
    ) -> None:
        self._returns = np.asarray(returns).flatten()
        self._cov = np.asarray(cov)
        self._n_assets = len(self._returns)

        if self._previous_weights is None:
            self._previous_weights = np.ones(self._n_assets) / self._n_assets
        else:
            assert len(self._previous_weights) == self._n_assets, "Initial weights length mismatch"

        self._corr = compute_corr(cov)

    def _cluster_assets(self) -> None:
        """
        groups assets into clusters using k means, using silhoueete scores
        to find the optimal number of clusters

        args:
            max_num_clusters: maximum number of clusters allowed
        """
        # distance matrix for silhouette scores
        dist = (1 - self._corr / 2) ** 0.5
        dist = np.nan_to_num(dist, nan=0.0)

        # Ensure we have enough assets for clustering
        if self._n_assets < MIN_ASSETS_FOR_CLUSTERING:  # Need at least 4 assets for meaningful clustering
            # Fall back to single cluster (no clustering)
            self._clusters = {0: list(range(self._n_assets))}
            self._n_clusters = 1
            logger.error("Not enough assets for clustering, using single cluster")
            return

        max_feasible_clusters = self._n_assets // 2

        if max_feasible_clusters < MIN_FEASIBLE_CLUSTERS:
            # Not enough assets for clustering
            self._clusters = {0: list(range(self._n_assets))}
            self._n_clusters = 1
            logger.error("Not enough assets for clustering, using single cluster")
            return

        self._update_clusters(max_feasible_clusters, dist)

        best_kmeans = self._find_best_cluster()

        self._apply_best_cluster(best_kmeans)

    def _update_clusters(self, max_feasible_clusters: int, dist: np.ndarray) -> None:
        if not self.config.enable_warm_start:
            self._cluster_results = {}

        not_updated_count = 0
        average_score_deteriation = 0.0

        for index in range(2, max_feasible_clusters + 1):
            if index not in self._cluster_results:
                # not trained before
                logger.debug(f"Training new cluster for {index} clusters")
                self._cluster_results[index] = self._update_cluster(index, dist)
                continue

            if index in self._previous_top_n_indices:
                # top n clusters are always updated, as they have a high chance of being the best
                logger.debug(f"Updating top cluster {index}")
                self._cluster_results[index] = self._update_cluster(index, dist)
                continue

            if self._cluster_results[index].age > self.config.cluster_age_limit:
                # cluster too old, always update
                logger.debug(f"Cluster {index} too old, updating")
                self._cluster_results[index] = self._update_cluster(index, dist)
                continue

            if not self._cluster_results[index].success:
                # cluster failed before, always update
                logger.debug(f"Cluster {index} failed before, updating")
                self._cluster_results[index] = self._update_cluster(index, dist)
                continue

            previous_score = self._cluster_results[index].score

            labels = self._cluster_results[index].kmeans.predict(dist)
            new_score = self._estimate_score(labels, dist)

            # train candidates which have a deteriation, but still a significant chance of being the best cluster
            if self._previous_best_score > 0:
                relative_new_score = new_score / self._previous_best_score
                relative_previous_score = previous_score / self._previous_best_score
                new_score_deteriated = relative_new_score < 0.95 * relative_previous_score
                has_best_cluster_chance = relative_new_score > CLUSTER_IMPROVEMENT_THRESHOLD

                if new_score_deteriated and has_best_cluster_chance:
                    logger.debug(f"Cluster {index} score deteriorated significantly, updating")
                    self._cluster_results[index] = self._update_cluster(index, dist)
                    continue

            # update score
            self._cluster_results[index].score = new_score
            self._cluster_results[index].age += 1

            if new_score < previous_score:
                average_score_deteriation += abs(new_score - previous_score)
            not_updated_count += 1

        logger.debug(f"Clusters updated: {max_feasible_clusters - 1 - not_updated_count}/{max_feasible_clusters - 1}")
        logger.debug(
            f"Average score deterioration for non-updated clusters: "
            f"{average_score_deteriation / max(not_updated_count, 1):.6f}"
        )

    def _find_best_cluster(self) -> Optional[KMeans]:
        score_array = np.array([cluster_result.score for cluster_result in self._cluster_results.values()])
        top_indices = list(np.argsort(score_array)[::-1])

        # Get cluster results sorted by score (best first)
        cluster_results_list = list(self._cluster_results.values())
        top_clusters = [cluster_results_list[i] for i in top_indices]

        at_least_one_success = any(cluster_result.success for cluster_result in top_clusters)
        if not at_least_one_success:
            logger.error("All clustering attempts failed, falling back to single cluster")
            return None

        best_cluster_result = None
        for cluster_result in top_clusters:
            if cluster_result.success:
                best_cluster_result = cluster_result
                best_kmeans = cluster_result.kmeans
                break

        # Update previous values using the best cluster result
        if best_cluster_result is not None:
            self._previous_best_score = best_cluster_result.score
            self._previous_top_n_indices = top_indices[: self.config.top_n]

        return best_kmeans

    def _apply_best_cluster(self, best_kmeans: Optional[KMeans]) -> None:
        # Use best clustering result, or fall back to single cluster
        if best_kmeans is not None:
            # assign clusters using best cluster sizes
            # Use numeric indices if _corr is a numpy array, otherwise use columns
            asset_names = self._corr.columns if hasattr(self._corr, "columns") else list(range(self._corr.shape[1]))

            self._clusters = {
                i: [asset_names[j] for j in np.where(best_kmeans.labels_ == i)[0]]
                for i in np.unique(best_kmeans.labels_)
            }
            self._n_clusters = len(self._clusters.keys())
        else:
            logger.error("All clustering attempts failed, falling back to single cluster")
            self._clusters = {0: list(range(self._n_assets))}
            self._n_clusters = 1

    def _estimate_score(self, labels: np.ndarray, dist: np.ndarray) -> float:
        scores = silhouette_samples(dist, labels)
        normalizes_score = scores.mean() / max(scores.std(), 1e-10)
        return normalizes_score

    def _update_cluster(self, n_clusters: int, dist: np.ndarray) -> ClusterResult:
        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(dist)
            score = self._estimate_score(kmeans.labels_, dist)
            cluster_result = ClusterResult(age=0, score=score, kmeans=kmeans, success=True)

        except (ValueError, np.linalg.LinAlgError):
            logger.error(f"Clustering failed for {n_clusters}")
            cluster_result = ClusterResult(age=0, score=-np.inf, kmeans=None, success=False)

        return cluster_result

    @property
    def name(self) -> str:
        return "NCOSharpeOptimizer"


class NCOVarianceOptimizer(NCOSharpeOptimizer):
    objective_type = ObjectiveType.VARIANCE

    @property
    def name(self) -> str:
        return "NCOVarianceOptimizer"
