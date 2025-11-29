"""Clustering analysis for portfolio optimizer comparison.

This module provides sophisticated clustering algorithms to analyze and group
portfolio optimizers based on their performance characteristics and portfolio
weight similarities. It helps identify patterns in optimizer behavior and
performance across different market conditions.

Key features:
- Hierarchical clustering of optimizer performance
- K-means clustering for portfolio weight similarity
- PCA-based dimensionality reduction
- Distance matrix calculations
- Cluster stability analysis
- Automated cluster number selection
- Visualization support for clustering results
"""

import logging

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Constants for clustering analysis
MIN_DATA_POINTS_FOR_CLUSTERING = 2
PCA_COMPONENT_THRESHOLD = 6
MAX_CLUSTERS_DEFAULT = 5
DEFAULT_N_CLUSTERS = 4
FALLBACK_N_CLUSTERS = 2


class ClusterAnalyzer:
    """Analyze clustering of optimizers based on performance and portfolio similarity."""

    def __init__(self, results: dict):
        """Initialize the cluster analyzer.

        Args:
            results: Dictionary containing backtest results for all optimizers.
        """
        self.results = results

    def analyze_clusters(self) -> dict:
        """Perform comprehensive clustering analysis."""
        logger.info("Starting clustering analysis")

        clustering_results = {}

        # 1. Performance-based clustering
        clustering_results["performance"] = self._cluster_by_performance()

        # 2. Portfolio weights correlation clustering
        clustering_results["portfolio_correlation"] = self._cluster_by_portfolio_correlation()

        # 3. Returns correlation clustering
        clustering_results["returns_correlation"] = self._cluster_by_returns_correlation()

        # 4. Combined clustering
        clustering_results["combined"] = self._combined_clustering()

        # 5. Euclidean distance analysis
        clustering_results["euclidean_distance"] = self._analyze_euclidean_distances()

        return clustering_results

    def _cluster_by_performance(self) -> dict:
        """Cluster optimizers by performance metrics."""
        # Extract performance metrics
        metrics_data = []
        optimizer_names = []

        for name, data in self.results.items():
            metrics = data["metrics"]

            # Select key performance metrics for clustering
            performance_vector = [
                metrics.get("sharpe_ratio", 0),
                metrics.get("max_drawdown", 0),
                metrics.get("cagr", 0),
                metrics.get("annual_volatility", 0),
                metrics.get("risk_adjusted_return", 0),
            ]

            metrics_data.append(performance_vector)
            optimizer_names.append(name)

        if len(metrics_data) < MIN_DATA_POINTS_FOR_CLUSTERING:
            return {"message": "Insufficient data for performance clustering"}

        metrics_array = np.array(metrics_data)

        # Standardize metrics
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics_array)

        # Perform hierarchical clustering
        linkage_matrix = linkage(metrics_scaled, method="ward")

        # Get clusters (use 3-5 clusters)
        n_clusters = min(4, len(optimizer_names) // 2)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

        # Group optimizers by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(optimizer_names[i])

        return {
            "method": "hierarchical_performance",
            "n_clusters": n_clusters,
            "clusters": clusters,
            "linkage_matrix": linkage_matrix,
            "optimizer_names": optimizer_names,
        }

    def _cluster_by_portfolio_correlation(self) -> dict:
        """Cluster optimizers by portfolio weight correlation."""
        # Extract portfolio weights time series
        weights_series = {}

        for name, data in self.results.items():
            if "weights_history" in data and not data["weights_history"].empty:
                # Flatten weights across time and assets
                weights_df = data["weights_history"]
                weights_flat = weights_df.values.flatten()
                weights_series[name] = weights_flat

        if len(weights_series) < MIN_DATA_POINTS_FOR_CLUSTERING:
            return {"message": "Insufficient data for portfolio correlation clustering"}

        # Create correlation matrix
        optimizer_names = list(weights_series.keys())

        # Check that all weight series have the same length
        lengths = [len(weights_series[name]) for name in optimizer_names]
        if len(set(lengths)) > 1:
            return {"message": "Inconsistent weight series lengths for portfolio correlation clustering"}

        correlation_matrix = np.corrcoef([weights_series[name] for name in optimizer_names])

        # Handle NaN values and ensure symmetry
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Force symmetry
        np.fill_diagonal(correlation_matrix, 1.0)  # Ensure diagonal is 1

        # Convert to distance matrix
        distance_matrix = 1 - np.abs(correlation_matrix)

        # Ensure distance matrix is symmetric and valid
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0.0)  # Distance to self should be 0

        # Convert distance matrix to condensed form for linkage
        # squareform converts between square and condensed distance matrices
        condensed_distance = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distance, method="average")

        # Get clusters
        n_clusters = min(4, len(optimizer_names) // 2)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(optimizer_names[i])

        return {
            "method": "portfolio_correlation",
            "n_clusters": n_clusters,
            "clusters": clusters,
            "correlation_matrix": correlation_matrix,
            "optimizer_names": optimizer_names,
        }

    def _cluster_by_returns_correlation(self) -> dict:
        """Cluster optimizers by returns correlation."""
        # Extract returns time series
        returns_series = {}

        for name, data in self.results.items():
            if "returns" in data and not data["returns"].empty:
                returns_series[name] = data["returns"].values

        if len(returns_series) < MIN_DATA_POINTS_FOR_CLUSTERING:
            return {"message": "Insufficient data for returns correlation clustering"}

        # Align series to same length (use shortest)
        min_length = min(len(series) for series in returns_series.values())
        aligned_series = {name: series[:min_length] for name, series in returns_series.items()}

        # Create correlation matrix
        optimizer_names = list(aligned_series.keys())
        correlation_matrix = np.corrcoef([aligned_series[name] for name in optimizer_names])

        # Handle NaN values in correlation matrix (can occur with constant or invalid returns)
        if np.any(np.isnan(correlation_matrix)):
            logger.warning("NaN values found in correlation matrix, replacing with zeros")
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        # K-means clustering on correlation features
        n_clusters = min(4, len(optimizer_names) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(correlation_matrix)

        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(optimizer_names[i])

        return {
            "method": "returns_correlation_kmeans",
            "n_clusters": n_clusters,
            "clusters": clusters,
            "correlation_matrix": correlation_matrix,
            "optimizer_names": optimizer_names,
        }

    def _combined_clustering(self) -> dict:
        """Combined clustering using multiple features."""
        # Combine performance metrics, portfolio similarity, and returns correlation
        combined_features = []
        optimizer_names = []

        for name, data in self.results.items():
            metrics = data["metrics"]

            # Performance features
            perf_features = [
                metrics.get("sharpe_ratio", 0),
                metrics.get("max_drawdown", 0),
                metrics.get("cagr", 0),
                metrics.get("annual_volatility", 0),
            ]

            # Portfolio characteristics
            portfolio_features = [metrics.get("turnover_mean", 0), metrics.get("avg_computation_time", 0)]

            # Returns characteristics
            returns_features = [metrics.get("returns_skew", 0), metrics.get("returns_kurtosis", 0)]

            combined_vector = perf_features + portfolio_features + returns_features
            combined_features.append(combined_vector)
            optimizer_names.append(name)

        if len(combined_features) < MIN_DATA_POINTS_FOR_CLUSTERING:
            return {"message": "Insufficient data for combined clustering"}

        # Standardize features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(combined_features)

        # Use PCA for dimensionality reduction if many features
        if features_scaled.shape[1] > PCA_COMPONENT_THRESHOLD:
            # Use min of (desired components, available features, samples) - 1 for safety
            n_components = min(6, features_scaled.shape[1] - 1, features_scaled.shape[0] - 1)
            if n_components >= 1:
                pca = PCA(n_components=n_components)
                features_scaled = pca.fit_transform(features_scaled)

        # K-means clustering
        n_clusters = min(5, len(optimizer_names) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)

        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(optimizer_names[i])

        return {
            "method": "combined_kmeans",
            "n_clusters": n_clusters,
            "clusters": clusters,
            "optimizer_names": optimizer_names,
            "feature_importance": "performance + portfolio + returns characteristics",
        }

    def _analyze_euclidean_distances(self) -> dict:
        """Analyze mean Euclidean distances between optimizer weights across all timesteps."""
        logger.info("Computing Euclidean distances between optimizer weights")

        # Extract weights history for all optimizers
        weights_data = {}

        for name, data in self.results.items():
            if "weights_history" in data and not data["weights_history"].empty:
                weights_data[name] = data["weights_history"]

        if len(weights_data) < MIN_DATA_POINTS_FOR_CLUSTERING:
            return {"message": "Insufficient data for Euclidean distance analysis"}

        optimizer_names = list(weights_data.keys())
        n_optimizers = len(optimizer_names)

        # Calculate pairwise Euclidean distances
        distance_matrix = np.zeros((n_optimizers, n_optimizers))
        pairwise_distances = {}

        for i, opt_a in enumerate(optimizer_names):
            for j, opt_b in enumerate(optimizer_names):
                if i <= j:  # Only compute upper triangle (matrix is symmetric)
                    if i == j:
                        distance_matrix[i, j] = 0.0
                        pairwise_distances[f"{opt_a}_{opt_b}"] = 0.0
                    else:
                        # Align weights to same timestamps and assets
                        weights_a = weights_data[opt_a]
                        weights_b = weights_data[opt_b]

                        # Find common timestamps and assets
                        common_times = weights_a.index.intersection(weights_b.index)
                        common_assets = weights_a.columns.intersection(weights_b.columns)

                        if len(common_times) > 0 and len(common_assets) > 0:
                            # Extract aligned data
                            aligned_a = weights_a.loc[common_times, common_assets]
                            aligned_b = weights_b.loc[common_times, common_assets]

                            # Calculate Euclidean distance for each timestep
                            timestep_distances = []
                            for t in common_times:
                                w_a = aligned_a.loc[t].values
                                w_b = aligned_b.loc[t].values
                                dist = np.linalg.norm(w_a - w_b)
                                timestep_distances.append(dist)

                            # Mean distance across all timesteps
                            mean_distance = np.mean(timestep_distances)
                            distance_matrix[i, j] = mean_distance
                            distance_matrix[j, i] = mean_distance  # Symmetric
                            pairwise_distances[f"{opt_a}_{opt_b}"] = mean_distance
                        else:
                            # No common data - set to maximum distance
                            distance_matrix[i, j] = np.inf
                            distance_matrix[j, i] = np.inf
                            pairwise_distances[f"{opt_a}_{opt_b}"] = np.inf

        # Find closest pairs (excluding self-comparisons)
        closest_pairs = []
        for i, opt_a in enumerate(optimizer_names):
            for j, opt_b in enumerate(optimizer_names):
                if i < j and not np.isinf(distance_matrix[i, j]):
                    closest_pairs.append(
                        {"optimizer_a": opt_a, "optimizer_b": opt_b, "mean_euclidean_distance": distance_matrix[i, j]}
                    )

        # Sort by distance
        closest_pairs.sort(key=lambda x: x["mean_euclidean_distance"])

        # Group optimizers by similarity (using hierarchical clustering on distance matrix)
        try:
            # Handle infinite values for clustering
            finite_distances = distance_matrix.copy()
            finite_distances[np.isinf(finite_distances)] = (
                np.nanmax(finite_distances[np.isfinite(finite_distances)]) * 2
            )

            # Convert to condensed distance matrix for linkage
            condensed_distances = squareform(finite_distances)
            linkage_matrix = linkage(condensed_distances, method="average")

            # Get clusters
            n_clusters = (
                min(MAX_CLUSTERS_DEFAULT, len(optimizer_names) // 2)
                if len(optimizer_names) > MIN_DATA_POINTS_FOR_CLUSTERING
                else FALLBACK_N_CLUSTERS
            )
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(optimizer_names[i])

            clustering_info = {"n_clusters": n_clusters, "clusters": clusters, "linkage_matrix": linkage_matrix}
        except Exception as e:
            logger.warning(f"Could not perform hierarchical clustering on distances: {e}")
            clustering_info = {"message": "Clustering failed due to insufficient valid distances"}

        return {
            "method": "mean_euclidean_distance",
            "distance_matrix": distance_matrix,
            "optimizer_names": optimizer_names,
            "pairwise_distances": pairwise_distances,
            "closest_pairs": closest_pairs[:10],  # Top 10 closest pairs
            "clustering": clustering_info,
        }
