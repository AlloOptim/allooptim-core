"""
Diverse Covariance Matrix Training Data Generation Module
Generates 50,000 diverse synthetic correlation matrices using multiple methods
for enhanced autoencoder training
"""

import multiprocessing as mp
import random
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, sqrtm
from scipy.stats import ortho_group, wishart

# Import existing generator
from training_data import CovarianceMatrixGenerator, SpectrumGenerator


@dataclass
class DiverseTrainingConfig:
    """Configuration for diverse training data generation"""

    n_samples: int = 50000
    n_assets: int = 500

    # Distribution percentages (should sum to 1.0)
    classical_pct: float = 0.4  # 40% classical synthetic
    gan_style_pct: float = 0.3  # 30% GAN-style
    bootstrap_pct: float = 0.2  # 20% bootstrap from real patterns
    block_pct: float = 0.1  # 10% block-structured

    # Noise parameters
    noise_level: float = 0.05  # Amount of sampling noise to add
    min_eigenval: float = 1e-6  # Minimum eigenvalue for numerical stability

    random_seed: int = 42


class DiverseCorrelationGenerator:
    """Enhanced correlation matrix generator with multiple diverse methods"""

    def __init__(self, config: DiverseTrainingConfig = None):
        self.config = config or DiverseTrainingConfig()
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

    def random_spectrum_generation(self) -> np.ndarray:
        """
        Generate diverse eigenvalue spectra using various distributions

        Returns:
            Array of eigenvalues (normalized to sum to n_assets)
        """
        n = self.config.n_assets
        spectrum_type = random.choice(
            ["exponential_decay", "power_law", "factor_model", "uniform_decay", "steep_decay", "flat_spectrum"]
        )

        if spectrum_type == "exponential_decay":
            eigenvals = SpectrumGenerator.exponential_decay(n, decay_rate=np.random.uniform(0.5, 4.0))
        elif spectrum_type == "power_law":
            eigenvals = SpectrumGenerator.power_law(n, exponent=np.random.uniform(0.5, 3.0))
        elif spectrum_type == "factor_model":
            eigenvals = SpectrumGenerator.factor_model(n, market_variance_explained=np.random.uniform(0.15, 0.7))
        elif spectrum_type == "uniform_decay":
            # Linear decay
            eigenvals = np.linspace(1, 0.1, n)
            eigenvals = eigenvals / np.sum(eigenvals) * n
        elif spectrum_type == "steep_decay":
            # Very steep exponential decay (few dominant factors)
            eigenvals = np.exp(-np.linspace(0, 8, n))
            eigenvals = eigenvals / np.sum(eigenvals) * n
        else:  # flat_spectrum
            # Nearly uniform eigenvalues (low correlation regime)
            eigenvals = np.ones(n) + np.random.uniform(-0.2, 0.2, n)
            eigenvals = np.maximum(eigenvals, 0.01)
            eigenvals = eigenvals / np.sum(eigenvals) * n

        return eigenvals

    def random_correlation_matrix(self, N: int, spectrum: np.ndarray = None) -> np.ndarray:
        """
        Generate correlation matrix from spectrum using random orthogonal basis

        Args:
            N: Matrix dimension
            spectrum: Eigenvalues (if None, generate random spectrum)

        Returns:
            Correlation matrix
        """
        if spectrum is None:
            spectrum = self.random_spectrum_generation()

        # Ensure positive eigenvalues and proper normalization
        spectrum = np.maximum(spectrum, self.config.min_eigenval)
        spectrum = spectrum / np.sum(spectrum) * N

        return CovarianceMatrixGenerator.from_eigenvalues(spectrum)

    def corrgan_style_generation(self, N: int) -> np.ndarray:
        """
        Simulate GAN-style generation with more complex patterns
        Mimics what a CorrGAN might produce with diverse structures

        Args:
            N: Matrix dimension

        Returns:
            Correlation matrix
        """
        # Choose generation style
        style = random.choice(["hierarchical", "random_blocks", "smooth_interpolation", "noisy_factor"])

        if style == "hierarchical":
            # Generate hierarchical correlation structure
            return self._generate_hierarchical_structure(N)
        elif style == "random_blocks":
            # Random block structure with varying sizes
            return self._generate_random_blocks(N)
        elif style == "smooth_interpolation":
            # Smooth interpolation between different correlation patterns
            return self._generate_smooth_interpolation(N)
        else:  # noisy_factor
            # Factor model with additional noise and complexity
            return self._generate_noisy_factor_model(N)

    def _generate_hierarchical_structure(self, N: int) -> np.ndarray:
        """Generate hierarchical correlation structure"""
        # Create base correlation with distance-based decay
        positions = np.random.uniform(0, 1, N)  # Random positions on [0,1]
        distances = np.abs(positions[:, None] - positions[None, :])

        # Exponential decay with random parameters
        decay_rate = np.random.uniform(2, 10)
        base_corr = np.exp(-decay_rate * distances)

        # Add hierarchical clustering
        n_clusters = random.randint(5, 15)
        cluster_assignments = np.random.randint(0, n_clusters, N)

        for i in range(N):
            for j in range(i + 1, N):
                if cluster_assignments[i] == cluster_assignments[j]:
                    # Within cluster: higher correlation
                    base_corr[i, j] *= np.random.uniform(1.2, 2.0)
                else:
                    # Between clusters: lower correlation
                    base_corr[i, j] *= np.random.uniform(0.3, 0.8)

        # Ensure symmetry and proper diagonal
        base_corr = (base_corr + base_corr.T) / 2
        np.fill_diagonal(base_corr, 1.0)

        # Make positive definite
        eigenvals, eigenvecs = np.linalg.eigh(base_corr)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _generate_random_blocks(self, N: int) -> np.ndarray:
        """Generate random block structure"""
        n_blocks = random.randint(3, 12)
        block_sizes = np.random.multinomial(N, np.ones(n_blocks) / n_blocks)
        block_sizes = block_sizes[block_sizes > 0]  # Remove zero-size blocks

        blocks = []
        for size in block_sizes:
            if size == 1:
                blocks.append(np.array([[1.0]]))
            else:
                # Generate random correlation within block
                eigenvals = np.random.uniform(0.1, 2.0, size)
                eigenvals = eigenvals / np.sum(eigenvals) * size
                block_corr = CovarianceMatrixGenerator.from_eigenvalues(eigenvals)
                blocks.append(block_corr)

        # Create block diagonal matrix
        block_matrix = block_diag(*blocks)

        # Add small off-block correlations
        off_block_corr = np.random.uniform(-0.2, 0.2, (N, N))
        off_block_corr = (off_block_corr + off_block_corr.T) / 2

        # Combine with decay for distant blocks
        final_matrix = block_matrix + 0.1 * off_block_corr

        # Ensure proper correlation matrix
        np.fill_diagonal(final_matrix, 1.0)
        eigenvals, eigenvecs = np.linalg.eigh(final_matrix)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _generate_smooth_interpolation(self, N: int) -> np.ndarray:
        """Generate smooth interpolation between different patterns"""
        # Create two different correlation patterns
        eigenvals1 = SpectrumGenerator.exponential_decay(N, decay_rate=1.0)
        eigenvals2 = SpectrumGenerator.power_law(N, exponent=2.0)

        matrix1 = CovarianceMatrixGenerator.from_eigenvalues(eigenvals1)
        matrix2 = CovarianceMatrixGenerator.from_eigenvalues(eigenvals2)

        # Random interpolation weight
        alpha = np.random.beta(2, 2)  # Beta distribution for smooth mixing

        interpolated = alpha * matrix1 + (1 - alpha) * matrix2

        # Ensure proper correlation matrix
        np.fill_diagonal(interpolated, 1.0)
        eigenvals, eigenvecs = np.linalg.eigh(interpolated)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _generate_noisy_factor_model(self, N: int) -> np.ndarray:
        """Generate noisy factor model with additional complexity"""
        n_factors = random.randint(2, min(10, N // 5))

        # Random factor loadings
        loadings = np.random.normal(0, 1, (N, n_factors))

        # Different factor variances
        factor_vars = np.random.uniform(0.5, 3.0, n_factors)

        # Factor correlation matrix (factors can be correlated)
        if n_factors > 1:
            factor_corr_eigenvals = np.random.uniform(0.1, 1.5, n_factors)
            factor_corr_eigenvals = factor_corr_eigenvals / np.sum(factor_corr_eigenvals) * n_factors
            factor_corr = CovarianceMatrixGenerator.from_eigenvalues(factor_corr_eigenvals)
        else:
            factor_corr = np.array([[1.0]])

        # Construct covariance matrix
        cov_matrix = loadings @ (factor_corr * factor_vars) @ loadings.T

        # Add idiosyncratic variances
        idiosyncratic_vars = np.random.uniform(0.3, 1.0, N)
        np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + idiosyncratic_vars)

        # Convert to correlation matrix
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(cov_matrix)))
        corr_matrix = diag_inv_sqrt @ cov_matrix @ diag_inv_sqrt

        # Ensure numerical stability
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def bootstrap_sample(self, N: int) -> np.ndarray:
        """
        Generate correlation matrix based on realistic market patterns
        Simulates bootstrap sampling from real financial data

        Args:
            N: Matrix dimension

        Returns:
            Correlation matrix
        """
        # Simulate different market regimes
        regime = random.choice(["normal", "crisis", "low_vol", "sector_rotation"])

        if regime == "normal":
            # Normal market: moderate correlations, factor structure
            market_beta = np.random.uniform(0.3, 0.8, N)
            sector_effects = self._generate_sector_effects(N)
            base_corr = np.outer(market_beta, market_beta) + sector_effects

        elif regime == "crisis":
            # Crisis: high correlations, few factors dominate
            crisis_factor = np.random.uniform(0.6, 0.9, N)
            base_corr = np.outer(crisis_factor, crisis_factor)
            # Add some negative correlations (safe havens)
            n_safe_havens = random.randint(1, max(1, N // 20))
            safe_haven_indices = np.random.choice(N, n_safe_havens, replace=False)
            for idx in safe_haven_indices:
                base_corr[idx, :] *= -0.3
                base_corr[:, idx] *= -0.3

        elif regime == "low_vol":
            # Low volatility: lower correlations, more idiosyncratic
            base_corr = np.random.uniform(0.05, 0.25, (N, N))
            base_corr = (base_corr + base_corr.T) / 2

        else:  # sector_rotation
            # Sector rotation: strong intra-sector, weak inter-sector correlations
            base_corr = self._generate_sector_rotation_pattern(N)

        # Add sampling noise and ensure proper correlation matrix
        base_corr = self.add_sampling_noise(base_corr)
        np.fill_diagonal(base_corr, 1.0)

        # Make positive definite
        eigenvals, eigenvecs = np.linalg.eigh(base_corr)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _generate_sector_effects(self, N: int) -> np.ndarray:
        """Generate sector-based correlation effects"""
        n_sectors = random.randint(5, 15)
        sector_assignments = np.random.randint(0, n_sectors, N)

        sector_corr = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if sector_assignments[i] == sector_assignments[j]:
                    sector_corr[i, j] = np.random.uniform(0.2, 0.6)
                else:
                    sector_corr[i, j] = np.random.uniform(-0.1, 0.2)

        return sector_corr + sector_corr.T

    def _generate_sector_rotation_pattern(self, N: int) -> np.ndarray:
        """Generate sector rotation correlation pattern"""
        n_sectors = random.randint(6, 12)
        sector_size = N // n_sectors

        corr_matrix = np.zeros((N, N))

        # Strong intra-sector correlations
        for sector in range(n_sectors):
            start_idx = sector * sector_size
            end_idx = min((sector + 1) * sector_size, N)

            sector_corr = np.random.uniform(0.4, 0.8)
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        corr_matrix[i, j] = sector_corr * np.random.uniform(0.8, 1.2)

        # Weak inter-sector correlations
        for i in range(N):
            for j in range(i + 1, N):
                if corr_matrix[i, j] == 0:  # Not in same sector
                    corr_matrix[i, j] = np.random.uniform(-0.2, 0.2)

        return corr_matrix + corr_matrix.T

    def generate_block_structure(self, N: int, n_blocks: int = None) -> np.ndarray:
        """
        Generate block-structured correlation matrix (sectors)

        Args:
            N: Matrix dimension
            n_blocks: Number of blocks (if None, randomly chosen)

        Returns:
            Block-structured correlation matrix
        """
        if n_blocks is None:
            n_blocks = random.randint(5, 15)

        # Generate block sizes
        block_sizes = np.random.multinomial(N, np.ones(n_blocks) / n_blocks)
        block_sizes = block_sizes[block_sizes > 0]

        # Adjust if sum doesn't match N due to multinomial sampling
        diff = N - np.sum(block_sizes)
        if diff > 0:
            block_sizes[-1] += diff
        elif diff < 0:
            block_sizes[-1] = max(1, block_sizes[-1] + diff)

        blocks = []
        for size in block_sizes:
            if size == 1:
                blocks.append(np.array([[1.0]]))
            else:
                # High intra-block correlation with some variation
                base_corr = np.random.uniform(0.4, 0.8)
                block_matrix = np.full((size, size), base_corr)
                np.fill_diagonal(block_matrix, 1.0)

                # Add some noise to make it more realistic
                noise = np.random.uniform(-0.1, 0.1, (size, size))
                noise = (noise + noise.T) / 2
                np.fill_diagonal(noise, 0.0)

                block_matrix += noise
                np.fill_diagonal(block_matrix, 1.0)

                # Ensure positive definite
                eigenvals, eigenvecs = np.linalg.eigh(block_matrix)
                eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
                eigenvals = eigenvals / np.sum(eigenvals) * size

                block_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                blocks.append(block_matrix)

        # Create block diagonal matrix
        block_matrix = block_diag(*blocks)

        # Add weak inter-block correlations
        inter_block_strength = np.random.uniform(0.05, 0.15)
        noise = np.random.uniform(-inter_block_strength, inter_block_strength, (N, N))
        noise = (noise + noise.T) / 2

        # Only add noise to off-block elements
        current_idx = 0
        for size in block_sizes:
            block_slice = slice(current_idx, current_idx + size)
            noise[block_slice, block_slice] = 0.0
            current_idx += size

        final_matrix = block_matrix + noise
        np.fill_diagonal(final_matrix, 1.0)

        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(final_matrix)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def add_sampling_noise(self, C: np.ndarray) -> np.ndarray:
        """
        Add finite sample estimation noise to correlation matrix
        Simulates the noise that comes from estimating correlations from limited data

        Args:
            C: Input correlation matrix

        Returns:
            Correlation matrix with added sampling noise
        """
        N = C.shape[0]

        # Simulate finite sample effects
        sample_size = random.randint(50, 500)  # Typical sample sizes in finance

        # Generate noise proportional to estimation error
        # Standard error of correlation ~ 1/sqrt(T-3) for T observations
        noise_std = self.config.noise_level / np.sqrt(max(sample_size - 3, 1))

        # Add correlated noise (estimation errors are not independent)
        noise = np.random.normal(0, noise_std, (N, N))
        noise = (noise + noise.T) / 2  # Ensure symmetry
        np.fill_diagonal(noise, 0.0)  # Keep diagonal = 1

        # Apply noise with varying intensity based on correlation strength
        # Stronger correlations have more stable estimates
        noise_weights = 1.0 / (1.0 + np.abs(C))  # Less noise for strong correlations
        weighted_noise = noise * noise_weights

        noisy_matrix = C + weighted_noise
        np.fill_diagonal(noisy_matrix, 1.0)

        # Clip to valid correlation range
        noisy_matrix = np.clip(noisy_matrix, -0.99, 0.99)

        return noisy_matrix

    def generate_diverse_training_set(self, n_samples: int = None) -> List[np.ndarray]:
        """
        Kombiniert mehrere Methoden für Diversität
        Generates diverse training set with multiple methods

        Args:
            n_samples: Number of samples to generate (uses config default if None)

        Returns:
            List of correlation matrices
        """
        if n_samples is None:
            n_samples = self.config.n_samples

        N = self.config.n_assets
        training_data = []

        print(f"Generating {n_samples:,} diverse correlation matrices ({N}×{N})")

        # Calculate sample counts for each method
        n_classical = int(self.config.classical_pct * n_samples)
        n_gan = int(self.config.gan_style_pct * n_samples)
        n_bootstrap = int(self.config.bootstrap_pct * n_samples)
        n_block = n_samples - n_classical - n_gan - n_bootstrap  # Remaining samples

        print(
            f"Distribution: {n_classical} classical, {n_gan} GAN-style, "
            f"{n_bootstrap} bootstrap, {n_block} block-structured"
        )

        # 40% classical synthetic matrices
        print("Generating classical synthetic matrices...")
        for i in range(n_classical):
            if i % 1000 == 0:
                print(f"  Classical: {i}/{n_classical}")

            spectrum = self.random_spectrum_generation()
            C = self.random_correlation_matrix(N, spectrum)
            C_noisy = self.add_sampling_noise(C)
            training_data.append(C_noisy)

        # 30% GAN-style generation
        print("Generating GAN-style matrices...")
        for i in range(n_gan):
            if i % 1000 == 0:
                print(f"  GAN-style: {i}/{n_gan}")

            C = self.corrgan_style_generation(N)
            C_noisy = self.add_sampling_noise(C)
            training_data.append(C_noisy)

        # 20% bootstrap from realistic patterns
        print("Generating bootstrap samples...")
        for i in range(n_bootstrap):
            if i % 1000 == 0:
                print(f"  Bootstrap: {i}/{n_bootstrap}")

            C = self.bootstrap_sample(N)
            # Bootstrap samples already have realistic noise, add minimal additional
            C_minimal_noise = self.add_sampling_noise(C)
            training_data.append(C_minimal_noise)

        # 10% block-structured (sectors)
        print("Generating block-structured matrices...")
        for i in range(n_block):
            if i % 1000 == 0:
                print(f"  Block-structured: {i}/{n_block}")

            n_blocks = random.randint(5, 15)
            C = self.generate_block_structure(N, n_blocks)
            C_noisy = self.add_sampling_noise(C)
            training_data.append(C_noisy)

        print(f"✓ Generated {len(training_data):,} diverse correlation matrices")

        return training_data


def validate_training_data(training_data: List[np.ndarray], sample_size: int = 100) -> Dict:
    """
    Validate the diversity and quality of generated training data

    Args:
        training_data: List of correlation matrices
        sample_size: Number of matrices to analyze in detail

    Returns:
        Dictionary with validation statistics
    """
    print(f"\nValidating training data quality (analyzing {sample_size} samples)...")

    # Sample random matrices for detailed analysis
    sample_indices = np.random.choice(len(training_data), min(sample_size, len(training_data)), replace=False)

    stats = {
        "n_matrices": len(training_data),
        "matrix_shape": training_data[0].shape,
        "all_symmetric": True,
        "all_unit_diagonal": True,
        "all_positive_semidefinite": True,
        "eigenvalue_stats": {},
        "correlation_stats": {},
        "diversity_metrics": {},
    }

    eigenvalue_ratios = []
    max_correlations = []
    mean_abs_correlations = []
    condition_numbers = []

    for idx in sample_indices:
        matrix = training_data[idx]

        # Check basic properties
        if not np.allclose(matrix, matrix.T):
            stats["all_symmetric"] = False

        if not np.allclose(np.diag(matrix), 1.0):
            stats["all_unit_diagonal"] = False

        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Descending order

        if np.min(eigenvals) < -1e-10:
            stats["all_positive_semidefinite"] = False

        eigenvalue_ratios.append(eigenvals[0] / eigenvals[-1] if eigenvals[-1] > 1e-10 else np.inf)
        condition_numbers.append(np.linalg.cond(matrix))

        # Correlation statistics
        off_diag = matrix[np.triu_indices_from(matrix, k=1)]
        max_correlations.append(np.max(np.abs(off_diag)))
        mean_abs_correlations.append(np.mean(np.abs(off_diag)))

    # Aggregate statistics
    stats["eigenvalue_stats"] = {
        "condition_number_mean": np.mean(condition_numbers),
        "condition_number_std": np.std(condition_numbers),
        "eigenvalue_ratio_mean": np.mean([x for x in eigenvalue_ratios if np.isfinite(x)]),
        "eigenvalue_ratio_std": np.std([x for x in eigenvalue_ratios if np.isfinite(x)]),
    }

    stats["correlation_stats"] = {
        "max_correlation_mean": np.mean(max_correlations),
        "max_correlation_std": np.std(max_correlations),
        "mean_abs_correlation_mean": np.mean(mean_abs_correlations),
        "mean_abs_correlation_std": np.std(mean_abs_correlations),
    }

    # Diversity metrics (compare first few matrices)
    if len(training_data) >= 10:
        diversity_sample = [training_data[i] for i in range(10)]
        pairwise_distances = []

        for i in range(len(diversity_sample)):
            for j in range(i + 1, len(diversity_sample)):
                distance = np.linalg.norm(diversity_sample[i] - diversity_sample[j], "fro")
                pairwise_distances.append(distance)

        stats["diversity_metrics"] = {
            "mean_pairwise_distance": np.mean(pairwise_distances),
            "std_pairwise_distance": np.std(pairwise_distances),
            "min_pairwise_distance": np.min(pairwise_distances),
            "max_pairwise_distance": np.max(pairwise_distances),
        }

    return stats


# Example usage and testing
if __name__ == "__main__":
    # Configuration for testing
    test_config = DiverseTrainingConfig(
        n_samples=1000,  # Small test set
        n_assets=50,  # Smaller matrices for testing
        random_seed=42,
    )

    # Generate diverse training data
    generator = DiverseCorrelationGenerator(test_config)
    training_data = generator.generate_diverse_training_set()

    # Validate the data
    validation_stats = validate_training_data(training_data)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for key, value in validation_stats.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    print(f"  {subkey}: {subvalue:.4f}")
                else:
                    print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
