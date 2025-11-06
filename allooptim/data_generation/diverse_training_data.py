"""Diverse Covariance Matrix Training Data Generation Module.

Generates 50,000 diverse synthetic correlation matrices using multiple methods
for enhanced autoencoder training.
"""

import random
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

# Import existing generator
from .training_data import CovarianceMatrixGenerator, SpectrumGenerator

# Constants for diverse training data generation
DEFAULT_N_SAMPLES = 50000
DEFAULT_N_ASSETS = 500
CLASSICAL_PCT = 0.4
GAN_STYLE_PCT = 0.3
BOOTSTRAP_PCT = 0.2
BLOCK_PCT = 0.1
DEFAULT_NOISE_LEVEL = 0.05
MIN_EIGENVAL = 1e-6
DEFAULT_RANDOM_SEED = 42

# Spectrum generation parameters
EXPONENTIAL_DECAY_RATE_MIN = 0.5
EXPONENTIAL_DECAY_RATE_MAX = 4.0
POWER_LAW_EXPONENT_MIN = 0.5
POWER_LAW_EXPONENT_MAX = 3.0
FACTOR_MODEL_VARIANCE_MIN = 0.15
FACTOR_MODEL_VARIANCE_MAX = 0.7
UNIFORM_DECAY_START = 1.0
UNIFORM_DECAY_END = 0.1
STEEP_DECAY_START = 0
STEEP_DECAY_END = 8
FLAT_SPECTRUM_VARIATION = 0.2
FLAT_SPECTRUM_MIN = 0.01

# Hierarchical structure parameters
HIERARCHICAL_DECAY_MIN = 2
HIERARCHICAL_DECAY_MAX = 10
HIERARCHICAL_CLUSTERS_MIN = 5
HIERARCHICAL_CLUSTERS_MAX = 15
HIERARCHICAL_WITHIN_CLUSTER_MIN = 1.2
HIERARCHICAL_WITHIN_CLUSTER_MAX = 2.0
HIERARCHICAL_BETWEEN_CLUSTER_MIN = 0.3
HIERARCHICAL_BETWEEN_CLUSTER_MAX = 0.8

# Block structure parameters
BLOCKS_MIN = 3
BLOCKS_MAX = 12
BLOCK_CORRELATION_MIN = 0.1
BLOCK_CORRELATION_MAX = 2.0
INTER_BLOCK_CORRELATION_MIN = -0.2
INTER_BLOCK_CORRELATION_MAX = 0.2
INTER_BLOCK_STRENGTH = 0.1

# Smooth interpolation parameters
BETA_SHAPE_PARAM = 2

# Factor model parameters
N_FACTORS_MIN = 2
N_FACTORS_MAX_RATIO = 5  # N // 5
FACTOR_VARIANCE_MIN = 0.5
FACTOR_VARIANCE_MAX = 3.0
FACTOR_CORR_EIGENVAL_MIN = 0.1
FACTOR_CORR_EIGENVAL_MAX = 1.5
IDIOSYNCRATIC_VARIANCE_MIN = 0.3
IDIOSYNCRATIC_VARIANCE_MAX = 1.0

# Bootstrap parameters
BOOTSTRAP_MARKET_BETA_MIN = 0.3
BOOTSTRAP_MARKET_BETA_MAX = 0.8
CRISIS_FACTOR_MIN = 0.6
CRISIS_FACTOR_MAX = 0.9
CRISIS_SAFE_HAVEN_RATIO = 20  # N // 20
CRISIS_SAFE_HAVEN_CORR = -0.3
LOW_VOL_CORR_MIN = 0.05
LOW_VOL_CORR_MAX = 0.25

# Sector parameters
SECTORS_MIN = 5
SECTORS_MAX = 15
SECTOR_INTRA_CORR_MIN = 0.2
SECTOR_INTRA_CORR_MAX = 0.6
SECTOR_INTER_CORR_MIN = -0.1
SECTOR_INTER_CORR_MAX = 0.2
SECTOR_ROTATION_SECTORS_MIN = 6
SECTOR_ROTATION_SECTORS_MAX = 12
SECTOR_ROTATION_INTRA_MIN = 0.4
SECTOR_ROTATION_INTRA_MAX = 0.8
SECTOR_ROTATION_VARIATION_MIN = 0.8
SECTOR_ROTATION_VARIATION_MAX = 1.2
SECTOR_ROTATION_INTER_MIN = -0.2
SECTOR_ROTATION_INTER_MAX = 0.2

# Block generation parameters
BLOCK_GENERATION_BLOCKS_MIN = 5
BLOCK_GENERATION_BLOCKS_MAX = 15
BLOCK_GENERATION_INTRA_CORR_MIN = 0.4
BLOCK_GENERATION_INTRA_CORR_MAX = 0.8
BLOCK_GENERATION_NOISE_MIN = -0.1
BLOCK_GENERATION_NOISE_MAX = 0.1
BLOCK_GENERATION_INTER_STRENGTH_MIN = 0.05
BLOCK_GENERATION_INTER_STRENGTH_MAX = 0.15

# Sampling noise parameters
SAMPLE_SIZE_MIN = 50
SAMPLE_SIZE_MAX = 500
VALIDATION_SAMPLE_SIZE = 100
VALIDATION_EIGENVAL_THRESHOLD = 1e-10
DIVERSITY_SAMPLE_SIZE = 10


@dataclass
class DiverseTrainingConfig:
    """Configuration for diverse training data generation."""

    n_samples: int = DEFAULT_N_SAMPLES
    n_assets: int = DEFAULT_N_ASSETS

    # Distribution percentages (should sum to 1.0)
    classical_pct: float = CLASSICAL_PCT  # 40% classical synthetic
    gan_style_pct: float = GAN_STYLE_PCT  # 30% GAN-style
    bootstrap_pct: float = BOOTSTRAP_PCT  # 20% bootstrap from real patterns
    block_pct: float = BLOCK_PCT  # 10% block-structured

    # Noise parameters
    noise_level: float = DEFAULT_NOISE_LEVEL  # Amount of sampling noise to add
    min_eigenval: float = MIN_EIGENVAL  # Minimum eigenvalue for numerical stability

    random_seed: int = DEFAULT_RANDOM_SEED


class DiverseCorrelationGenerator:
    """Enhanced correlation matrix generator with multiple diverse methods."""

    def __init__(self, config: DiverseTrainingConfig = None):
        """Initialize the diverse correlation generator.

        Args:
            config: Configuration object for generation parameters. If None, uses default config.
        """
        self.config = config or DiverseTrainingConfig()
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

    def random_spectrum_generation(self) -> np.ndarray:
        """Generate diverse eigenvalue spectra using various distributions.

        Returns:
            Array of eigenvalues (normalized to sum to n_assets)
        """
        n = self.config.n_assets
        spectrum_type = random.choice(
            ["exponential_decay", "power_law", "factor_model", "uniform_decay", "steep_decay", "flat_spectrum"]
        )

        if spectrum_type == "exponential_decay":
            eigenvals = SpectrumGenerator.exponential_decay(
                n, decay_rate=np.random.uniform(EXPONENTIAL_DECAY_RATE_MIN, EXPONENTIAL_DECAY_RATE_MAX)
            )
        elif spectrum_type == "power_law":
            eigenvals = SpectrumGenerator.power_law(
                n, exponent=np.random.uniform(POWER_LAW_EXPONENT_MIN, POWER_LAW_EXPONENT_MAX)
            )
        elif spectrum_type == "factor_model":
            eigenvals = SpectrumGenerator.factor_model(
                n, market_variance_explained=np.random.uniform(FACTOR_MODEL_VARIANCE_MIN, FACTOR_MODEL_VARIANCE_MAX)
            )
        elif spectrum_type == "uniform_decay":
            # Linear decay
            eigenvals = np.linspace(UNIFORM_DECAY_START, UNIFORM_DECAY_END, n)
            eigenvals = eigenvals / np.sum(eigenvals) * n
        elif spectrum_type == "steep_decay":
            # Very steep exponential decay (few dominant factors)
            eigenvals = np.exp(-np.linspace(STEEP_DECAY_START, STEEP_DECAY_END, n))
            eigenvals = eigenvals / np.sum(eigenvals) * n
        else:  # flat_spectrum
            # Nearly uniform eigenvalues (low correlation regime)
            eigenvals = np.ones(n) + np.random.uniform(-FLAT_SPECTRUM_VARIATION, FLAT_SPECTRUM_VARIATION, n)
            eigenvals = np.maximum(eigenvals, FLAT_SPECTRUM_MIN)
            eigenvals = eigenvals / np.sum(eigenvals) * n

        return eigenvals

    def random_correlation_matrix(self, N: int, spectrum: np.ndarray = None) -> np.ndarray:
        """Generate correlation matrix from spectrum using random orthogonal basis.

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
        """Simulate GAN-style generation with more complex patterns.

        Mimics what a CorrGAN might produce with diverse structures.

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
        """Generate hierarchical correlation structure."""
        # Create base correlation with distance-based decay
        positions = np.random.uniform(0, 1, N)  # Random positions on [0,1]
        distances = np.abs(positions[:, None] - positions[None, :])

        # Exponential decay with random parameters
        decay_rate = np.random.uniform(HIERARCHICAL_DECAY_MIN, HIERARCHICAL_DECAY_MAX)
        base_corr = np.exp(-decay_rate * distances)

        # Add hierarchical clustering
        n_clusters = random.randint(HIERARCHICAL_CLUSTERS_MIN, HIERARCHICAL_CLUSTERS_MAX)
        cluster_assignments = np.random.randint(0, n_clusters, N)

        for i in range(N):
            for j in range(i + 1, N):
                if cluster_assignments[i] == cluster_assignments[j]:
                    # Within cluster: higher correlation
                    base_corr[i, j] *= np.random.uniform(
                        HIERARCHICAL_WITHIN_CLUSTER_MIN, HIERARCHICAL_WITHIN_CLUSTER_MAX
                    )
                else:
                    # Between clusters: lower correlation
                    base_corr[i, j] *= np.random.uniform(
                        HIERARCHICAL_BETWEEN_CLUSTER_MIN, HIERARCHICAL_BETWEEN_CLUSTER_MAX
                    )

        # Ensure symmetry and proper diagonal
        base_corr = (base_corr + base_corr.T) / 2
        np.fill_diagonal(base_corr, 1.0)

        # Make positive definite
        eigenvals, eigenvecs = np.linalg.eigh(base_corr)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _generate_random_blocks(self, N: int) -> np.ndarray:
        """Generate random block structure."""
        n_blocks = random.randint(BLOCKS_MIN, BLOCKS_MAX)
        block_sizes = np.random.multinomial(N, np.ones(n_blocks) / n_blocks)
        block_sizes = block_sizes[block_sizes > 0]  # Remove zero-size blocks

        blocks = []
        for size in block_sizes:
            if size == 1:
                blocks.append(np.array([[1.0]]))
            else:
                # Generate random correlation within block
                eigenvals = np.random.uniform(BLOCK_CORRELATION_MIN, BLOCK_CORRELATION_MAX, size)
                eigenvals = eigenvals / np.sum(eigenvals) * size
                block_corr = CovarianceMatrixGenerator.from_eigenvalues(eigenvals)
                blocks.append(block_corr)

        # Create block diagonal matrix
        block_matrix = block_diag(*blocks)

        # Add small off-block correlations
        off_block_corr = np.random.uniform(INTER_BLOCK_CORRELATION_MIN, INTER_BLOCK_CORRELATION_MAX, (N, N))
        off_block_corr = (off_block_corr + off_block_corr.T) / 2

        # Combine with decay for distant blocks
        final_matrix = block_matrix + INTER_BLOCK_STRENGTH * off_block_corr

        # Ensure proper correlation matrix
        np.fill_diagonal(final_matrix, 1.0)
        eigenvals, eigenvecs = np.linalg.eigh(final_matrix)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _generate_smooth_interpolation(self, N: int) -> np.ndarray:
        """Generate smooth interpolation between different patterns."""
        # Create two different correlation patterns
        eigenvals1 = SpectrumGenerator.exponential_decay(N, decay_rate=1.0)
        eigenvals2 = SpectrumGenerator.power_law(N, exponent=2.0)

        matrix1 = CovarianceMatrixGenerator.from_eigenvalues(eigenvals1)
        matrix2 = CovarianceMatrixGenerator.from_eigenvalues(eigenvals2)

        # Random interpolation weight
        alpha = np.random.beta(BETA_SHAPE_PARAM, BETA_SHAPE_PARAM)  # Beta distribution for smooth mixing

        interpolated = alpha * matrix1 + (1 - alpha) * matrix2

        # Ensure proper correlation matrix
        np.fill_diagonal(interpolated, 1.0)
        eigenvals, eigenvecs = np.linalg.eigh(interpolated)
        eigenvals = np.maximum(eigenvals, self.config.min_eigenval)
        eigenvals = eigenvals / np.sum(eigenvals) * N

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _generate_noisy_factor_model(self, N: int) -> np.ndarray:
        """Generate noisy factor model with additional complexity."""
        n_factors = random.randint(N_FACTORS_MIN, min(N_FACTORS_MAX_RATIO, N // N_FACTORS_MAX_RATIO))

        # Random factor loadings
        loadings = np.random.normal(0, 1, (N, n_factors))

        # Different factor variances
        factor_vars = np.random.uniform(FACTOR_VARIANCE_MIN, FACTOR_VARIANCE_MAX, n_factors)

        # Factor correlation matrix (factors can be correlated)
        if n_factors > 1:
            factor_corr_eigenvals = np.random.uniform(FACTOR_CORR_EIGENVAL_MIN, FACTOR_CORR_EIGENVAL_MAX, n_factors)
            factor_corr_eigenvals = factor_corr_eigenvals / np.sum(factor_corr_eigenvals) * n_factors
            factor_corr = CovarianceMatrixGenerator.from_eigenvalues(factor_corr_eigenvals)
        else:
            factor_corr = np.array([[1.0]])

        # Construct covariance matrix
        cov_matrix = loadings @ (factor_corr * factor_vars) @ loadings.T

        # Add idiosyncratic variances
        idiosyncratic_vars = np.random.uniform(IDIOSYNCRATIC_VARIANCE_MIN, IDIOSYNCRATIC_VARIANCE_MAX, N)
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
        """Generate correlation matrix based on realistic market patterns.

        Simulates bootstrap sampling from real financial data.

        Args:
            N: Matrix dimension

        Returns:
            Correlation matrix
        """
        # Simulate different market regimes
        regime = random.choice(["normal", "crisis", "low_vol", "sector_rotation"])

        if regime == "normal":
            # Normal market: moderate correlations, factor structure
            market_beta = np.random.uniform(BOOTSTRAP_MARKET_BETA_MIN, BOOTSTRAP_MARKET_BETA_MAX, N)
            sector_effects = self._generate_sector_effects(N)
            base_corr = np.outer(market_beta, market_beta) + sector_effects

        elif regime == "crisis":
            # Crisis: high correlations, few factors dominate
            crisis_factor = np.random.uniform(CRISIS_FACTOR_MIN, CRISIS_FACTOR_MAX, N)
            base_corr = np.outer(crisis_factor, crisis_factor)
            # Add some negative correlations (safe havens)
            n_safe_havens = random.randint(1, max(1, N // CRISIS_SAFE_HAVEN_RATIO))
            safe_haven_indices = np.random.choice(N, n_safe_havens, replace=False)
            for idx in safe_haven_indices:
                base_corr[idx, :] *= CRISIS_SAFE_HAVEN_CORR
                base_corr[:, idx] *= CRISIS_SAFE_HAVEN_CORR

        elif regime == "low_vol":
            # Low volatility: lower correlations, more idiosyncratic
            base_corr = np.random.uniform(LOW_VOL_CORR_MIN, LOW_VOL_CORR_MAX, (N, N))
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
        """Generate sector-based correlation effects."""
        n_sectors = random.randint(SECTORS_MIN, SECTORS_MAX)
        sector_assignments = np.random.randint(0, n_sectors, N)

        sector_corr = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if sector_assignments[i] == sector_assignments[j]:
                    sector_corr[i, j] = np.random.uniform(SECTOR_INTRA_CORR_MIN, SECTOR_INTRA_CORR_MAX)
                else:
                    sector_corr[i, j] = np.random.uniform(SECTOR_INTER_CORR_MIN, SECTOR_INTER_CORR_MAX)

        return sector_corr + sector_corr.T

    def _generate_sector_rotation_pattern(self, N: int) -> np.ndarray:
        """Generate sector rotation correlation pattern."""
        n_sectors = random.randint(SECTOR_ROTATION_SECTORS_MIN, SECTOR_ROTATION_SECTORS_MAX)
        sector_size = N // n_sectors

        corr_matrix = np.zeros((N, N))

        # Strong intra-sector correlations
        for sector in range(n_sectors):
            start_idx = sector * sector_size
            end_idx = min((sector + 1) * sector_size, N)

            sector_corr = np.random.uniform(SECTOR_ROTATION_INTRA_MIN, SECTOR_ROTATION_INTRA_MAX)
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        corr_matrix[i, j] = sector_corr * np.random.uniform(
                            SECTOR_ROTATION_VARIATION_MIN, SECTOR_ROTATION_VARIATION_MAX
                        )

        # Weak inter-sector correlations
        for i in range(N):
            for j in range(i + 1, N):
                if corr_matrix[i, j] == 0:  # Not in same sector
                    corr_matrix[i, j] = np.random.uniform(SECTOR_ROTATION_INTER_MIN, SECTOR_ROTATION_INTER_MAX)

        return corr_matrix + corr_matrix.T

    def generate_block_structure(self, N: int, n_blocks: int = None) -> np.ndarray:
        """Generate block-structured correlation matrix (sectors).

        Args:
            N: Matrix dimension
            n_blocks: Number of blocks (if None, randomly chosen)

        Returns:
            Block-structured correlation matrix
        """
        if n_blocks is None:
            n_blocks = random.randint(BLOCK_GENERATION_BLOCKS_MIN, BLOCK_GENERATION_BLOCKS_MAX)

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
                base_corr = np.random.uniform(BLOCK_GENERATION_INTRA_CORR_MIN, BLOCK_GENERATION_INTRA_CORR_MAX)
                block_matrix = np.full((size, size), base_corr)
                np.fill_diagonal(block_matrix, 1.0)

                # Add some noise to make it more realistic
                noise = np.random.uniform(BLOCK_GENERATION_NOISE_MIN, BLOCK_GENERATION_NOISE_MAX, (size, size))
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
        inter_block_strength = np.random.uniform(
            BLOCK_GENERATION_INTER_STRENGTH_MIN, BLOCK_GENERATION_INTER_STRENGTH_MAX
        )
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
        """Add finite sample estimation noise to correlation matrix.

        Simulates the noise that comes from estimating correlations from limited data.

        Args:
            C: Input correlation matrix

        Returns:
            Correlation matrix with added sampling noise
        """
        N = C.shape[0]

        # Simulate finite sample effects
        sample_size = random.randint(SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX)  # Typical sample sizes in finance

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

    def generate_diverse_training_set(self, n_samples: int = None) -> list[np.ndarray]:
        """Kombiniert mehrere Methoden für Diversität.

        Generates diverse training set with multiple methods.

        Args:
            n_samples: Number of samples to generate (uses config default if None)

        Returns:
            list of correlation matrices
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


def validate_training_data(training_data: list[np.ndarray], sample_size: int = VALIDATION_SAMPLE_SIZE) -> dict:
    """Validate the diversity and quality of generated training data.

    Args:
        training_data: list of correlation matrices
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

        if np.min(eigenvals) < VALIDATION_EIGENVAL_THRESHOLD:
            stats["all_positive_semidefinite"] = False

        eigenvalue_ratios.append(
            eigenvals[0] / eigenvals[-1] if eigenvals[-1] > VALIDATION_EIGENVAL_THRESHOLD else np.inf
        )
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
    if len(training_data) >= DIVERSITY_SAMPLE_SIZE:
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
