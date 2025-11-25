"""Covariance Matrix Training Data Generation Module.

Generates synthetic correlation matrices for training denoising autoencoders.
"""

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Optional

import h5py
import numpy as np
from scipy.stats import ortho_group

# Constants for numerical tolerances
MATRIX_CLOSE_TO_IDENTITY_TOLERANCE = 1e-10
MINIMUM_EIGENVALUE_THRESHOLD = 1e-8


@dataclass
class TrainingConfig:
    """Configuration for training data generation."""

    n_assets: int = 500
    n_samples: int = 50000
    min_observations: int = 100
    max_observations: int = 2000
    n_processes: Optional[int] = None  # None = use all available cores
    output_file: str = "training_data.h5"
    random_seed: int = 42


class SpectrumGenerator:
    """Generates various types of eigenvalue spectra."""

    @staticmethod
    def exponential_decay(n: int, decay_rate: Optional[float] = None) -> np.ndarray:
        """Exponentially decaying spectrum (most common in equity markets).

        Args:
            n: Number of eigenvalues
            decay_rate: Decay parameter (if None, randomly chosen)

        Returns:
            Array of eigenvalues in descending order
        """
        if decay_rate is None:
            decay_rate = np.random.uniform(0.5, 5.0)

        eigenvalues = np.exp(-np.linspace(0, decay_rate, n))
        return eigenvalues / np.sum(eigenvalues) * n

    @staticmethod
    def power_law(n: int, exponent: Optional[float] = None) -> np.ndarray:
        """Power-law spectrum (Zipf-like distribution).

        Args:
            n: Number of eigenvalues
            exponent: Power law exponent (if None, randomly chosen)

        Returns:
            Array of eigenvalues in descending order
        """
        if exponent is None:
            exponent = np.random.uniform(0.3, 1.5)

        eigenvalues = 1.0 / (np.arange(1, n + 1) ** exponent)
        return eigenvalues / np.sum(eigenvalues) * n

    @staticmethod
    def spiked_model(
        n: int, n_factors: Optional[int] = None, factor_strength: tuple[float, float] = (5.0, 20.0)
    ) -> np.ndarray:
        """Spiked model: few large factors + noise.

        Models market with strong factors (market, sector, etc.).

        Args:
            n: Number of eigenvalues
            n_factors: Number of large factors (if None, randomly chosen)
            factor_strength: Range for factor eigenvalues

        Returns:
            Array of eigenvalues in descending order
        """
        if n_factors is None:
            n_factors = np.random.randint(3, max(4, min(15, n // 10)))

        # Large eigenvalues (factors)
        large_eigs = np.random.uniform(factor_strength[0], factor_strength[1], n_factors)

        # Small eigenvalues (noise)
        small_eigs = np.random.uniform(0.1, 1.0, n - n_factors)

        eigenvalues = np.concatenate([large_eigs, small_eigs])
        return eigenvalues / np.sum(eigenvalues) * n

    @staticmethod
    def hierarchical(n: int, n_levels: Optional[int] = None) -> np.ndarray:
        """Hierarchical spectrum: multiple scales of correlation.

        Models markets with sector structure.

        Args:
            n: Number of eigenvalues
            n_levels: Number of hierarchical levels

        Returns:
            Array of eigenvalues in descending order
        """
        if n_levels is None:
            n_levels = np.random.randint(2, 5)

        eigenvalues: list[float] = []
        remaining = n

        for level in range(n_levels):
            # Number of eigenvalues at this level
            n_level = max(1, remaining // (n_levels - level))

            # Strength decreases with level
            strength = 10.0 / (level + 1)
            level_eigs = np.random.uniform(strength * 0.5, strength, n_level)

            eigenvalues.extend(level_eigs)
            remaining -= n_level

        eigenvalues_array = np.array(eigenvalues)
        return eigenvalues_array / np.sum(eigenvalues_array) * n

    @staticmethod
    def flat_with_noise(n: int, base_level: Optional[float] = None) -> np.ndarray:
        """Nearly flat spectrum with small perturbations.

        Args:
            n: Number of eigenvalues
            base_level: Base eigenvalue level

        Returns:
            Array of eigenvalues in descending order
        """
        if base_level is None:
            base_level = 1.0

        eigenvalues = base_level + np.random.normal(0, 0.2 * base_level, n)
        eigenvalues = np.maximum(eigenvalues, 0.1)  # Ensure positive
        return eigenvalues / np.sum(eigenvalues) * n

    @staticmethod
    def market_model(n: int, market_variance_explained: Optional[float] = None) -> np.ndarray:
        """Single factor (market) model with idiosyncratic risk.

        Args:
            n: Number of eigenvalues
            market_variance_explained: Fraction of variance explained by market

        Returns:
            Array of eigenvalues in descending order
        """
        if market_variance_explained is None:
            market_variance_explained = np.random.uniform(0.2, 0.6)

        # First eigenvalue is the market
        market_eig = market_variance_explained * n

        # Remaining eigenvalues are idiosyncratic
        idio_eigs = np.random.uniform(0.5, 1.5, n - 1)
        idio_eigs = idio_eigs / np.sum(idio_eigs) * (1 - market_variance_explained) * n

        eigenvalues = np.concatenate([[market_eig], idio_eigs])
        return eigenvalues


class CovarianceMatrixGenerator:
    """Generates random correlation/covariance matrices."""

    @staticmethod
    def from_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
        """Generate correlation matrix from specified eigenvalues.

        Using Davies-Higham algorithm with Givens rotations.

        Args:
            eigenvalues: Array of eigenvalues (should sum to N)

        Returns:
            Correlation matrix
        """
        n = len(eigenvalues)

        # Generate random orthogonal matrix
        Q = ortho_group.rvs(n)

        # Construct matrix with given eigenvalues
        Lambda = np.diag(eigenvalues)
        M = Q.T @ Lambda @ Q

        # Apply Givens rotations to get 1's on diagonal
        corr = CovarianceMatrixGenerator._apply_givens_rotations(M)

        return corr

    @staticmethod
    def _apply_givens_rotations(M: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """Apply Givens rotations to transform matrix to correlation form.

        Args:
            M: Symmetric positive definite matrix
            max_iter: Maximum iterations

        Returns:
            Correlation matrix with 1's on diagonal
        """
        corr = M.copy()
        precision = 1e-6

        for _ in range(max_iter):
            diag = np.diag(corr)

            # Find elements not equal to 1
            deviations = np.abs(diag - 1.0)

            if np.max(deviations) < precision:
                break

            # Find pair to rotate
            bigger = np.where(diag > 1.0 + precision)[0]
            smaller = np.where(diag < 1.0 - precision)[0]

            if len(bigger) == 0 or len(smaller) == 0:
                break

            i, j = smaller[0], bigger[-1]
            if i > j:
                i, j = bigger[0], smaller[-1]

            # Apply Givens rotation
            corr = CovarianceMatrixGenerator._givens_rotation(corr, i, j)

        # Ensure exact 1's on diagonal
        np.fill_diagonal(corr, 1.0)

        # Ensure symmetry
        corr = (corr + corr.T) / 2

        return corr

    @staticmethod
    def _givens_rotation(M: np.ndarray, i: int, j: int) -> np.ndarray:
        """Apply Givens rotation to matrix M at position (i, j).

        Args:
            M: Matrix to rotate
            i: First rotation index
            j: Second rotation index

        Returns:
            Rotated matrix
        """
        Mii, Mij, Mjj = M[i, i], M[i, j], M[j, j]

        # Compute rotation parameters
        if abs(Mjj - 1.0) < MATRIX_CLOSE_TO_IDENTITY_TOLERANCE:
            return M

        discriminant = Mij**2 - (Mii - 1) * (Mjj - 1)
        if discriminant < 0:
            discriminant = 0

        t = (Mij + np.sqrt(discriminant)) / (Mjj - 1)
        c = 1.0 / np.sqrt(1 + t**2)
        s = c * t

        # Apply rotation: G = [[c, -s], [s, c]]
        # M' = G^T M G
        G = np.eye(M.shape[0])
        G[i, i] = c
        G[i, j] = -s
        G[j, i] = s
        G[j, j] = c

        return G.T @ M @ G

    @staticmethod
    def block_structure(
        n: int, n_blocks: int, intra_block_corr: Optional[float] = None, inter_block_corr: Optional[float] = None
    ) -> np.ndarray:
        """Generate correlation matrix with block structure (sectors).

        Args:
            n: Matrix dimension
            n_blocks: Number of blocks
            intra_block_corr: Correlation within blocks
            inter_block_corr: Correlation between blocks

        Returns:
            Block-structured correlation matrix
        """
        if intra_block_corr is None:
            intra_block_corr = np.random.uniform(0.5, 0.9)
        if inter_block_corr is None:
            inter_block_corr = np.random.uniform(0.1, 0.3)

        # Divide assets into blocks
        block_sizes = np.random.multinomial(n, [1 / n_blocks] * n_blocks)

        corr = np.ones((n, n)) * inter_block_corr

        start = 0
        for block_size in block_sizes:
            end = start + block_size
            corr[start:end, start:end] = intra_block_corr
            start = end

        np.fill_diagonal(corr, 1.0)

        # Add small random noise
        noise = np.random.normal(0, 0.05, (n, n))
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)

        corr = corr + noise

        # Project to valid correlation matrix
        corr = CovarianceMatrixGenerator._project_to_correlation(corr)

        return corr

    @staticmethod
    def _project_to_correlation(M: np.ndarray) -> np.ndarray:
        """Project matrix to valid correlation matrix space.

        Args:
            M: Symmetric matrix

        Returns:
            Valid correlation matrix
        """
        # Ensure symmetry
        M = (M + M.T) / 2

        # Clip off-diagonal elements to [-1, 1]
        M = np.clip(M, -0.99, 0.99)
        np.fill_diagonal(M, 1.0)

        # Make positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        eigenvalues = np.maximum(eigenvalues, MINIMUM_EIGENVALUE_THRESHOLD)
        M = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Rescale to correlation matrix
        d = np.sqrt(np.diag(M))
        M = M / np.outer(d, d)
        np.fill_diagonal(M, 1.0)

        return M

    @staticmethod
    def toeplitz_blocks(n: int, n_blocks: int) -> np.ndarray:
        """Generate correlation matrix with Toeplitz-structured blocks.

        Args:
            n: Matrix dimension
            n_blocks: Number of blocks

        Returns:
            Toeplitz-block correlation matrix
        """
        block_sizes = np.random.multinomial(n, [1 / n_blocks] * n_blocks)

        corr = np.zeros((n, n))
        start = 0

        for block_size in block_sizes:
            end = start + block_size

            # Random correlation parameter for this block
            rho = np.random.uniform(0.3, 0.8)

            # Create Toeplitz structure
            block = np.array([[rho ** abs(i - j) for j in range(block_size)] for i in range(block_size)])

            corr[start:end, start:end] = block
            start = end

        # Add small inter-block correlations
        inter_corr = np.random.uniform(0.05, 0.15)
        corr = corr + (1 - np.eye(n)) * inter_corr * (corr == 0)

        np.fill_diagonal(corr, 1.0)

        return CovarianceMatrixGenerator._project_to_correlation(corr)


class NoisyObservationGenerator:
    """Generates noisy sample covariance matrices from true covariance."""

    @staticmethod
    def add_estimation_noise(true_cov: np.ndarray, n_observations: int, return_eigvals: bool = True) -> dict:
        """Generate noisy sample covariance matrix by simulating data.

        Args:
            true_cov: True correlation/covariance matrix
            n_observations: Number of observations to simulate
            return_eigvals: If True, return eigenvalues; if False, return full matrix

        Returns:
            Dictionary containing sample and true information
        """
        n = true_cov.shape[0]

        # Generate synthetic returns from multivariate normal
        returns = np.random.multivariate_normal(mean=np.zeros(n), cov=true_cov, size=n_observations)

        # Compute sample covariance matrix
        sample_cov = np.cov(returns.T)

        # Ensure positive definite
        min_eig = np.min(np.linalg.eigvalsh(sample_cov))
        if min_eig < MINIMUM_EIGENVALUE_THRESHOLD:
            sample_cov += (1e-6 - min_eig) * np.eye(n)

        if return_eigvals:
            # Extract eigenvalues (sorted in ascending order for consistency)
            true_eigenvalues = np.sort(np.linalg.eigvalsh(true_cov))
            sample_eigenvalues = np.sort(np.linalg.eigvalsh(sample_cov))

            return {
                "sample_eigenvalues": sample_eigenvalues.astype(np.float32),
                "true_eigenvalues": true_eigenvalues.astype(np.float32),
                "q": n / n_observations,
                "n_observations": n_observations,
            }
        else:
            return {
                "sample_cov": sample_cov.astype(np.float32),
                "true_cov": true_cov.astype(np.float32),
                "q": n / n_observations,
                "n_observations": n_observations,
            }


class TrainingDataGenerator:
    """Main class for generating complete training dataset."""

    def __init__(self, config: TrainingConfig):
        """Initialize the training data generator.

        Args:
            config: Configuration object containing generation parameters
        """
        self.config = config
        self.spectrum_gen = SpectrumGenerator()
        self.cov_gen = CovarianceMatrixGenerator()
        self.noise_gen = NoisyObservationGenerator()

    def generate_single_sample(self, sample_idx: int) -> dict:
        """Generate a single training sample.

        Args:
            sample_idx: Sample index (used for seeding)

        Returns:
            Dictionary with training data
        """
        # Set seed for reproducibility
        np.random.seed(self.config.random_seed + sample_idx)

        # Randomly choose generation method
        method = np.random.choice(
            [
                "eigenvalue_exponential",
                "eigenvalue_power_law",
                "eigenvalue_spiked",
                "eigenvalue_hierarchical",
                "eigenvalue_flat",
                "eigenvalue_market",
                "block_structure",
                "toeplitz_blocks",
            ],
            p=[0.25, 0.15, 0.20, 0.10, 0.05, 0.10, 0.10, 0.05],
        )

        # Generate true correlation matrix
        if method.startswith("eigenvalue"):
            spectrum_type = method.replace("eigenvalue_", "")
            eigenvalues = self._generate_spectrum(spectrum_type)
            true_cov = self.cov_gen.from_eigenvalues(eigenvalues)
        elif method == "block_structure":
            n_blocks = np.random.randint(5, 20)
            true_cov = self.cov_gen.block_structure(self.config.n_assets, n_blocks)
        elif method == "toeplitz_blocks":
            n_blocks = np.random.randint(5, 15)
            true_cov = self.cov_gen.toeplitz_blocks(self.config.n_assets, n_blocks)

        # Random number of observations
        n_obs = np.random.randint(self.config.min_observations, self.config.max_observations + 1)

        # Generate noisy observation
        sample_data = self.noise_gen.add_estimation_noise(true_cov, n_obs)

        return sample_data

    def _generate_spectrum(self, spectrum_type: str) -> np.ndarray:
        """Generate eigenvalue spectrum based on type."""
        if spectrum_type == "exponential":
            return self.spectrum_gen.exponential_decay(self.config.n_assets)
        elif spectrum_type == "power_law":
            return self.spectrum_gen.power_law(self.config.n_assets)
        elif spectrum_type == "spiked":
            return self.spectrum_gen.spiked_model(self.config.n_assets)
        elif spectrum_type == "hierarchical":
            return self.spectrum_gen.hierarchical(self.config.n_assets)
        elif spectrum_type == "flat":
            return self.spectrum_gen.flat_with_noise(self.config.n_assets)
        elif spectrum_type == "market":
            return self.spectrum_gen.market_model(self.config.n_assets)
        else:
            raise ValueError(f"Unknown spectrum type: {spectrum_type}")

    def generate_parallel(self, verbose: bool = True) -> list[dict]:
        """Generate training data in parallel.

        Args:
            verbose: Print progress

        Returns:
            list of training samples
        """
        n_processes = self.config.n_processes or mp.cpu_count()

        if verbose:
            print(f"Generating {self.config.n_samples} samples using {n_processes} processes...")

        with mp.Pool(processes=n_processes) as pool:
            samples = pool.map(self.generate_single_sample, range(self.config.n_samples))

        if verbose:
            print(f"Generated {len(samples)} samples successfully!")

        return samples

    def save_to_hdf5(self, samples: list[dict], filename: Optional[str] = None):
        """Save training data to HDF5 file.

        Args:
            samples: list of training samples
            filename: Output filename (uses config if None)
        """
        if filename is None:
            filename = self.config.output_file

        # Convert list of dicts to arrays
        sample_eigenvalues = np.array([s["sample_eigenvalues"] for s in samples])
        true_eigenvalues = np.array([s["true_eigenvalues"] for s in samples])
        q_values = np.array([s["q"] for s in samples], dtype=np.float32)
        n_observations = np.array([s["n_observations"] for s in samples], dtype=np.int32)

        # Save to HDF5
        with h5py.File(filename, "w") as f:
            f.create_dataset("sample_eigenvalues", data=sample_eigenvalues, compression="gzip", compression_opts=9)
            f.create_dataset("true_eigenvalues", data=true_eigenvalues, compression="gzip", compression_opts=9)
            f.create_dataset("q_values", data=q_values)
            f.create_dataset("n_observations", data=n_observations)

            # Store metadata
            f.attrs["n_assets"] = self.config.n_assets
            f.attrs["n_samples"] = self.config.n_samples
            f.attrs["min_observations"] = self.config.min_observations
            f.attrs["max_observations"] = self.config.max_observations

        print(f"Data saved to {filename}")
        print(f"File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")


def load_training_data(filename: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load training data from HDF5 file.

    Args:
        filename: HDF5 filename

    Returns:
        Dictionary with training data
    """
    with h5py.File(filename, "r") as f:
        data = {
            "sample_eigenvalues": f["sample_eigenvalues"][:],
            "true_eigenvalues": f["true_eigenvalues"][:],
            "q_values": f["q_values"][:],
            "n_observations": f["n_observations"][:],
        }

        metadata = {
            "n_assets": f.attrs["n_assets"],
            "n_samples": f.attrs["n_samples"],
            "min_observations": f.attrs["min_observations"],
            "max_observations": f.attrs["max_observations"],
        }

    return data, metadata
