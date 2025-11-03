"""
Diverse Covariance Matrix Training Data Generation Module
Generates 30,000 artificial covariance matrices using multiple methods for autoencoder training
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ortho_group

# Constants for data generation thresholds
MIN_ASSETS_FOR_BOOTSTRAP = 10
OVERDETERMINED_EXTRA_COLUMNS = 10


@dataclass
class CovarianceConfig:
	"""Configuration for covariance matrix generation"""

	n_assets: int = 500
	n_samples: int = 30000
	random_seed: int = 42

	# Distribution percentages (should sum to 1.0)
	pct_synthetic: float = 0.50  # 50% synthetic eigenvalue-based
	pct_gan_style: float = 0.25  # 25% GAN-style diverse patterns
	pct_real_based: float = 0.15  # 15% real data bootstrap
	pct_block_struct: float = 0.10  # 10% block-structured (sectors)

	# Scaling parameters
	volatility_scale: tuple[float, float] = (0.01, 0.05)  # Annual volatility range
	noise_level: float = 0.02  # Sampling noise level


class SpectrumGenerator:
	"""Generates various types of eigenvalue spectra for covariance matrices"""

	@staticmethod
	def exponential_decay(n_assets: int, decay_rate: float = 2.0) -> np.ndarray:
		"""Generate exponentially decaying eigenvalues (market-like)"""
		i = np.arange(1, n_assets + 1)
		eigenvals = np.exp(-decay_rate * (i - 1) / n_assets)
		return eigenvals * n_assets / np.sum(eigenvals)

	@staticmethod
	def power_law(n_assets: int, alpha: float = 1.5) -> np.ndarray:
		"""Generate power-law eigenvalues (heavy-tailed)"""
		i = np.arange(1, n_assets + 1)
		eigenvals = i ** (-alpha)
		return eigenvals * n_assets / np.sum(eigenvals)

	@staticmethod
	def mixed_regime(n_assets: int, n_factors: int = None) -> np.ndarray:
		"""Generate mixed regime eigenvalues (few large, many small)"""
		if n_factors is None:
			n_factors = max(3, n_assets // 20)

		# Large factors
		large_vals = np.linspace(8.0, 2.0, n_factors)

		# Small factors with exponential decay
		remaining = n_assets - n_factors
		small_vals = np.exp(-2 * np.arange(remaining) / remaining) * 0.5

		eigenvals = np.concatenate([large_vals, small_vals])
		return eigenvals * n_assets / np.sum(eigenvals)

	@staticmethod
	def random_spectrum(n_assets: int) -> np.ndarray:
		"""Generate random but realistic eigenvalue spectrum"""
		# Ensure valid range for mixed_regime n_factors
		max_factors = max(4, n_assets // 15)  # At least 4, up to n_assets//15
		min_factors = min(3, max_factors - 1)  # At least 3, but less than max

		methods = [
			lambda: SpectrumGenerator.exponential_decay(n_assets, np.random.uniform(1.0, 3.0)),
			lambda: SpectrumGenerator.power_law(n_assets, np.random.uniform(1.2, 2.5)),
			lambda: SpectrumGenerator.mixed_regime(n_assets, np.random.randint(min_factors, max_factors + 1)),
		]
		return random.choice(methods)()


class CovarianceMatrixGenerator:
	"""Main class for generating diverse covariance matrices"""

	def __init__(self, config: CovarianceConfig):
		self.config = config
		np.random.seed(config.random_seed)
		random.seed(config.random_seed)

	def generate_synthetic_covariance(self) -> np.ndarray:
		"""Generate synthetic covariance matrix from eigenvalue spectrum"""
		# Get random eigenvalue spectrum
		eigenvals = SpectrumGenerator.random_spectrum(self.config.n_assets)

		# Generate random orthogonal matrix
		Q = ortho_group.rvs(self.config.n_assets)

		# Construct correlation matrix
		corr_matrix = Q @ np.diag(eigenvals) @ Q.T

		# Convert to covariance matrix with realistic volatilities
		volatilities = np.random.uniform(*self.config.volatility_scale, self.config.n_assets)
		vol_matrix = np.outer(volatilities, volatilities)
		cov_matrix = corr_matrix * vol_matrix

		return self._ensure_positive_definite(cov_matrix)

	def generate_gan_style_covariance(self) -> np.ndarray:
		"""Generate covariance matrix using GAN-style diverse patterns"""
		# Create multiple random components and combine
		n = self.config.n_assets

		# Base structure: random correlation
		base_corr = self._generate_random_correlation()

		# Add structured noise patterns
		noise_patterns = []
		for _ in range(3):  # Multiple noise sources
			pattern = np.random.randn(n, n)
			pattern = (pattern + pattern.T) / 2  # Symmetrize
			noise_patterns.append(pattern * np.random.uniform(0.05, 0.15))

		# Combine patterns
		combined = base_corr + sum(noise_patterns)
		combined = (combined + combined.T) / 2  # Ensure symmetry

		# Convert to covariance with scaling
		volatilities = np.random.uniform(*self.config.volatility_scale, n)
		vol_matrix = np.outer(volatilities, volatilities)
		cov_matrix = combined * vol_matrix

		return self._ensure_positive_definite(cov_matrix)

	def generate_block_structured_covariance(self) -> np.ndarray:
		"""Generate block-structured covariance (sector-based)"""
		n = self.config.n_assets
		n_blocks = np.random.randint(5, 15)  # 5-15 sectors

		# Create block structure
		block_sizes = self._random_block_sizes(n, n_blocks)
		corr_matrix = np.zeros((n, n))

		start_idx = 0
		for block_size in block_sizes:
			end_idx = start_idx + block_size

			# Intra-block correlation (high)
			intra_corr = np.random.uniform(0.3, 0.8)
			block = np.full((block_size, block_size), intra_corr)
			np.fill_diagonal(block, 1.0)

			corr_matrix[start_idx:end_idx, start_idx:end_idx] = block
			start_idx = end_idx

		# Inter-block correlations (low)
		inter_corr = np.random.uniform(0.05, 0.25)
		mask = corr_matrix == 0
		corr_matrix[mask] = np.random.uniform(-inter_corr, inter_corr, np.sum(mask))

		# Ensure symmetry
		corr_matrix = (corr_matrix + corr_matrix.T) / 2
		np.fill_diagonal(corr_matrix, 1.0)

		# Convert to covariance
		volatilities = self._generate_sector_volatilities(block_sizes)
		vol_matrix = np.outer(volatilities, volatilities)
		cov_matrix = corr_matrix * vol_matrix

		return self._ensure_positive_definite(cov_matrix)

	def generate_real_data_based_covariance(self) -> np.ndarray:
		"""Generate covariance based on real market data patterns"""
		# Look for CSV files with market data
		csv_files = self._find_market_data_files()

		if csv_files:
			# Use real data as base
			try:
				data = pd.read_csv(csv_files[0], index_col=0)
				if len(data.columns) >= MIN_ASSETS_FOR_BOOTSTRAP:  # Need some assets
					return self._bootstrap_from_real_data(data)
			except Exception:
				pass

		# Fallback: simulate realistic market patterns
		return self._generate_market_like_covariance()

	def _generate_random_correlation(self) -> np.ndarray:
		"""Generate random correlation matrix"""
		# Use Wishart distribution approach
		n = self.config.n_assets
		A = np.random.randn(n, n + OVERDETERMINED_EXTRA_COLUMNS)  # Overdetermined for stability
		C = A @ A.T

		# Convert to correlation
		D = np.sqrt(np.diag(C))
		corr = C / np.outer(D, D)

		# Ensure symmetry and real values
		corr = (corr + corr.T) / 2
		corr = np.real(corr)
		np.fill_diagonal(corr, 1.0)

		return self._ensure_positive_definite(corr)

	def _random_block_sizes(self, n_assets: int, n_blocks: int) -> list[int]:
		"""Generate random block sizes that sum to n_assets"""
		# Generate random splits
		splits = sorted(np.random.choice(n_assets - 1, n_blocks - 1, replace=False))
		splits = [0] + list(splits) + [n_assets]

		# Calculate block sizes
		block_sizes = [splits[i + 1] - splits[i] for i in range(n_blocks)]
		return [max(1, size) for size in block_sizes if size > 0]  # Ensure positive sizes

	def _generate_sector_volatilities(self, block_sizes: list[int]) -> np.ndarray:
		"""Generate realistic sector-based volatilities"""
		volatilities = []

		for block_size in block_sizes:
			# Each sector has base volatility + individual variations
			sector_vol = np.random.uniform(*self.config.volatility_scale)
			individual_vols = sector_vol * np.random.uniform(0.8, 1.2, block_size)
			volatilities.extend(individual_vols)

		return np.array(volatilities)

	def _find_market_data_files(self) -> list[str]:
		"""Find CSV files that might contain market data"""
		base_path = Path(__file__).parent.parent.parent
		csv_files = []

		# Look in common locations
		search_paths = [
			base_path,
			base_path / "data",
			base_path / "generated_output",
			base_path / "experimental" / "data",
		]

		for path in search_paths:
			if path.exists():
				csv_files.extend(list(path.glob("*.csv")))

		# Filter for likely market data files
		market_files = []
		for file in csv_files:
			if any(
				keyword in file.name.lower() for keyword in ["price", "stock", "return", "market", "equity", "trade"]
			):
				market_files.append(str(file))

		return market_files[:5]  # Limit to first 5 files

	def _bootstrap_from_real_data(self, data: pd.DataFrame) -> np.ndarray:
		"""Bootstrap covariance matrix from real market data"""
		# Calculate returns
		returns = data.pct_change().dropna() if len(data) > 1 else data

		# Resample for bootstrap
		n_samples = min(len(returns), np.random.randint(50, 200))
		bootstrap_idx = np.random.choice(len(returns), n_samples, replace=True)
		bootstrap_returns = returns.iloc[bootstrap_idx]

		# Calculate covariance
		cov_matrix = bootstrap_returns.cov().values

		# Resize to target dimensions
		if cov_matrix.shape[0] != self.config.n_assets:
			cov_matrix = self._resize_covariance_matrix(cov_matrix, self.config.n_assets)

		return self._ensure_positive_definite(cov_matrix)

	def _generate_market_like_covariance(self) -> np.ndarray:
		"""Generate covariance matrix with realistic market properties"""
		# Use mixed regime eigenvalues
		eigenvals = SpectrumGenerator.mixed_regime(self.config.n_assets, np.random.randint(5, 15))

		# Generate correlation matrix
		Q = ortho_group.rvs(self.config.n_assets)
		corr_matrix = Q @ np.diag(eigenvals) @ Q.T

		# Add market-like volatility clustering
		volatilities = self._generate_market_volatilities()
		vol_matrix = np.outer(volatilities, volatilities)
		cov_matrix = corr_matrix * vol_matrix

		return self._ensure_positive_definite(cov_matrix)

	def _generate_market_volatilities(self) -> np.ndarray:
		"""Generate realistic market volatility distribution"""
		# Most assets have moderate volatility, few have high volatility
		n = self.config.n_assets

		# 70% moderate volatility
		moderate = np.random.uniform(0.15, 0.35, int(0.7 * n))
		# 20% low volatility
		low = np.random.uniform(0.05, 0.15, int(0.2 * n))
		# 10% high volatility
		high = np.random.uniform(0.35, 0.60, n - len(moderate) - len(low))

		volatilities = np.concatenate([moderate, low, high])
		np.random.shuffle(volatilities)

		# Scale to annual terms
		return volatilities / np.sqrt(252)  # Daily to annual conversion

	def _resize_covariance_matrix(self, cov_matrix: np.ndarray, target_size: int) -> np.ndarray:
		"""Resize covariance matrix to target dimensions"""
		current_size = cov_matrix.shape[0]

		if current_size == target_size:
			return cov_matrix
		elif current_size > target_size:
			# Subsample
			idx = np.random.choice(current_size, target_size, replace=False)
			return cov_matrix[np.ix_(idx, idx)]
		else:
			# Expand by replication and noise
			factor = target_size // current_size
			remainder = target_size % current_size

			# Replicate blocks
			expanded = np.kron(np.ones((factor, factor)), cov_matrix)

			# Add remainder
			if remainder > 0:
				extra_rows = cov_matrix[:remainder, :]
				extra_cols = cov_matrix[:, :remainder]
				extra_corner = cov_matrix[:remainder, :remainder]

				# Combine
				top = np.hstack([expanded, np.kron(np.ones((factor * current_size, 1)), extra_cols)])
				bottom_left = np.kron(np.ones((1, factor)), extra_rows)
				bottom_right = extra_corner
				bottom = np.hstack([bottom_left, bottom_right])

				expanded = np.vstack([top, bottom])

			return expanded

	def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
		"""Ensure matrix is positive definite"""
		# Symmetrize
		matrix = (matrix + matrix.T) / 2

		# Get eigenvalues (use eigh for symmetric matrices to ensure real eigenvalues)
		eigenvals, eigenvecs = np.linalg.eigh(matrix)

		# Ensure all eigenvalues are positive
		min_eigval = max(1e-8, np.abs(eigenvals[0]) * 1e-6)
		eigenvals = np.maximum(eigenvals, min_eigval)

		# Reconstruct matrix (ensure real result)
		matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
		matrix = np.real(matrix)  # Ensure real result

		return matrix

	def add_sampling_noise(self, cov_matrix: np.ndarray) -> np.ndarray:
		"""Add realistic sampling noise to covariance matrix"""
		n = cov_matrix.shape[0]

		# Generate noise matrix
		noise = np.random.randn(n, n) * self.config.noise_level
		noise = (noise + noise.T) / 2  # Symmetrize

		# Scale noise relative to matrix magnitude
		matrix_scale = np.mean(np.abs(cov_matrix))
		noise = noise * matrix_scale

		# Add noise
		noisy_matrix = cov_matrix + noise

		return self._ensure_positive_definite(noisy_matrix)

	def generate_diverse_training_set(self) -> list[np.ndarray]:
		"""
		Generate diverse training set of 30,000 covariance matrices
		Using multiple generation methods for maximum diversity
		"""
		print(
			f"Generating {self.config.n_samples} diverse covariance matrices "
			f"({self.config.n_assets}×{self.config.n_assets})..."
		)

		training_data = []

		# Calculate sample counts for each method
		n_synthetic = int(self.config.pct_synthetic * self.config.n_samples)
		n_gan_style = int(self.config.pct_gan_style * self.config.n_samples)
		n_real_based = int(self.config.pct_real_based * self.config.n_samples)
		n_block_struct = self.config.n_samples - n_synthetic - n_gan_style - n_real_based

		print(f"  - {n_synthetic} synthetic eigenvalue-based matrices ({self.config.pct_synthetic:.0%})")
		print(f"  - {n_gan_style} GAN-style diverse matrices ({self.config.pct_gan_style:.0%})")
		print(f"  - {n_real_based} real data-based matrices ({self.config.pct_real_based:.0%})")
		print(f"  - {n_block_struct} block-structured matrices ({self.config.pct_block_struct:.0%})")

		# Generate synthetic matrices
		print("\nGenerating synthetic matrices...")
		for i in range(n_synthetic):
			if i % 1000 == 0:
				print(f"  Progress: {i}/{n_synthetic}")
			cov_matrix = self.generate_synthetic_covariance()
			training_data.append(self.add_sampling_noise(cov_matrix))

		# Generate GAN-style matrices
		print("\nGenerating GAN-style matrices...")
		for i in range(n_gan_style):
			if i % 1000 == 0:
				print(f"  Progress: {i}/{n_gan_style}")
			cov_matrix = self.generate_gan_style_covariance()
			training_data.append(self.add_sampling_noise(cov_matrix))

		# Generate real data-based matrices
		print("\nGenerating real data-based matrices...")
		for i in range(n_real_based):
			if i % 1000 == 0:
				print(f"  Progress: {i}/{n_real_based}")
			cov_matrix = self.generate_real_data_based_covariance()
			training_data.append(self.add_sampling_noise(cov_matrix))

		# Generate block-structured matrices
		print("\nGenerating block-structured matrices...")
		for i in range(n_block_struct):
			if i % 1000 == 0:
				print(f"  Progress: {i}/{n_block_struct}")
			cov_matrix = self.generate_block_structured_covariance()
			training_data.append(self.add_sampling_noise(cov_matrix))

		print(f"\nGenerated {len(training_data)} covariance matrices successfully!")

		return training_data


def main():
	"""Demo function to generate diverse covariance training set"""
	config = CovarianceConfig(n_assets=500, n_samples=30000, random_seed=42)

	generator = CovarianceMatrixGenerator(config)
	training_data = generator.generate_diverse_training_set()

	# Validate the data
	print("\nDataset validation:")
	print(f"  Total matrices: {len(training_data)}")
	print(f"  Matrix shape: {training_data[0].shape}")

	# Check properties of first few matrices
	for i in range(min(3, len(training_data))):
		matrix = training_data[i]
		eigenvals = np.linalg.eigvals(matrix)

		print(f"  Matrix {i+1}:")
		print(f"    Positive definite: {np.all(eigenvals > 0)}")
		print(f"    Symmetric: {np.allclose(matrix, matrix.T)}")
		print(f"    Condition number: {np.linalg.cond(matrix):.2e}")
		print(f"    Trace: {np.trace(matrix):.4f}")

	return training_data


if __name__ == "__main__":
	training_data = main()
