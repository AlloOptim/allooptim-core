#!/usr/bin/env python3
"""Improved Autoencoder with Lower Triangle Optimization and Reconstruction Metrics.

Educational implementation using synthetic training data generation.
"""

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from tinygrad import Tensor, dtypes, nn
from tinygrad.nn import optim

from allooptim.covariance_transformer.covariance_transformer import (
    _create_cov_dataframe,
    _extract_cov_info,
)
from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)
from allooptim.data_generation.lower_triangle_utils import get_packed_size, pack_lower_triangle, unpack_lower_triangle
from allooptim.data_generation.training_data import TrainingConfig, TrainingDataGenerator, load_training_data

logger = logging.getLogger(__name__)

# Constants for autoencoder thresholds
EXTREMELY_LOW_SAMPLES_PER_PARAM_THRESHOLD = 0.001
MATRIX_DIMENSION_CHECK = 2
CRITICAL_SAMPLES_PER_PARAM_THRESHOLD = 0.01
CONSERVATIVE_SAMPLES_PER_PARAM_THRESHOLD = 0.1


class AutoencoderCovarianceTransformer(AbstractCovarianceTransformer):
    """Autoencoder with symmetric matrix optimization and reconstruction metrics."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_dims: Optional[list] = None,
        learning_rate: float = 0.001,
        epochs: int = 100,  # Reduced from 200 due to overfitting risk
        batch_size: int = 8,  # Smaller batches for limited data
        window_size: int = 63,  # 3 months for maximum samples
        validation_split: float = 0.2,
        patience: int = 10,  # Aggressive early stopping
        min_delta: float = 1e-4,
        dropout_rate: float = 0.5,  # Aggressive dropout
        l2_lambda: float = 1e-3,  # Strong L2 regularization
        gradient_clip_value: float = 1.0,  # Gradient clipping
        training_data_overlapping: float = 0.5,  # 50% overlap for max samples
        use_lower_triangle: bool = True,  # Enable symmetric optimization
        use_synthetic: bool = True,
        n_synthetic_samples: int = 50000,
        n_assets: Optional[int] = None,  # Number of assets (inferred from data if None)
    ):
        """Initialize improved autoencoder with aggressive regularization.

        Args:
            hidden_dims: Architecture layers (None for auto-sizing)
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            window_size: Rolling window size for covariance estimation
            validation_split: Fraction of data for validation
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            dropout_rate: Dropout rate for regularization
            l2_lambda: L2 regularization strength
            gradient_clip_value: Gradient clipping threshold
            training_data_overlapping: Overlap fraction for training windows
            use_lower_triangle: Use symmetric matrix optimization (recommended)
            use_synthetic: Whether to use synthetic training data
            n_synthetic_samples: Number of synthetic samples to generate
            n_assets: Number of assets (inferred from data if None)
        """
        self.use_lower_triangle = use_lower_triangle
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.validation_split = validation_split
        self.patience = patience
        self.min_delta = min_delta
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.gradient_clip_value = gradient_clip_value
        self.training_data_overlapping = training_data_overlapping
        self.use_synthetic = use_synthetic
        self.n_synthetic_samples = n_synthetic_samples
        self.n_assets = n_assets

        # Will be set during fit
        self.input_size: int
        self.total_params: int
        self.model: list
        self.optimizer: Any

        # Training state
        self.is_fitted = False
        self.reconstruction_metrics: dict = {}
        self.training_history: dict = {}

    def _build_autoencoder(self):
        """Build ultra-minimal autoencoder with aggressive regularization."""
        layers = []

        # Encoder
        input_dim = self.input_size
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm(hidden_dim))
            input_dim = hidden_dim

        # Decoder (reverse of encoder)
        for i in reversed(range(len(self.hidden_dims))):
            if i == 0:
                # Output layer
                layers.append(nn.Linear(self.hidden_dims[i], self.input_size))
            else:
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1]))
                layers.append(nn.BatchNorm(self.hidden_dims[i - 1]))

        return layers

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        assert self.hidden_dims is not None, "hidden_dims must be set before calling _count_parameters"  # nosec B101 - Assert for internal consistency checks
        total = 0

        # Encoder
        input_dim = self.input_size
        for hidden_dim in self.hidden_dims:
            # Linear layer: input * hidden + bias
            total += input_dim * hidden_dim + hidden_dim
            # BatchNorm: 2 * hidden (weight + bias)
            total += 2 * hidden_dim
            input_dim = hidden_dim

        # Decoder
        for i in reversed(range(len(self.hidden_dims))):
            if i == 0:
                # Output layer
                total += self.hidden_dims[i] * self.input_size + self.input_size
            else:
                # Hidden layer
                total += self.hidden_dims[i] * self.hidden_dims[i - 1] + self.hidden_dims[i - 1]
                # BatchNorm
                total += 2 * self.hidden_dims[i - 1]

        return total

    def _generate_synthetic_training_data(
        self, n_samples: int = 50000, use_cached: bool = True, cache_file: str = "covariance_training_data.h5"
    ) -> tuple[Tensor, Tensor]:
        """Generate synthetic training data using the training_data module.

        Args:
            n_samples: Number of synthetic samples to generate
            use_cached: Whether to use cached data if available
            cache_file: Cache file path

        Returns:
            tuple of (noisy_data, clean_data) tensors
        """
        assert self.n_assets is not None, "n_assets must be set before generating training data"  # nosec B101 - Assert for internal consistency checks
        # Check for cached data
        if use_cached and os.path.exists(cache_file):
            logger.debug(f"📁 Loading cached training data from {cache_file}")
            try:
                data, metadata = load_training_data(cache_file)

                # Verify dimensions match
                if metadata["n_assets"] == self.n_assets:
                    sample_eigenvals = data["sample_eigenvalues"]
                    true_eigenvals = data["true_eigenvalues"]

                    # Convert eigenvalues to full covariance matrices
                    noisy_matrices = []
                    clean_matrices = []

                    logger.debug(f"   Converting {len(sample_eigenvals)} eigenvalue pairs to covariance matrices...")

                    for i in range(min(n_samples, len(sample_eigenvals))):
                        # Generate random orthogonal matrix for reconstruction
                        np.random.seed(42 + i)  # Reproducible
                        Q = np.random.randn(self.n_assets, self.n_assets)
                        Q, _ = np.linalg.qr(Q)

                        # Construct matrices from eigenvalues
                        sample_cov = Q @ np.diag(sample_eigenvals[i]) @ Q.T
                        true_cov = Q @ np.diag(true_eigenvals[i]) @ Q.T

                        # Apply lower triangle packing if enabled
                        if self.use_lower_triangle:
                            noisy_matrices.append(pack_lower_triangle(sample_cov))
                            clean_matrices.append(pack_lower_triangle(true_cov))
                        else:
                            noisy_matrices.append(sample_cov.flatten())
                            clean_matrices.append(true_cov.flatten())

                    logger.debug(f"   ✅ Loaded {len(noisy_matrices)} training samples from cache")
                    return (
                        Tensor(np.array(noisy_matrices), dtype=dtypes.float32),
                        Tensor(np.array(clean_matrices), dtype=dtypes.float32),
                    )
                else:
                    logger.debug(f"   ⚠️ Cached data is for {metadata['n_assets']} assets, need {self.n_assets}")
            except Exception as e:
                logger.debug(f"   ⚠️ Error loading cached data: {e}")

        # Generate new synthetic data
        logger.debug(f"🎲 Generating {n_samples} synthetic covariance matrix pairs...")

        config = TrainingConfig(
            n_assets=self.n_assets,
            n_samples=n_samples,
            min_observations=60,  # 3 months minimum
            max_observations=1000,  # ~4 years maximum
            output_file=cache_file,
            random_seed=42,
        )

        generator = TrainingDataGenerator(config)
        samples = generator.generate_parallel(verbose=True)

        # Save to cache for future use
        generator.save_to_hdf5(samples, cache_file)

        # Convert to matrices and pack
        noisy_matrices = []
        clean_matrices = []

        logger.debug("   Converting eigenvalue pairs to covariance matrices...")

        for i, sample in enumerate(samples):
            # Generate random orthogonal matrix for each sample
            np.random.seed(42 + i)
            Q = np.random.randn(self.n_assets, self.n_assets)
            Q, _ = np.linalg.qr(Q)

            # Construct matrices from eigenvalues
            sample_cov = Q @ np.diag(sample["sample_eigenvalues"]) @ Q.T
            true_cov = Q @ np.diag(sample["true_eigenvalues"]) @ Q.T

            # Apply lower triangle packing if enabled
            if self.use_lower_triangle:
                noisy_matrices.append(pack_lower_triangle(sample_cov))
                clean_matrices.append(pack_lower_triangle(true_cov))
            else:
                noisy_matrices.append(sample_cov.flatten())
                clean_matrices.append(true_cov.flatten())

        logger.debug(f"   ✅ Generated {len(noisy_matrices)} synthetic training pairs")

        return (
            Tensor(np.array(noisy_matrices), dtype=dtypes.float32),
            Tensor(np.array(clean_matrices), dtype=dtypes.float32),
        )

    def _generate_training_data_legacy(self, historical_prices: pd.DataFrame) -> Tensor:
        """Generate training data from historical prices (legacy method for comparison)."""
        returns = historical_prices.pct_change().dropna()
        n_periods = len(returns)

        logger.debug(f"📈 Generating legacy training data from {n_periods} periods ({n_periods/252:.1f} years)")

        cov_matrices = []
        step_size = max(1, int(self.window_size * self.training_data_overlapping))

        logger.debug(f"   Window size: {self.window_size} days")
        logger.debug(f"   Overlap: {self.training_data_overlapping:.0%} (step size: {step_size})")

        for i in range(self.window_size, n_periods, step_size):
            window_returns = returns.iloc[i - self.window_size : i]
            if len(window_returns) >= self.window_size // 2:
                cov = window_returns.cov().values

                # Apply lower triangle optimization if enabled
                cov_flat = pack_lower_triangle(cov) if self.use_lower_triangle else cov.flatten()

                cov_matrices.append(cov_flat)

        if len(cov_matrices) == 0:
            raise ValueError(
                f"Not enough data to generate training covariance matrices. "
                f"Need at least {self.window_size} periods, got {n_periods}"
            )

        samples_per_param = len(cov_matrices) / self.total_params
        logger.debug(f"   Generated samples: {len(cov_matrices)}")
        logger.debug(f"   Samples/parameter: {samples_per_param:.6f}")

        if samples_per_param < EXTREMELY_LOW_SAMPLES_PER_PARAM_THRESHOLD:
            logger.debug("   ⚠️ WARNING: Extremely low samples/parameter ratio!")
            logger.debug("   ⚠️ Expected severe overfitting and poor generalization!")

        return Tensor(np.array(cov_matrices), dtype=dtypes.float32)

    def _apply_gradient_clipping(self):
        """Apply gradient clipping to prevent exploding gradients."""
        total_norm = 0.0
        for param in self.optimizer.params:
            if param.grad is not None:
                param_norm = param.grad.numpy().flatten()
                total_norm += np.sum(param_norm**2)

        total_norm = np.sqrt(total_norm)

        if total_norm > self.gradient_clip_value:
            clip_coef = self.gradient_clip_value / (total_norm + 1e-6)
            for param in self.optimizer.params:
                if param.grad is not None:
                    param.grad = param.grad * clip_coef

    def _forward_pass(self, x, training=True):
        """Forward pass with dropout control."""
        for i in range(0, len(self.model), 2):
            # Linear layer
            x = self.model[i](x)

            # BatchNorm (if not last layer)
            if i + 1 < len(self.model):
                x = self.model[i + 1](x)
                # ReLU activation
                x = x.relu()
                # Dropout during training (except last layer)
                if training and i + 2 < len(self.model):
                    x = x.dropout(self.dropout_rate)
        return x

    def _initial_fit(self, n_assets: int) -> None:
        """Initial setup before training."""
        self.n_assets = n_assets

        # Calculate input/output dimensions
        if self.use_lower_triangle:
            self.input_size = get_packed_size(n_assets)  # n(n+1)/2
            logger.debug(
                f"🔧 Lower triangle optimization: {n_assets}x{n_assets} → {self.input_size} ({50.1:.1f}% reduction)"
            )
        else:
            self.input_size = n_assets * n_assets
            logger.debug(f"⚠️ Using full matrix: {self.input_size} elements")

        # Default architecture - extremely conservative for limited data
        if self.hidden_dims is None:
            if self.use_lower_triangle:
                # Ultra-minimal architecture for 125K input
                self.hidden_dims = [64, 32, 16]  # Extreme compression
                logger.debug(
                    f"🏗️ Ultra-minimal architecture: {self.input_size} → {self.hidden_dims} → {self.input_size}"
                )
            else:
                self.hidden_dims = [128, 64, 32]  # Slightly larger for full matrix

        # Build model
        self.model = self._build_autoencoder()

        # Collect parameters
        all_params = []
        for layer in self.model:
            if hasattr(layer, "weight"):
                all_params.append(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                all_params.append(layer.bias)
            # BatchNorm parameters
            if hasattr(layer, "running_mean"):
                all_params.append(layer.running_mean)
            if hasattr(layer, "running_var"):
                all_params.append(layer.running_var)

        self.optimizer = optim.AdamW(
            all_params,
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )

        # Calculate total parameters
        self.total_params = self._count_parameters()

        logger.debug("📊 Model statistics:")
        logger.debug(f"   Total parameters: {self.total_params:,}")
        logger.debug(f"   Minimum samples needed (1:100): {self.total_params // 100:,}")
        logger.debug(f"   Conservative samples needed (1:10): {self.total_params // 10:,}")

    def fit(
        self,
        df_prices: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train the autoencoder using synthetic or historical data.

        Args:
            df_prices: Historical price data (optional if use_synthetic=True or n_assets provided)
        """
        logger.debug("\n🚀 Training improved autoencoder with enhanced data generation...")

        if not self.is_fitted:
            if self.n_assets is None:
                if df_prices is None:
                    raise ValueError("n_assets must be provided if df_prices is None")
                n_assets = df_prices.shape[1]
            else:
                n_assets = self.n_assets
                if df_prices is not None and df_prices.shape[1] != n_assets:
                    raise ValueError(f"n_assets {n_assets} does not match df_prices shape {df_prices.shape[1]}")
            self._initial_fit(n_assets)

        if self.use_synthetic:
            logger.debug("🎲 Using synthetic covariance matrix training data")
            # Generate synthetic training data (denoising task)
            X_noisy, X_clean = self._generate_synthetic_training_data(
                n_samples=self.n_synthetic_samples, use_cached=True
            )

            # Use noisy samples as input, clean samples as targets
            X_all = X_noisy
            Y_all = X_clean  # Target clean matrices
            denoising_task = True
        else:
            if df_prices is None:
                raise ValueError("historical_prices required when use_synthetic=False")

            logger.debug("📈 Using historical price data (legacy method)")
            X_all = self._generate_training_data_legacy(df_prices)
            Y_all = X_all  # Autoencoder task (reconstruct input)
            denoising_task = False

        # Split data
        n_val = max(1, int(len(X_all) * self.validation_split))
        X_train = X_all[:-n_val] if n_val > 0 else X_all
        X_val = X_all[-n_val:] if n_val > 0 else X_all[:1]

        if denoising_task:
            Y_train = Y_all[:-n_val] if n_val > 0 else Y_all
            Y_val = Y_all[-n_val:] if n_val > 0 else Y_all[:1]
        else:
            Y_train = X_train
            Y_val = X_val

        logger.debug("\n📚 Training setup:")
        logger.debug(f"   Training samples: {len(X_train)}")
        logger.debug(f"   Validation samples: {len(X_val)}")
        logger.debug(f"   Task type: {'Denoising' if denoising_task else 'Autoencoder'}")
        logger.debug(f"   Batch size: {self.batch_size}")
        logger.debug(f"   Dropout rate: {self.dropout_rate}")
        logger.debug(f"   L2 regularization: {self.l2_lambda}")

        # Training tracking
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        # Simple batching
        def get_batches(data_x, data_y, batch_size):
            """Generate batches of training data for mini-batch training.

            Args:
                data_x: Input data tensor
                data_y: Target data tensor
                batch_size: Size of each batch

            Yields:
                tuple: (batch_x, batch_y) tensors for each batch
            """
            for i in range(0, len(data_x), batch_size):
                yield data_x[i : i + batch_size], data_y[i : i + batch_size]

        # Training loop
        with Tensor.train():
            for epoch in range(self.epochs):
                # Training phase
                train_loss = 0.0
                train_batches = list(get_batches(X_train, Y_train, self.batch_size))

                for batch_X, batch_Y in train_batches:
                    # Forward pass
                    x_reconstructed = self._forward_pass(batch_X, training=True)

                    # Compute loss (MSE between reconstruction and target)
                    loss = ((x_reconstructed - batch_Y) ** 2).mean()

                    # Backward pass with gradient clipping
                    self.optimizer.zero_grad()
                    loss.backward()
                    self._apply_gradient_clipping()
                    self.optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_batches)

                # Validation phase
                val_loss = 0.0
                val_batches = list(get_batches(X_val, Y_val, self.batch_size))

                for batch_X, batch_Y in val_batches:
                    x_reconstructed = self._forward_pass(batch_X, training=False)
                    loss = ((x_reconstructed - batch_Y) ** 2).mean()
                    val_loss += loss.item()

                val_loss /= len(val_batches)

                # Track losses
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Print progress
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"   Epoch {epoch+1}/{self.epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")

                if patience_counter >= self.patience:
                    logger.debug(f"   Early stopping at epoch {epoch+1}")
                    break

        # Store training history
        self.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "epochs_trained": len(train_losses),
        }

        self.is_fitted = True
        logger.debug("✅ Training completed!")
        logger.debug(f"   Final train loss: {train_losses[-1]:.6f}")
        logger.debug(f"   Final val loss: {val_losses[-1]:.6f}")
        logger.debug(f"   Epochs trained: {len(train_losses)}")

    def _calculate_reconstruction_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """Calculate comprehensive reconstruction error metrics."""
        # Ensure same shape
        if original.shape != reconstructed.shape:
            raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")

        # Frobenius norm error
        frobenius_error = np.linalg.norm(original - reconstructed, "fro")
        frobenius_norm_original = np.linalg.norm(original, "fro")
        relative_frobenius_error = frobenius_error / (frobenius_norm_original + 1e-8)

        # Element-wise relative error
        relative_error = np.mean(np.abs(original - reconstructed) / (np.abs(original) + 1e-8))

        # Correlation coefficient
        correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]

        # Matrix-specific metrics for covariance matrices
        if len(original.shape) == MATRIX_DIMENSION_CHECK and original.shape[0] == original.shape[1]:
            # Eigenvalue preservation
            orig_eigs = np.linalg.eigvals(original)
            recon_eigs = np.linalg.eigvals(reconstructed)

            # Sort eigenvalues for comparison
            orig_eigs_sorted = np.sort(orig_eigs)[::-1]
            recon_eigs_sorted = np.sort(recon_eigs)[::-1]

            eigenvalue_error = np.mean(np.abs(orig_eigs_sorted - recon_eigs_sorted) / (np.abs(orig_eigs_sorted) + 1e-8))

            # Condition number preservation
            orig_cond = np.linalg.cond(original)
            recon_cond = np.linalg.cond(reconstructed)
            condition_error = np.abs(orig_cond - recon_cond) / (orig_cond + 1e-8)
        else:
            eigenvalue_error = np.nan
            condition_error = np.nan

        return {
            "frobenius_error": frobenius_error,
            "relative_frobenius_error": relative_frobenius_error,
            "relative_error": relative_error,
            "correlation": correlation,
            "eigenvalue_preservation_error": eigenvalue_error,
            "condition_number_error": condition_error,
        }

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Transform covariance matrix with reconstruction metrics."""
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before transforming")

        assert self.n_assets is not None, "Autoencoder must be fitted before transforming"  # nosec B101 - Assert for internal consistency checks
        cov_array, asset_names = _extract_cov_info(df_cov)

        # Prepare input
        cov_input = pack_lower_triangle(cov_array) if self.use_lower_triangle else cov_array.flatten()

        cov_tensor = Tensor(cov_input.astype(np.float32)).unsqueeze(0)

        # Forward pass (no dropout during inference)
        denoised_tensor = self._forward_pass(cov_tensor, training=False)
        denoised_flat = denoised_tensor.squeeze(0).numpy()

        # Reconstruct matrix
        if self.use_lower_triangle:
            denoised_cov = unpack_lower_triangle(denoised_flat, self.n_assets)
        else:
            denoised_cov = denoised_flat.reshape((self.n_assets, self.n_assets))

        # Ensure positive semi-definite
        eigenvals = np.linalg.eigvals(denoised_cov)
        if np.any(eigenvals < 0):
            min_eigenval = np.min(eigenvals)
            denoised_cov += np.eye(self.n_assets) * (-min_eigenval + 1e-8)

        # Calculate reconstruction metrics
        self.reconstruction_metrics = self._calculate_reconstruction_metrics(cov_array, denoised_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(denoised_cov, asset_names)

    def get_reconstruction_metrics(self) -> dict:
        """Get the latest reconstruction error metrics."""
        return self.reconstruction_metrics.copy() if self.reconstruction_metrics else {}

    def print_analysis_summary(self):
        """Print comprehensive analysis of the autoencoder's viability."""
        logger.debug("\n" + "=" * 70)
        logger.debug("AUTOENCODER ANALYSIS SUMMARY")
        logger.debug("=" * 70)

        logger.debug("🏗️ Architecture:")
        logger.debug(f"   Input dimension: {self.input_size:,}")
        logger.debug(f"   Hidden layers: {self.hidden_dims}")
        logger.debug(f"   Total parameters: {self.total_params:,}")
        logger.debug(f"   Lower triangle optimization: {'✅ Enabled' if self.use_lower_triangle else '❌ Disabled'}")

        if hasattr(self, "training_history"):
            samples = len(self.training_history.get("train_losses", [0]))
            if samples > 0:
                samples_per_param = samples / self.total_params
                logger.debug("\n📊 Training Data:")
                logger.debug(f"   Training samples: {samples}")
                logger.debug(f"   Samples/parameter: {samples_per_param:.6f}")
                logger.debug(f"   Minimum needed (1:100): {self.total_params // 100:,}")
                logger.debug(f"   Conservative needed (1:10): {self.total_params // 10:,}")

                if samples_per_param < CRITICAL_SAMPLES_PER_PARAM_THRESHOLD:
                    logger.debug(f"   ❌ CRITICAL: {(0.01 / samples_per_param):.0f}x below minimum threshold!")
                elif samples_per_param < CONSERVATIVE_SAMPLES_PER_PARAM_THRESHOLD:
                    logger.debug(f"   ⚠️ WARNING: {(0.1 / samples_per_param):.0f}x below conservative threshold")
                else:
                    logger.debug("   ✅ Adequate sample ratio")

        if self.reconstruction_metrics:
            logger.debug("\n🎯 Reconstruction Quality:")
            metrics = self.reconstruction_metrics
            logger.debug(f"   Correlation: {metrics.get('correlation', 0):.4f}")
            logger.debug(f"   Relative error: {metrics.get('relative_error', 0):.4f}")
            logger.debug(f"   Frobenius error: {metrics.get('relative_frobenius_error', 0):.4f}")

        logger.debug("\n💡 Recommendation:")
        logger.debug("   For production: Use OracleCovarianceTransformer or SimpleShrinkage")
        logger.debug("   For research: This autoencoder demonstrates concepts but lacks viable data")
        logger.debug("   Mathematical reality: Need 1,000x-10,000x more training samples")
