#!/usr/bin/env python3
"""
Demonstration of Synthetic Training Data Integration with Improved Autoencoder
Shows how the training_data module provides realistic covariance matrix training samples.
"""

import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from improved_autoencoder import ImprovedAutoencoderCovarianceTransformer
from lower_triangle_utils import pack_lower_triangle, unpack_lower_triangle, validate_symmetric
from training_data import TrainingConfig, TrainingDataGenerator, load_training_data


def generate_demo_training_data(n_assets: int = 100, n_samples: int = 5000) -> Dict:
    """Generate demonstration training data for smaller scale testing."""
    print(f"🎲 Generating demo training data ({n_assets} assets, {n_samples} samples)...")

    config = TrainingConfig(
        n_assets=n_assets,
        n_samples=n_samples,
        min_observations=50,
        max_observations=500,
        output_file=f"demo_training_data_{n_assets}x{n_samples}.h5",
        random_seed=42,
    )

    generator = TrainingDataGenerator(config)

    # Time the generation
    start_time = time.time()
    samples = generator.generate_parallel(verbose=True)
    generation_time = time.time() - start_time

    # Save to cache
    generator.save_to_hdf5(samples)

    # Load back for validation
    data, metadata = load_training_data(config.output_file)

    print(f"✅ Generation completed in {generation_time:.2f} seconds")
    print(f"📊 Data statistics:")
    print(f"   Q-ratio range: [{data['q_values'].min():.3f}, {data['q_values'].max():.3f}]")
    print(f"   Observations range: [{data['n_observations'].min()}, {data['n_observations'].max()}]")

    return data, metadata


def demonstrate_eigenvalue_diversity(data: Dict) -> None:
    """Show the diversity of eigenvalue spectra in the training data."""
    print(f"\n📈 Analyzing eigenvalue spectrum diversity...")

    sample_eigenvals = data["sample_eigenvalues"]
    true_eigenvals = data["true_eigenvalues"]

    # Sample a few examples for visualization
    n_examples = 5
    indices = np.linspace(0, len(sample_eigenvals) - 1, n_examples, dtype=int)

    plt.figure(figsize=(15, 8))

    # Plot eigenvalue spectra
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i + 1)

        true_eigs = np.sort(true_eigenvals[idx])[::-1]  # Descending order
        sample_eigs = np.sort(sample_eigenvals[idx])[::-1]

        plt.plot(true_eigs, "b-", label="True", alpha=0.8, linewidth=2)
        plt.plot(sample_eigs, "r--", label="Noisy", alpha=0.8, linewidth=2)

        plt.title(f'Spectrum {i+1}\nQ={data["q_values"][idx]:.3f}')
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Log scale for better visualization
        plt.yscale("log")

    # Overall statistics
    plt.subplot(2, 3, 6)

    # Distribution of largest eigenvalue
    max_true = np.max(true_eigenvals, axis=1)
    max_sample = np.max(sample_eigenvals, axis=1)

    plt.hist(max_true, bins=50, alpha=0.6, label="True Max Eigenval", color="blue")
    plt.hist(max_sample, bins=50, alpha=0.6, label="Noisy Max Eigenval", color="red")
    plt.xlabel("Max Eigenvalue")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Eigenvalues")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eigenvalue_diversity_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"   True eigenvalue range: [{true_eigenvals.min():.3f}, {true_eigenvals.max():.3f}]")
    print(f"   Sample eigenvalue range: [{sample_eigenvals.min():.3f}, {sample_eigenvals.max():.3f}]")


def demonstrate_lower_triangle_optimization(n_assets: int = 100) -> None:
    """Demonstrate the lower triangle optimization efficiency."""
    print(f"\n🔧 Demonstrating lower triangle optimization for {n_assets}x{n_assets} matrices...")

    # Generate a random covariance matrix
    np.random.seed(42)
    A = np.random.randn(n_assets, n_assets)
    cov_matrix = A @ A.T  # Ensure positive semi-definite

    # Add identity for numerical stability
    cov_matrix += 0.1 * np.eye(n_assets)

    print(f"   Original matrix shape: {cov_matrix.shape}")
    print(f"   Original elements: {cov_matrix.size:,}")

    # Test packing/unpacking
    start_time = time.time()
    packed = pack_lower_triangle(cov_matrix)
    packing_time = time.time() - start_time

    start_time = time.time()
    reconstructed = unpack_lower_triangle(packed, n_assets)
    unpacking_time = time.time() - start_time

    # Validate
    is_symmetric = validate_symmetric(reconstructed)
    max_error = np.max(np.abs(cov_matrix - reconstructed))

    print(f"   Packed elements: {len(packed):,} ({len(packed)/cov_matrix.size*100:.1f}% of original)")
    print(f"   Size reduction: {(1 - len(packed)/cov_matrix.size)*100:.1f}%")
    print(f"   Packing time: {packing_time*1000:.3f} ms")
    print(f"   Unpacking time: {unpacking_time*1000:.3f} ms")
    print(f"   Reconstruction error: {max_error:.2e}")
    print(f"   Is symmetric: {is_symmetric}")

    # Memory savings calculation
    original_memory = cov_matrix.size * 8  # 8 bytes per float64
    packed_memory = len(packed) * 8
    memory_saved = original_memory - packed_memory

    print(f"   Memory usage:")
    print(f"     Original: {original_memory / 1024:.1f} KB")
    print(f"     Packed: {packed_memory / 1024:.1f} KB")
    print(f"     Saved: {memory_saved / 1024:.1f} KB ({memory_saved/original_memory*100:.1f}%)")


def demonstrate_autoencoder_training(n_assets: int = 50, n_samples: int = 2000) -> None:
    """Demonstrate autoencoder training with synthetic data."""
    print(f"\n🤖 Demonstrating autoencoder training ({n_assets} assets, {n_samples} samples)...")

    # Create autoencoder with conservative settings
    autoencoder = ImprovedAutoencoderCovarianceTransformer(
        n_assets=n_assets,
        hidden_dims=[32, 16, 8],  # Small architecture for demo
        epochs=20,  # Quick training for demo
        batch_size=16,
        learning_rate=0.01,
        use_lower_triangle=True,
        validation_split=0.2,
        patience=5,
    )

    # Train with synthetic data
    print(f"\n🎯 Training autoencoder...")
    start_time = time.time()

    autoencoder.fit(
        historical_prices=None,  # Not needed for synthetic training
        use_synthetic=True,
        n_synthetic_samples=n_samples,
    )

    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")

    # Test reconstruction on a synthetic example
    print(f"\n🔍 Testing reconstruction quality...")

    # Generate a test covariance matrix
    np.random.seed(123)
    A = np.random.randn(n_assets, n_assets)
    test_cov = A @ A.T
    test_cov += 0.1 * np.eye(n_assets)  # Ensure positive definite

    # Transform (denoise) the matrix
    denoised_cov = autoencoder.transform(test_cov, n_observations=100)

    # Get reconstruction metrics
    metrics = autoencoder.get_reconstruction_metrics()

    print(f"   Reconstruction metrics:")
    for key, value in metrics.items():
        if not np.isnan(value):
            print(f"     {key}: {value:.6f}")

    # Visualize reconstruction quality
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(test_cov, cmap="RdBu_r", aspect="auto")
    plt.title("Original Matrix")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(denoised_cov, cmap="RdBu_r", aspect="auto")
    plt.title("Denoised Matrix")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    diff = test_cov - denoised_cov
    plt.imshow(diff, cmap="RdBu_r", aspect="auto")
    plt.title("Difference")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("autoencoder_reconstruction_demo.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print analysis summary
    autoencoder.print_analysis_summary()


def run_comprehensive_demo():
    """Run comprehensive demonstration of the integrated system."""
    print("=" * 80)
    print("🎯 SYNTHETIC TRAINING DATA + AUTOENCODER INTEGRATION DEMO")
    print("=" * 80)

    print("This demonstration shows how the training_data module provides")
    print("realistic synthetic covariance matrices for autoencoder training.")
    print("Even with synthetic data, the fundamental sample complexity")
    print("challenges remain for high-dimensional problems.\n")

    # Step 1: Generate demo training data
    n_assets_demo = 50  # Manageable size for demonstration
    n_samples_demo = 3000

    data, metadata = generate_demo_training_data(n_assets_demo, n_samples_demo)

    # Step 2: Analyze eigenvalue diversity
    demonstrate_eigenvalue_diversity(data)

    # Step 3: Show lower triangle optimization
    demonstrate_lower_triangle_optimization(n_assets_demo)

    # Step 4: Train autoencoder
    demonstrate_autoencoder_training(n_assets_demo, n_samples_demo)

    print("\n" + "=" * 80)
    print("🎓 EDUCATIONAL CONCLUSIONS")
    print("=" * 80)
    print("1. ✅ Synthetic data generation works excellently")
    print("2. ✅ Lower triangle optimization reduces dimensions by ~50%")
    print("3. ✅ Autoencoder can be trained on synthetic covariance data")
    print("4. ⚠️ Sample complexity remains the fundamental limitation")
    print("5. 💡 For production: Use traditional methods (Oracle, Shrinkage, PCA)")
    print("6. 🧪 For research: This framework enables covariance ML exploration")


if __name__ == "__main__":
    # Configure matplotlib for better plots
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Run the demo
    run_comprehensive_demo()
