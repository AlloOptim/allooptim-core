"""Generate Full 30K Covariance Training Dataset
Creates 30,000 diverse covariance matrices and saves them for autoencoder training
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Add the enhanced allocation directory to path
sys.path.append(str(Path(__file__).parent))

from diverse_covariance_generator import CovarianceConfig, CovarianceMatrixGenerator
from lower_triangle_utils import pack_lower_triangle

# Constants for numerical validation
EIGENVALUE_POSITIVE_TOLERANCE = -1e-10


def generate_full_training_dataset(save_path: str = None):
    """Generate full 30,000 covariance matrix training dataset"""
    print("=" * 60)
    print("GENERATING 30,000 COVARIANCE MATRIX TRAINING DATASET")
    print("=" * 60)

    # Configuration for full dataset
    config = CovarianceConfig(
        n_assets=500,  # 500x500 covariance matrices
        n_samples=30000,  # 30,000 total samples
        random_seed=42,
        # Diverse generation methods
        pct_synthetic=0.50,  # 50% synthetic eigenvalue-based
        pct_gan_style=0.25,  # 25% GAN-style diverse
        pct_real_based=0.15,  # 15% real data bootstrap
        pct_block_struct=0.10,  # 10% block-structured
    )

    print("Configuration:")
    print(f"  Matrix size: {config.n_assets}×{config.n_assets}")
    print(f"  Total samples: {config.n_samples:,}")
    print("  Method distribution:")
    print(
        f"    - Synthetic eigenvalue: {config.pct_synthetic:.0%} "
        f"({int(config.pct_synthetic * config.n_samples):,} matrices)"
    )
    print(
        f"    - GAN-style diverse:    {config.pct_gan_style:.0%} "
        f"({int(config.pct_gan_style * config.n_samples):,} matrices)"
    )
    print(
        f"    - Real data bootstrap:  {config.pct_real_based:.0%} "
        f"({int(config.pct_real_based * config.n_samples):,} matrices)"
    )
    n_block_struct = (
        config.n_samples
        - int(config.pct_synthetic * config.n_samples)
        - int(config.pct_gan_style * config.n_samples)
        - int(config.pct_real_based * config.n_samples)
    )
    print(f"    - Block-structured:     {config.pct_block_struct:.0%} ({n_block_struct:,} matrices)")

    # Start generation
    print("\nStarting generation...")
    start_time = time.time()

    generator = CovarianceMatrixGenerator(config)
    covariance_matrices = generator.generate_diverse_training_set()

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.1f} seconds")
    print(f"Generation rate: {len(covariance_matrices) / generation_time:.1f} matrices/second")

    # Convert to lower triangle format
    print("\nConverting to lower triangle format...")
    conversion_start = time.time()

    packed_matrices = []
    for i, cov_matrix in enumerate(covariance_matrices):
        if i % 5000 == 0:
            print(f"  Progress: {i:,}/{len(covariance_matrices):,} ({i/len(covariance_matrices)*100:.1f}%)")

        packed = pack_lower_triangle(cov_matrix)
        packed_matrices.append(packed)

    conversion_time = time.time() - conversion_start
    print(f"Conversion completed in {conversion_time:.1f} seconds")

    # Convert to numpy array
    X_train = np.array(packed_matrices)

    print("\nDataset statistics:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Original matrix size: {config.n_assets}×{config.n_assets} = {config.n_assets**2:,} elements")
    print(f"  Packed size: {X_train.shape[1]:,} elements")
    print(f"  Size reduction: {(1 - X_train.shape[1] / (config.n_assets**2)) * 100:.1f}%")
    print(f"  Memory usage: {X_train.nbytes / (1024**3):.2f} GB")
    print(f"  Data range: [{np.min(X_train):.4f}, {np.max(X_train):.4f}]")
    print(f"  Mean: {np.mean(X_train):.6f}")
    print(f"  Std: {np.std(X_train):.6f}")

    # Validate a few matrices
    print("\nValidating random samples...")
    test_indices = np.random.choice(len(covariance_matrices), 5, replace=False)
    all_valid = True

    for idx in test_indices:
        matrix = covariance_matrices[idx]
        eigenvals = np.linalg.eigvals(matrix)

        is_real = np.all(np.isreal(eigenvals))
        is_positive = np.all(eigenvals > EIGENVALUE_POSITIVE_TOLERANCE)
        is_symmetric = np.allclose(matrix, matrix.T)
        is_valid = is_real and is_positive and is_symmetric

        if not is_valid:
            all_valid = False
            print(f"  Matrix {idx}: INVALID (real={is_real}, pos={is_positive}, sym={is_symmetric})")
        else:
            print(f"  Matrix {idx}: VALID")

    if all_valid:
        print("✓ All sampled matrices are valid!")
    else:
        print("⚠ Some matrices failed validation!")

    # Save the dataset
    if save_path is None:
        output_dir = Path(__file__).parent.parent.parent / "generated_output"
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / "covariance_training_data_30k.pkl"

    print(f"\nSaving dataset to: {save_path}")

    dataset = {
        "X_train": X_train,
        "config": config,
        "generation_time": generation_time,
        "conversion_time": conversion_time,
        "total_time": generation_time + conversion_time,
        "metadata": {
            "n_matrices": len(covariance_matrices),
            "matrix_size": config.n_assets,
            "packed_size": X_train.shape[1],
            "size_reduction_pct": (1 - X_train.shape[1] / (config.n_assets**2)) * 100,
            "memory_gb": X_train.nbytes / (1024**3),
            "generation_rate": len(covariance_matrices) / generation_time,
            "validation_passed": all_valid,
        },
    }

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Dataset saved successfully!")
    print(f"File size: {Path(save_path).stat().st_size / (1024**3):.2f} GB")

    # Summary
    total_time = generation_time + conversion_time
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Dataset: {len(covariance_matrices):,} matrices of {config.n_assets}×{config.n_assets}")
    print(f"Training data: {X_train.shape}")
    print(f"Saved to: {save_path}")
    print("Ready for autoencoder training!")

    return X_train, dataset


def load_training_dataset(save_path: str = None):
    """Load the pre-generated training dataset"""
    if save_path is None:
        output_dir = Path(__file__).parent.parent.parent / "generated_output"
        save_path = output_dir / "covariance_training_data_30k.pkl"

    if not Path(save_path).exists():
        raise FileNotFoundError(f"Training dataset not found at: {save_path}")

    print(f"Loading training dataset from: {save_path}")

    with open(save_path, "rb") as f:
        dataset = pickle.load(f)

    X_train = dataset["X_train"]
    metadata = dataset["metadata"]

    print("Loaded dataset:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Matrix size: {metadata['matrix_size']}×{metadata['matrix_size']}")
    print(f"  Generation time: {dataset['generation_time']:.1f}s")
    print(f"  Memory: {metadata['memory_gb']:.2f} GB")
    print(f"  Validation passed: {metadata['validation_passed']}")

    return X_train, dataset
