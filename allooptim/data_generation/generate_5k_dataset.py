"""Generate Manageable 5K Covariance Training Dataset
Creates 5,000 diverse covariance matrices as a starting point
"""

import pickle
import time
from pathlib import Path

import numpy as np

# Import our modules
from diverse_covariance_generator import CovarianceConfig, CovarianceMatrixGenerator
from lower_triangle_utils import pack_lower_triangle


def main():
    """Generate 5K covariance dataset for initial testing"""
    print("=" * 60)
    print("GENERATING 5,000 COVARIANCE MATRIX TRAINING DATASET")
    print("=" * 60)

    # More manageable configuration
    config = CovarianceConfig(
        n_assets=200,  # 200x200 matrices (manageable size)
        n_samples=5000,  # 5K samples (reasonable for testing)
        random_seed=42,
    )

    print("Configuration:")
    print(f"  Matrix size: {config.n_assets}×{config.n_assets}")
    print(f"  Total samples: {config.n_samples:,}")
    expected_memory_gb = (config.n_samples * config.n_assets * (config.n_assets + 1) // 2 * 8) / (1024**3)
    print(f"  Expected memory: ~{expected_memory_gb:.2f} GB")

    # Generate dataset
    print("\nStarting generation...")
    start_time = time.time()

    generator = CovarianceMatrixGenerator(config)
    covariance_matrices = generator.generate_diverse_training_set()

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.1f} seconds ({generation_time/60:.1f} minutes)")

    # Convert to lower triangle format
    print("\nConverting to lower triangle format...")
    conversion_start = time.time()

    packed_matrices = []
    for i, cov_matrix in enumerate(covariance_matrices):
        if i % 1000 == 0:
            print(f"  Progress: {i:,}/{len(covariance_matrices):,}")

        packed = pack_lower_triangle(cov_matrix)
        packed_matrices.append(packed)

    conversion_time = time.time() - conversion_start
    X_train = np.array(packed_matrices)

    print(f"Conversion completed in {conversion_time:.1f} seconds")

    # Statistics
    print("\nDataset statistics:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Size reduction: {(1 - X_train.shape[1] / (config.n_assets**2)) * 100:.1f}%")
    print(f"  Memory usage: {X_train.nbytes / (1024**3):.2f} GB")
    print(f"  Data range: [{np.min(X_train):.4f}, {np.max(X_train):.4f}]")

    # Save dataset
    output_dir = Path(__file__).parent.parent.parent / "generated_output"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "covariance_training_data_5k.pkl"

    dataset = {
        "X_train": X_train,
        "config": config,
        "generation_time": generation_time,
        "conversion_time": conversion_time,
        "metadata": {
            "n_matrices": len(covariance_matrices),
            "matrix_size": config.n_assets,
            "packed_size": X_train.shape[1],
            "memory_gb": X_train.nbytes / (1024**3),
        },
    }

    print(f"\nSaving dataset to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved! File size: {Path(save_path).stat().st_size / (1024**2):.1f} MB")

    total_time = generation_time + conversion_time
    print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("Ready for autoencoder training!")

    return X_train, dataset
