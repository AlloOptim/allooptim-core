#!/usr/bin/env python3
"""
Data Generation Examples for AlloOptim

This script demonstrates how to use the various data generation utilities
in the allo_optim package to create training datasets for portfolio optimization.

The examples show:
1. Generating synthetic covariance matrices
2. Creating training data for covariance transformers
3. Generating diverse correlation matrices
4. Working with different dataset sizes

Usage:
    python examples/data_generation_examples.py
"""

from pathlib import Path

from allo_optim.data_generation.diverse_covariance_generator import CovarianceConfig, CovarianceMatrixGenerator
from allo_optim.data_generation.diverse_training_data import (
    DiverseCorrelationGenerator,
    DiverseTrainingConfig,
    validate_training_data,
)
from allo_optim.data_generation.generate_30k_dataset import generate_full_training_dataset
from allo_optim.data_generation.training_data import TrainingConfig, TrainingDataGenerator, load_training_data


def example_basic_covariance_generation():
    """Example 1: Generate basic synthetic covariance matrices."""
    print("=" * 60)
    print("Example 1: Basic Covariance Matrix Generation")
    print("=" * 60)

    # Configuration for small test dataset
    config = TrainingConfig(
        n_assets=50,  # 50 assets
        n_samples=100,  # 100 samples for quick testing
        min_observations=100,
        max_observations=500,
        n_processes=2,  # Use 2 processes for parallel generation
        output_file="example_basic_covariance.h5",
        random_seed=42,
    )

    print(f"Generating {config.n_samples} covariance matrices with {config.n_assets} assets...")

    # Create generator and generate data
    generator = TrainingDataGenerator(config)
    samples = generator.generate_parallel(verbose=True)

    # Save to disk
    generator.save_to_hdf5(samples)
    print(f"Saved dataset to {config.output_file}")

    # Load and inspect
    data, metadata = load_training_data(config.output_file)
    print(f"Loaded dataset with {data['sample_eigenvalues'].shape[0]} samples")
    print(".4f")
    print(f"Observations range: [{data['n_observations'].min()}, {data['n_observations'].max()}]")

    return config.output_file


def example_diverse_covariance_generation():
    """Example 2: Generate diverse covariance matrices with various structures."""
    print("\n" + "=" * 60)
    print("Example 2: Diverse Covariance Matrix Generation")
    print("=" * 60)

    # Configuration for diverse matrices
    config = CovarianceConfig(
        n_samples=50,  # Smaller for example
        n_assets=30,  # Smaller matrices
        random_seed=123,
    )

    print(f"Generating {config.n_samples} diverse covariance matrices...")

    generator = CovarianceMatrixGenerator(config)
    training_data = generator.generate_diverse_training_set()

    print(f"Generated {len(training_data)} covariance matrices")
    print(f"Matrix shapes: {training_data[0].shape}")


    return training_data


def example_large_dataset_generation():
    """Example 3: Generate a large training dataset (30k samples)."""
    print("\n" + "=" * 60)
    print("Example 3: Large Dataset Generation (30k samples)")
    print("=" * 60)

    print("Generating large training dataset...")
    print("Note: This may take several minutes on slower machines...")

    try:
        # Generate the full dataset
        X_train, dataset = generate_full_training_dataset()

        print(f"Generated dataset with {X_train.shape[0]} samples")
        print(f"Feature dimensions: {X_train.shape[1]}")
        print(f"Dataset keys: {list(dataset.keys())}")

        # Show some statistics
        if "q_values" in dataset:
            print(".4f")
            print(".4f")

    except Exception as e:
        print(f"Large dataset generation failed (expected on some systems): {e}")
        print("This requires significant computational resources.")


def example_correlation_matrix_generation():
    """Example 4: Generate diverse correlation matrices for training."""
    print("\n" + "=" * 60)
    print("Example 4: Diverse Correlation Matrix Generation")
    print("=" * 60)

    # Configuration for correlation matrices
    config = DiverseTrainingConfig(
        n_samples=25,  # Small test set
        n_assets=20,  # Smaller matrices for testing
        random_seed=456,
    )

    print(f"Generating {config.n_samples} diverse correlation matrices...")

    generator = DiverseCorrelationGenerator(config)
    training_data = generator.generate_diverse_training_set()

    # Validate the generated data
    validation_stats = validate_training_data(training_data)

    print("\nValidation Results:")
    print("-" * 30)
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

    return training_data


def main():
    """Run all data generation examples."""
    print("AlloOptim Data Generation Examples")
    print("===================================")

    # Create examples directory if it doesn't exist
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    # Run examples
    try:
        # Example 1: Basic covariance generation
        basic_file = example_basic_covariance_generation()

        # Example 2: Diverse covariance generation
        diverse_data = example_diverse_covariance_generation()

        # Example 3: Large dataset (may be slow)
        example_large_dataset_generation()

        # Example 4: Correlation matrices
        correlation_data = example_correlation_matrix_generation()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print(f"  - {basic_file}")
        print("\nGenerated data in memory:")
        print(f"  - {len(diverse_data)} diverse covariance matrices")
        print(f"  - {len(correlation_data)} correlation matrices")

        print("\nNext steps:")
        print("  - Use the generated data to train covariance transformers")
        print("  - Experiment with different configurations")
        print("  - Integrate into your portfolio optimization pipeline")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
