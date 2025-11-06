"""Lower Triangle Matrix Utilities for Symmetric Covariance Matrices
Optimizes autoencoder input/output by using only lower triangle elements.
"""

import numpy as np


def pack_lower_triangle(matrix: np.array) -> np.array:
    """Pack a symmetric matrix into a 1D array containing only lower triangle elements.

    For an n×n symmetric matrix, this reduces storage from n² to n(n+1)/2 elements.
    For 500×500 matrix: 250,000 → 125,250 (50% reduction)

    Args:
        matrix: Symmetric n×n matrix

    Returns:
        1D array of lower triangle elements in row-major order

    Example:
        [[1, 2, 3],      →     [1, 2, 5, 3, 6, 9]
         [2, 5, 6],            (elements: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2))
         [3, 6, 9]]
    """
    # Ensure matrix is numpy array
    matrix = np.asarray(matrix)
    n = matrix.shape[0]

    # Validate square matrix
    if matrix.shape != (n, n):
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}")

    # Extract lower triangle indices
    row_indices, col_indices = np.tril_indices(n)

    # Pack lower triangle elements
    packed = matrix[row_indices, col_indices]

    return packed


def unpack_lower_triangle(packed: np.array, n: int) -> np.array:
    """Unpack a 1D array of lower triangle elements back to symmetric n×n matrix.

    Reconstructs the full symmetric matrix and applies (A + A.T)/2 to ensure symmetry.

    Args:
        packed: 1D array of lower triangle elements
        n: Dimension of target square matrix

    Returns:
        Symmetric n×n matrix

    Example:
        [1, 2, 5, 3, 6, 9], n=3  →  [[1, 2, 3],
                                     [2, 5, 6],
                                     [3, 6, 9]]
    """
    # Validate input size
    expected_size = n * (n + 1) // 2
    if len(packed) != expected_size:
        raise ValueError(f"Packed array size {len(packed)} doesn't match expected {expected_size} for n={n}")

    # Initialize matrix
    matrix = np.zeros((n, n))

    # Fill lower triangle (including diagonal)
    row_indices, col_indices = np.tril_indices(n)
    matrix[row_indices, col_indices] = packed

    # Copy lower triangle to upper triangle (excluding diagonal)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))

    # Ensure perfect symmetry with (A + A.T) / 2
    matrix = (matrix + matrix.T) / 2

    return matrix


def get_packed_size(n: int) -> int:
    """Get the size of packed lower triangle for n×n matrix."""
    return n * (n + 1) // 2


def validate_symmetric(matrix: np.array, tolerance: float = 1e-10) -> bool:
    """Check if a matrix is symmetric within tolerance.

    Args:
        matrix: Matrix to check
        tolerance: Maximum allowed difference for symmetry

    Returns:
        True if symmetric, False otherwise
    """
    return np.allclose(matrix, matrix.T, atol=tolerance)


def test_packing_functions():
    """Test the packing/unpacking functions with various matrix sizes."""
    print("Testing Lower Triangle Packing Functions")
    print("=" * 50)

    # Test with small matrix
    test_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]])

    print("Original 3x3 matrix:")
    print(test_matrix)

    # Pack and unpack
    packed = pack_lower_triangle(test_matrix)
    print(f"\nPacked lower triangle ({len(packed)} elements):")
    print(packed)

    reconstructed = unpack_lower_triangle(packed, 3)
    print("\nReconstructed matrix:")
    print(reconstructed)

    # Check symmetry and accuracy
    is_symmetric = validate_symmetric(reconstructed)
    is_accurate = np.allclose(test_matrix, reconstructed)

    print("\nValidation:")
    print(f"  Symmetric: {is_symmetric}")
    print(f"  Accurate reconstruction: {is_accurate}")
    print(f"  Max error: {np.max(np.abs(test_matrix - reconstructed)):.2e}")

    # Test with 500x500 size calculation
    n_500 = 500
    full_size = n_500 * n_500
    packed_size = get_packed_size(n_500)
    reduction = (1 - packed_size / full_size) * 100

    print("\n500x500 Matrix Optimization:")
    print(f"  Full matrix elements: {full_size:,}")
    print(f"  Packed elements: {packed_size:,}")
    print(f"  Size reduction: {reduction:.1f}%")
    print(f"  Memory savings: {(full_size - packed_size) * 8 / 1024 / 1024:.1f} MB (float64)")
